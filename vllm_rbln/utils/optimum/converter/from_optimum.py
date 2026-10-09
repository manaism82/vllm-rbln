# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, Any

from vllm_rbln import envs
from vllm_rbln.logger import init_logger
from vllm_rbln.utils.optimum.block_size import (
    get_block_ratio,
    is_full_block_available,
)

from .common import (
    apply_user_prefill_chunk_size,
    store_image_prefill_chunk_size,
    update_block_size,
    update_max_num_batched_tokens,
)
from .params import RBLNParams

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

logger = init_logger(__name__)


# rbln_config keys a user may still override for a precompiled model. They
# only decide how the artefact is loaded and run on the host, never what was
# compiled: ``device`` places a (sub)module on NPUs, ``pos_embed_cache_size``
# bounds the host cache of the Qwen3.5 vision encoder's interpolated position
# embeddings (an optimum-rbln load-time option, not saved in rbln_config.json).
_LOAD_TIME_KEYS = frozenset({"device", "pos_embed_cache_size"})


def _keep_only_load_time_keys(obj: dict) -> dict:
    """Recursively keep only ``_LOAD_TIME_KEYS`` entries from nested dicts."""
    result: dict[str, Any] = {}
    for k, v in obj.items():
        if k in _LOAD_TIME_KEYS:
            result[k] = v
        elif isinstance(v, dict):
            filtered = _keep_only_load_time_keys(v)
            if filtered:
                result[k] = filtered
    return result


def sync_from_optimum(
    vllm_config: VllmConfig,
    compiled_rbln_config: dict,
) -> None:
    """
    optimum -> vLLM config synchronization
    """

    params = RBLNParams.from_rbln_config(vllm_config, compiled_rbln_config)

    # The compiled artefact is the source of truth. Strip the user's
    # additional_config down to load-time keys so submodule placement (and
    # other host-side load options) can still be set, but no compile
    # parameter sneaks in.
    vllm_config.additional_config["rbln_config"] = _keep_only_load_time_keys(
        vllm_config.additional_config.get("rbln_config", {})
    )

    assert params.num_blocks is not None, (
        "num_blocks must be specified in rbln_config.json"
    )
    assert params.batch_size is not None, (
        "batch_size must be specified in rbln_config.json"
    )
    assert params.max_seq_len is not None, (
        "max_seq_len must be specified in rbln_config.json"
    )
    assert params.kvcache_block_size is not None, (
        "kvcache_block_size must be specified in rbln_config.json"
    )

    # Set max_num_seqs in scheduler_config based on rbln_config.json
    if vllm_config.scheduler_config.max_num_seqs != params.batch_size:
        logger.info(
            "Updating scheduler_config.max_num_seqs from %s to %s "
            "based on rbln_config.json",
            vllm_config.scheduler_config.max_num_seqs,
            params.batch_size,
        )
        vllm_config.scheduler_config.max_num_seqs = params.batch_size

    # Set max_model_len in model_config based on rbln_config.json
    if vllm_config.model_config.max_model_len != params.max_seq_len:
        logger.info(
            "Updating model_config.max_model_len "
            "from %s to %s "
            "based on rbln_config.json",
            vllm_config.model_config.max_model_len,
            params.max_seq_len,
        )
        vllm_config.model_config.max_model_len = params.max_seq_len

    apply_user_prefill_chunk_size(vllm_config, params, precompiled=True)
    # Set max_num_batched_tokens: the prefill chunk size for decoder/multimodal
    # models, or a full-prefill-plus-batch budget for enc-dec/pooling models.
    update_max_num_batched_tokens(vllm_config, params)
    update_mamba_block_size(vllm_config, params)

    # Persist the image-prefill buckets (gemma3/gemma4) into additional_config.
    store_image_prefill_chunk_size(vllm_config, params.image_prefill_chunk_size)
    # Set block_size in cache_config based on rbln_config.json.
    update_block_size(
        vllm_config,
        params.kvcache_block_size,
        params.prefill_chunk_size,
        params.image_prefill_chunk_size,
    )
    # Set num_blocks in cache_config based on rbln_config.json
    update_num_blocks(vllm_config, params.num_blocks)
    # Sync num_devices in envs with optimum pre-compiled model
    envs.VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK = params.num_devices


def update_num_blocks(vllm_config: VllmConfig, num_blocks: int) -> None:
    # This function is called twice during startup: once in the main process
    # when we first read rbln_config.json, and once in each subprocess when
    # the VllmConfig is deserialized. We only want to perform the num_blocks
    # calculation and update once, in the main process, to avoid redundant
    # calculations and potential inconsistencies. We use an additional_config
    # flag to track whether we have already synced num_blocks.
    if vllm_config.additional_config.get("num_blocks_synced", False):
        logger.debug(
            "num_blocks already synced to %s, skipping...",
            vllm_config.cache_config.num_gpu_blocks,
        )
        return
    # num_blocks is determined by rbln_config or overridden by user.
    if vllm_config.cache_config.num_gpu_blocks_override is not None:
        num_blocks = vllm_config.cache_config.num_gpu_blocks_override
        # This is kept for optimum based num blocks
        # not considering ob-ib logic for prefix caching
        vllm_config.additional_config["num_blocks_override"] = num_blocks
    blk_ratio = get_block_ratio(vllm_config)

    if is_full_block_available(num_blocks, vllm_config):
        adjusted_num_blocks = num_blocks * blk_ratio + 1
    else:
        adjusted_num_blocks = (num_blocks - 1) * blk_ratio + 1

    vllm_config.cache_config.num_gpu_blocks = adjusted_num_blocks

    if vllm_config.cache_config.num_gpu_blocks_override is not None:
        vllm_config.cache_config.num_gpu_blocks_override = adjusted_num_blocks
    vllm_config.additional_config["num_blocks_synced"] = True


def update_mamba_block_size(vllm_config: VllmConfig, params: "RBLNParams") -> None:
    cache_config = vllm_config.cache_config
    if (
        not cache_config.user_specified_mamba_block_size
        and cache_config.mamba_block_size is not None
    ):
        # The compiled graph keeps one linear-attention state row per sequence,
        # not vLLM's per-block mamba states ('align'/'all' cache modes). A hybrid
        # prefix is instead resumed from the artifact's linear-state snapshot
        # rows (see LinearStateSnapshotPool), so vLLM's mamba bookkeeping stays
        # at one block per sequence either way.
        if cache_config.enable_prefix_caching:
            if params.linear_state_snapshot_slots > 0:
                vllm_config.additional_config["linear_state_snapshot_slots"] = (
                    params.linear_state_snapshot_slots
                )
                # vLLM's MambaModelConfig has already logged that prefix
                # caching switched this model to the experimental 'align' mode.
                logger.info(
                    "Qwen3.5 snapshot prefix caching: K=%d slots, "
                    "mamba_cache_mode reset to none (the earlier vLLM "
                    "align/experimental lines do not apply).",
                    params.linear_state_snapshot_slots,
                )
            else:
                logger.warning(
                    "Prefix caching is disabled: this hybrid (linear-attention) "
                    "model was compiled without linear-state snapshot rows. "
                    "Recompile with rbln_config linear_state_snapshot_slots > 0 "
                    "to use it."
                )
                cache_config.enable_prefix_caching = False
        cache_config.mamba_cache_mode = "none"
        cache_config.mamba_block_size = params.max_seq_len
