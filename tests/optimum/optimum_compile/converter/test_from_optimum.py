# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace
from unittest.mock import patch

from vllm_rbln.utils.optimum.converter import from_optimum
from vllm_rbln.utils.optimum.converter.params import RBLNParams

MAX_SEQ_LEN = 65536


def _hybrid_vllm_config(enable_prefix_caching: bool) -> SimpleNamespace:
    """A hybrid model's config after vLLM's MambaModelConfig: with prefix
    caching it switched to the per-block 'align' mamba cache mode."""
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            enable_prefix_caching=enable_prefix_caching,
            user_specified_mamba_block_size=False,
            mamba_block_size=128 if enable_prefix_caching else MAX_SEQ_LEN,
            mamba_cache_mode="align" if enable_prefix_caching else "none",
        ),
        additional_config={},
    )


class TestUpdateMambaBlockSize:
    def test_snapshot_slots_keep_prefix_caching(self):
        vllm_config = _hybrid_vllm_config(enable_prefix_caching=True)

        with patch.object(from_optimum, "logger") as logger:
            from_optimum.update_mamba_block_size(
                vllm_config,
                RBLNParams(max_seq_len=MAX_SEQ_LEN, linear_state_snapshot_slots=8),
            )

        cache_config = vllm_config.cache_config
        assert cache_config.enable_prefix_caching
        assert cache_config.mamba_cache_mode == "none"
        assert cache_config.mamba_block_size == MAX_SEQ_LEN
        assert vllm_config.additional_config == {"linear_state_snapshot_slots": 8}
        # Operators read this line against vLLM's earlier 'align' warning.
        logger.info.assert_called_once()
        message = logger.info.call_args.args[0] % logger.info.call_args.args[1:]
        assert message.startswith("Qwen3.5 snapshot prefix caching: K=8 slots,")
        assert "mamba_cache_mode reset to none" in message
        logger.warning.assert_not_called()

    def test_no_snapshot_slots_disables_prefix_caching_with_a_warning(self):
        vllm_config = _hybrid_vllm_config(enable_prefix_caching=True)

        with patch.object(from_optimum, "logger") as logger:
            from_optimum.update_mamba_block_size(
                vllm_config, RBLNParams(max_seq_len=MAX_SEQ_LEN)
            )

        cache_config = vllm_config.cache_config
        assert not cache_config.enable_prefix_caching
        assert cache_config.mamba_cache_mode == "none"
        assert cache_config.mamba_block_size == MAX_SEQ_LEN
        assert vllm_config.additional_config == {}
        logger.warning.assert_called_once()
        logger.info.assert_not_called()

    def test_prefix_caching_off_publishes_no_slots(self):
        vllm_config = _hybrid_vllm_config(enable_prefix_caching=False)

        from_optimum.update_mamba_block_size(
            vllm_config,
            RBLNParams(max_seq_len=MAX_SEQ_LEN, linear_state_snapshot_slots=8),
        )

        assert not vllm_config.cache_config.enable_prefix_caching
        assert vllm_config.additional_config == {}


class TestKeepOnlyLoadTimeKeys:
    def test_keeps_placement_and_vision_cache_size_only(self):
        overrides = {
            "device": [0, 1],
            "max_seq_len": 4096,
            "visual": {
                "device": [2],
                "pos_embed_cache_size": 4,
                "max_seq_len": [8192],
            },
            "language_model": {"batch_size": 8},
        }

        assert from_optimum._keep_only_load_time_keys(overrides) == {
            "device": [0, 1],
            "visual": {"device": [2], "pos_embed_cache_size": 4},
        }
