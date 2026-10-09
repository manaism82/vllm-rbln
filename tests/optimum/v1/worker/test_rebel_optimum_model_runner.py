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

import tempfile
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.config import (
    CacheConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.platforms import current_platform
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.core.sched.output import CachedRequestData
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_rbln.model_executor.models.optimum.base import LinearStateRestoreError
from vllm_rbln.v1.core.optimum_scheduler import RBLNSchedulerOutput
from vllm_rbln.v1.core.prefix_cache_manager import LinearStateSnapshot
from vllm_rbln.v1.worker import optimum_model_runner
from vllm_rbln.v1.worker.optimum_model_runner import RBLNOptimumModelRunner

from ..models.optimum.test_qwen3_5_linear_state import (
    KV_NAMES,
    _FakeKVRuntime,
    _snapshot_qwen3_5,
)
from .utils import (
    _schedule_new_request,
    _schedule_new_request_from_request,
    create_model_runner,
    fake_load_model,
    make_request,
)

BLOCK_SIZE = 16
NUM_BLOCKS = 8
DEVICE = current_platform.device_type


# TODO add tests for both `enable_prefix_caching = True` and `False`
def get_vllm_config(async_scheduling=False):
    scheduler_config = SchedulerConfig(
        max_num_seqs=10,
        max_num_batched_tokens=128,
        max_model_len=128,
        async_scheduling=async_scheduling,
        is_encoder_decoder=False,
    )
    model_config = ModelConfig(
        model="facebook/opt-125m",
        dtype=torch.float,
        seed=42,
    )
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE,
        cache_dtype="auto",
    )
    vllm_config = VllmConfig(
        cache_config=cache_config,
        model_config=model_config,
        scheduler_config=scheduler_config,
        additional_config={
            "prefix_block_size": 4,
            "rbln_config": {
                "prefill_chunk_size": 4,
            },
        },
    )
    return vllm_config


@pytest.fixture
def model_runner():
    vllm_config = get_vllm_config()
    with set_current_vllm_config(vllm_config, check_compile=False):
        temp_file = tempfile.mkstemp()[1]
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"file://{temp_file}",
            backend="gloo",
        )
        ensure_model_parallel_initialized(
            1,
            1,
        )
    runner = RBLNOptimumModelRunner(vllm_config, DEVICE)
    fake_load_model(runner)
    return runner


def _is_req_scheduled(model_runner, req_id: str) -> bool:
    return req_id in model_runner.input_batch.req_id_to_index


def _is_req_added(model_runner, req_id: str) -> bool:
    return req_id in model_runner.requests


def _is_sampling_metadata_changed(
    model_runner, sampling_metadata_before: SamplingMetadata
):
    return model_runner.input_batch.sampling_metadata is not (sampling_metadata_before)


def _is_req_state_block_table_match(model_runner, req_id: str) -> bool:
    req_index = model_runner.input_batch.req_id_to_index[req_id]
    block_table = model_runner.input_batch.block_table[0]
    req_state = model_runner.requests[req_id]

    num_block_of_runner = block_table.num_blocks_per_row[req_index]
    num_block_of_req_state = len(req_state.block_ids[0])
    if num_block_of_runner != num_block_of_req_state:
        return False
    return (
        block_table.block_table.np[req_index, :num_block_of_runner]
        == req_state.block_ids[0]
    ).all()


def test_update_states_new_request(model_runner):
    req_id = "req_0"

    # schedule new request
    scheduler_output = _schedule_new_request(
        req_id, block_ids=([0],), outer_block_ids=[0]
    )
    metadata_before = model_runner.input_batch.sampling_metadata
    model_runner._update_states(scheduler_output)
    assert _is_sampling_metadata_changed(model_runner, metadata_before)
    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)
    assert _is_req_state_block_table_match(model_runner, req_id)


def test_update_states_request_finished(model_runner):
    req_id = "req_0"

    # schedule new request
    scheduler_output = _schedule_new_request(
        req_id, block_ids=([0],), outer_block_ids=[0]
    )

    model_runner._update_states(scheduler_output)
    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)

    # finish request
    scheduler_output = RBLNSchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=0,
        finished_req_ids={req_id},
        free_encoder_mm_hashes=[],
    )

    metadata_before = model_runner.input_batch.sampling_metadata
    model_runner._update_states(scheduler_output)
    assert _is_sampling_metadata_changed(model_runner, metadata_before)
    assert not _is_req_added(model_runner, req_id)
    assert not _is_req_scheduled(model_runner, req_id)


def test_update_states_request_resumed(model_runner):
    req_id = "req_0"

    # schedule new request
    scheduler_output = _schedule_new_request(
        req_id, block_ids=([0],), outer_block_ids=[0]
    )

    model_runner._update_states(scheduler_output)
    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)

    # unschedule request
    scheduler_output = RBLNSchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=0,
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    model_runner._update_states(scheduler_output)
    assert _is_req_added(model_runner, req_id)
    assert not _is_req_scheduled(model_runner, req_id)

    # resume request
    cached_req_data = CachedRequestData(
        req_ids=[req_id],
        resumed_req_ids=set(),
        new_token_ids=[],
        all_token_ids={},
        new_block_ids=[([0],)],
        num_computed_tokens=[0],
        num_output_tokens=[0],
    )

    scheduler_output = RBLNSchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached_req_data,
        num_scheduled_tokens={req_id: 1},
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=0,
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    metadata_before = model_runner.input_batch.sampling_metadata
    model_runner._update_states(scheduler_output)
    assert _is_sampling_metadata_changed(model_runner, metadata_before)
    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)
    assert _is_req_state_block_table_match(model_runner, req_id)


def test_update_states_request_unscheduled(model_runner):
    req_id = "req_0"

    # schedule req0
    scheduler_output = _schedule_new_request(
        req_id, block_ids=([0],), outer_block_ids=[0]
    )

    model_runner._update_states(scheduler_output)

    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)

    new_req_id = "req_1"

    # schedule req1
    # scheduling new request(req1)
    # prevent req0 from being scheduled
    scheduler_output = _schedule_new_request(
        new_req_id, block_ids=([1],), outer_block_ids=torch.tensor([[1]])
    )

    metadata_before = model_runner._update_states(scheduler_output)
    assert _is_sampling_metadata_changed(model_runner, metadata_before)

    assert _is_req_added(model_runner, req_id)
    assert not _is_req_scheduled(model_runner, req_id)

    assert _is_req_added(model_runner, new_req_id)
    assert _is_req_scheduled(model_runner, new_req_id)


@pytest.mark.parametrize(
    ("rbln_config", "sampler_device"),
    [
        pytest.param(
            {"device": list(range(8, 16)), "visual": {"device": [12, 13]}},
            8,
            id="device-list",
        ),
        pytest.param({"device": 5}, 5, id="device-int"),
        pytest.param({}, 0, id="device-absent"),
        pytest.param(
            {"language_model": {"device": [4, 5]}, "device": [0, 1]},
            4,
            id="language-model-device",
        ),
        pytest.param(
            {"text_model": {"device": 6}, "device": [0, 1]},
            6,
            id="text-model-device",
        ),
    ],
)
def test_rbln_sampler_runs_on_the_first_language_model_device(
    monkeypatch, rbln_config, sampler_device
):
    monkeypatch.setenv("VLLM_RBLN_SAMPLER", "1")
    sampler_cls = Mock()
    monkeypatch.setattr(optimum_model_runner, "RBLNSampler", sampler_cls)
    vllm_config = get_vllm_config()
    vllm_config.additional_config["rbln_config"] = rbln_config

    RBLNOptimumModelRunner(vllm_config, DEVICE)

    assert sampler_cls.call_args.kwargs["device_id"] == sampler_device


RESTORE = LinearStateSnapshot(slot=0, boundary=4, generation=7)
CAPTURE = LinearStateSnapshot(slot=1, boundary=8, generation=8)


def _hybrid_prefix_hit(monkeypatch):
    """A prefix-caching runner and a 10-token prefill that resumes from
    RESTORE (outer block 3) and fills CAPTURE."""
    monkeypatch.setenv("VLLM_RBLN_SAMPLER", "0")
    init_none_hash(sha256)
    runner = create_model_runner()
    request = make_request("req_0", prompt_token_ids=list(range(1, 11)))
    scheduler_output = _schedule_new_request_from_request(
        request, block_ids=([1, 2, 3],), outer_block_ids=[3]
    )
    scheduler_output.cached_length = [RESTORE.boundary]
    scheduler_output.linear_state_restore = RESTORE
    scheduler_output.linear_state_capture = CAPTURE
    return runner, scheduler_output


def test_execute_model_restores_before_and_captures_after_the_prefill(monkeypatch):
    runner, scheduler_output = _hybrid_prefix_hit(monkeypatch)
    events: list[tuple] = []
    forward = runner.model.forward

    def recording_forward(model_input, **kwargs):
        events.append(("forward", model_input.input_tokens.tolist()))
        return forward(model_input, **kwargs)

    runner.model.forward = recording_forward
    runner.model.restore_linear_state_prefix = lambda snapshot, block: events.append(
        ("restore", snapshot, block)
    )
    runner.model.capture_linear_state_prefix = lambda snapshot, block: events.append(
        ("capture", snapshot, block)
    )

    runner.execute_model(scheduler_output)

    assert events == [
        ("restore", RESTORE, 3),
        ("forward", [list(range(5, 11))]),
        ("capture", CAPTURE, 3),
    ]


@pytest.mark.parametrize(
    "capture, captures",
    [
        pytest.param(CAPTURE, (RESTORE, CAPTURE), id="capture_into_another_slot"),
        pytest.param(None, (RESTORE,), id="no_capture"),
        # The scheduler already maps the restored slot to the new capture.
        pytest.param(
            LinearStateSnapshot(slot=0, boundary=8, generation=8),
            (LinearStateSnapshot(slot=0, boundary=8, generation=8),),
            id="capture_into_the_restored_slot",
        ),
    ],
)
def test_failed_restore_prefills_the_whole_prompt(monkeypatch, capture, captures):
    runner, scheduler_output = _hybrid_prefix_hit(monkeypatch)
    scheduler_output.linear_state_capture = capture
    runner._update_states(scheduler_output)
    model_input, _ = runner._prepare_inputs(scheduler_output)
    runner.model.restore_linear_state_prefix = Mock(
        side_effect=LinearStateRestoreError("stale snapshot")
    )

    model_input = runner._restore_linear_state_prefix(model_input, scheduler_output)

    runner.model.restore_linear_state_prefix.assert_called_once_with(RESTORE, 3)
    assert model_input.input_tokens.tolist() == [list(range(1, 11))]
    assert model_input.input_positions.tolist() == [list(range(10))]
    assert model_input.linear_state_restore is None
    assert model_input.linear_state_captures == captures


@pytest.mark.parametrize(
    "held",
    [
        pytest.param(None, id="lost_capture"),
        # The host copy of the prefix the scheduler evicted from slot 0 for
        # RESTORE, whose own capture never reached the worker.
        pytest.param(
            LinearStateSnapshot(slot=0, boundary=12, generation=3),
            id="stale_generation",
        ),
    ],
)
def test_a_slot_that_cannot_be_restored_is_captured_again(monkeypatch, held):
    runner, scheduler_output = _hybrid_prefix_hit(monkeypatch)
    runtime = _FakeKVRuntime(KV_NAMES)
    store = _snapshot_qwen3_5(runtime)
    if held is not None:
        store.capture_linear_state_prefix(held, block=1)
    runner.model.restore_linear_state_prefix = store.restore_linear_state_prefix
    runner.model.capture_linear_state_prefix = store.capture_linear_state_prefix
    prefix = {name: runtime.device[name][3, :, :4].clone() for name in KV_NAMES}
    runtime.calls.clear()

    with patch.object(optimum_model_runner, "logger") as logger:
        runner.execute_model(scheduler_output)
    runner.sample_tokens(None)

    # The whole prompt was prefilled into outer block 3, and slot 0 now holds
    # RESTORE under the scheduler's generation.
    logger.warning.assert_called_once()
    logger.error.assert_not_called()
    assert runtime.calls == [
        *(("get", name, 3, 0, 4) for name in KV_NAMES),
        *(("get", name, 3, 0, 8) for name in KV_NAMES),
    ]
    assert store._snapshot_kv[0][0] == RESTORE

    # The scheduler sends the next request with the prefix to the same slot.
    next_hit = _schedule_new_request_from_request(
        make_request("req_1", prompt_token_ids=list(range(1, 11))),
        block_ids=([4, 5, 6],),
        outer_block_ids=[2],
        finished_req_ids=["req_0"],
    )
    next_hit.cached_length = [RESTORE.boundary]
    next_hit.linear_state_restore = RESTORE
    forwarded = []
    forward = runner.model.forward

    def recording_forward(model_input, **kwargs):
        forwarded.append(model_input.input_tokens.tolist())
        return forward(model_input, **kwargs)

    runner.model.forward = recording_forward
    runtime.calls.clear()

    runner.execute_model(next_hit)

    assert runtime.calls == [("set", name, 2, 0, 4) for name in KV_NAMES]
    for name in KV_NAMES:
        assert torch.equal(runtime.device[name][2, :, :4], prefix[name])
    assert forwarded == [[list(range(5, 11))]]
