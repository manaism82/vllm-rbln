# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from unittest.mock import Mock

import pytest
import torch
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata

import vllm_rbln.compilation.compiler as compiler
from vllm_rbln.v1.sample import rbln_sampler as module
from vllm_rbln.v1.sample.rbln_sampler import RBLNSampler

VOCAB_SIZE = 8


def make_sampling_metadata(
    temperature: torch.Tensor | None,
    all_greedy: bool,
    all_random: bool,
) -> SamplingMetadata:
    return SamplingMetadata(
        temperature=temperature,
        all_greedy=all_greedy,
        all_random=all_random,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.tensor([]),
        presence_penalties=torch.tensor([]),
        repetition_penalties=torch.tensor([]),
        output_token_ids=[],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


@pytest.fixture
def sampler(monkeypatch) -> RBLNSampler:
    monkeypatch.setattr(module, "compile", lambda target, **kwargs: target)
    return RBLNSampler(compile_context=Mock())


@pytest.fixture
def op_args(sampler) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Record the (top_k, top_p) tensors the sampling op receives."""
    recorded: list[tuple[torch.Tensor, torch.Tensor]] = []
    original = sampler.topk_topp_sampler._compiled_rbln_topk_topp_sampler

    def spy(logits, temperature, k, p):
        recorded.append((k, p))
        return original(logits, temperature, k, p)

    sampler.topk_topp_sampler._compiled_rbln_topk_topp_sampler = spy
    return recorded


def near_tied_logits() -> torch.Tensor:
    """Two rows whose top two logits are almost equal.

    Under the previous encoding -- dividing a greedy row by 1e-3 to collapse its
    softmax toward a one-hot -- a gap this small survives the scaling (0.0001 *
    1000 = 0.1), leaving the argmax only ~52% likely. Narrowing the candidate
    set instead makes the outcome exact.
    """
    logits = torch.full((2, VOCAB_SIZE), -100.0)
    logits[0, 4] = 10.0
    logits[0, 6] = 10.0001  # row 0 argmax
    logits[1, 1] = 10.0001  # row 1 argmax
    logits[1, 5] = 10.0
    return logits


def test_greedy_row_of_a_mixed_batch_samples_its_argmax(sampler):
    metadata = make_sampling_metadata(
        temperature=torch.tensor([0.0, 1.0]),
        all_greedy=False,
        all_random=False,
    )

    for _ in range(10):
        sampled, _ = sampler.sample(near_tied_logits(), metadata)
        assert sampled[0].item() == 6


def test_all_greedy_batch_takes_the_argmax_op(sampler, op_args):
    """The reference for the test above: `rbln::argmax` must answer the same."""
    metadata = make_sampling_metadata(
        temperature=None,
        all_greedy=True,
        all_random=False,
    )

    sampled, _ = sampler.sample(near_tied_logits(), metadata)

    # An all-greedy batch skips the top-k/top-p op entirely.
    assert op_args == []
    assert sampled.tolist() == [6, 1]


@pytest.fixture
def compile_options(monkeypatch) -> list[dict]:
    """Record the options each sampler op reaches torch.compile with."""
    monkeypatch.setattr(module, "HAS_TORCH_RBLN", True)
    monkeypatch.setattr(module, "USE_DEVICE_TENSOR", False)
    # Keep compile() from rewriting the process-wide dynamo config.
    monkeypatch.setattr(compiler, "_DYNAMO_CONFIGURED", True)
    recorded: list[dict] = []

    def fake_torch_compile(target, *, backend, dynamic, fullgraph, options):
        recorded.append(options)
        return target

    monkeypatch.setattr(torch, "compile", fake_torch_compile)
    return recorded


def test_sampler_ops_compile_on_the_given_device(monkeypatch, compile_options):
    context = Mock()
    context_kwargs: list[dict] = []

    def fake_compile_context(**kwargs):
        context_kwargs.append(kwargs)
        return context

    monkeypatch.setattr(module.rebel, "CompileContext", fake_compile_context)

    RBLNSampler(device_id=8)

    assert context_kwargs == [{"use_global_ctx": True, "device_id": 8}]
    # Both the top-k/top-p op and the greedy op.
    assert len(compile_options) == 2
    for options in compile_options:
        assert options["device"] == 8
        assert options["compile_context"] is context
        assert options["global_device_id"] == 8


def test_sampler_ops_keep_rebels_default_device_without_one(compile_options):
    # How the native worker builds it: its RBLN_DEVICES is already narrowed to
    # its own NPUs, and the context it passes lives on device 0.
    context = Mock()

    RBLNSampler(compile_context=context)

    assert len(compile_options) == 2
    for options in compile_options:
        assert "device" not in options
        assert options["compile_context"] is context
        assert options["global_device_id"] == 0
