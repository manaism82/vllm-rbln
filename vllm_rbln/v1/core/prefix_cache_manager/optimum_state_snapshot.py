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
"""Prefix caching for hybrid models with linear-state snapshot rows.

A Qwen3.5 hybrid prefill cannot resume from a cached outer block alone: the
GatedDeltaNet conv/recurrent state after the prefix is not in the KV cache, and
rebel's ``copy_kv_cache`` (the outer-block hit path) also rewrites those state
tensors, indexed by block id, so it corrupts live requests. An artifact compiled
with ``linear_state_snapshot_slots`` K > 0 instead keeps K spare state rows
(row ``batch_size + s`` is slot ``s``). A slot holds the state after exactly
``boundary`` prompt tokens; the worker keeps a host copy of the full-attention
KV of ``[0, boundary)`` beside it. A hit writes that KV into the request's own
outer block and resumes the prefill from the slot's state row; finished
requests' outer blocks are never read.

Every boundary is a multiple of the hash block size, at or before the first
multimodal placeholder (block hashes of blocks that overlap an item include its
identifier, and the uncached tail must keep every item whole), inside the first
outer block, and before the last prompt token (which produces the logits). The
slot key is the request's chained block hash at the boundary, so a key match
means the whole prefix matches.

Admission: on its first sighting a prefix cannot tell which part is shared with
later requests, so it only records the keys of all its boundaries. A later
request captures the largest boundary whose key was already seen, which is the
longest shared prefix rounded down. A stream of requests that share no full
block therefore never captures and never evicts a reused prefix. The victim is
a free slot, else the least recently used slot (capture or hit), never the slot
the same request restores from.

The pool lives in the scheduler and decides a capture before the worker runs
it. Each capture gets a fresh generation, which the worker records with the
host copy and checks on restore, so a capture that the worker never completed
cannot be restored silently.
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import NamedTuple

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.request import Request

from vllm_rbln.logger import init_logger

logger = init_logger(__name__)

# Bound on the remembered boundary keys. A request adds at most one key per
# hash block before its first image, so this covers thousands of requests
# between two sightings of the same prefix.
GHOST_KEY_CAPACITY = 65536


class LinearStateSnapshot(NamedTuple):
    """One capture into snapshot slot ``slot``: the state and full-attention KV
    of prompt tokens ``[0, boundary)``. ``generation`` is unique per capture."""

    slot: int
    boundary: int
    generation: int


@dataclass
class _Slot:
    key: BlockHash
    snapshot: LinearStateSnapshot
    last_used: int


class LinearStateSnapshotPool:
    def __init__(self, num_slots: int, block_size: int, max_boundary: int) -> None:
        """
        Args:
            num_slots: K, the artifact's ``linear_state_snapshot_slots``.
            block_size: granularity of ``Request.block_hashes``; every boundary
                is a multiple of it.
            max_boundary: the outer block size; the restored prefix must lie in
                the request's first outer block.
        """
        assert num_slots > 0
        self.num_slots = num_slots
        self.block_size = block_size
        self.max_boundary = max_boundary
        self._slots: dict[int, _Slot] = {}
        self._seen: OrderedDict[BlockHash, None] = OrderedDict()
        self._clock = 0
        self._generation = 0

    def plan(
        self, request: Request
    ) -> tuple[LinearStateSnapshot | None, LinearStateSnapshot | None]:
        """Pick the snapshot this prefill restores from and the one it captures.

        Returns ``(restore, capture)``. ``restore.boundary`` is the number of
        prompt tokens the prefill skips; ``capture.boundary`` is larger.
        """
        self._clock += 1
        limit = min(request.num_prompt_tokens - 1, self.max_boundary)
        if request.mm_features:
            limit = min(limit, min(f.mm_position.offset for f in request.mm_features))
        # keys[i] is the key of boundary (i + 1) * block_size.
        keys = request.block_hashes[: limit // self.block_size]

        restore = None
        if not request.skip_reading_prefix_cache:
            hits = [
                slot
                for slot in self._slots.values()
                if slot.snapshot.boundary <= len(keys) * self.block_size
                and keys[slot.snapshot.boundary // self.block_size - 1] == slot.key
            ]
            if hits:
                hit = max(hits, key=lambda slot: slot.snapshot.boundary)
                hit.last_used = self._clock
                restore = hit.snapshot
                logger.debug(
                    "[PFX] [SNAPSHOT-HIT] REQUEST=%s | %s", request.request_id, restore
                )

        capture = None
        first = restore.boundary // self.block_size if restore is not None else 0
        shared = next(
            (i for i in reversed(range(first, len(keys))) if keys[i] in self._seen),
            None,
        )
        if shared is not None and all(
            slot.key != keys[shared] for slot in self._slots.values()
        ):
            boundary = (shared + 1) * self.block_size
            victims = [
                s for s in range(self.num_slots) if restore is None or s != restore.slot
            ]
            free = [s for s in victims if s not in self._slots]
            if free:
                victim = free[0]
            elif victims:
                victim = min(victims, key=lambda s: self._slots[s].last_used)
            else:
                victim = None
                logger.debug(
                    "[PFX] [SNAPSHOT-DENY] REQUEST=%s | BOUNDARY=%d | "
                    "REASON=only_slot_is_restored",
                    request.request_id,
                    boundary,
                )
            if victim is not None:
                evicted = self._slots.get(victim)
                self._generation += 1
                capture = LinearStateSnapshot(victim, boundary, self._generation)
                self._slots[victim] = _Slot(keys[shared], capture, self._clock)
                logger.debug(
                    "[PFX] [SNAPSHOT-CAPTURE] REQUEST=%s | %s | EVICTED=%s",
                    request.request_id,
                    capture,
                    evicted.snapshot if evicted is not None else None,
                )

        for key in keys:
            self._seen[key] = None
            self._seen.move_to_end(key)
        while len(self._seen) > GHOST_KEY_CAPACITY:
            self._seen.popitem(last=False)
        return restore, capture

    def reset(self) -> None:
        """Forget every slot and sighting. Generations keep increasing, so the
        worker's host copies from before the reset can never match again."""
        self._slots.clear()
        self._seen.clear()
