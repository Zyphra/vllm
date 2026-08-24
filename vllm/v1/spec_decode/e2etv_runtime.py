# SPDX-License-Identifier: Apache-2.0
"""Default-off runtime carrier for exact TiDAR verifier events.

The hot-path recorder is deliberately small: a request carries only a bounded
group context in its opaque vLLM request id, and each worker keeps a local
deterministic bottom-k window. Tensor payloads are copied to CPU only when an
event wins that local window. Control-plane code seals descriptor-only
manifests and requests selected payloads in a second RPC before destroying the
window.

This module owns no replay buffer, file writer, optimizer, trainer hook, or
policy objective. When it is not explicitly configured, it allocates no
reservoir and performs no event work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vllm.v1.spec_decode.e2etv_event_inputs import TiDARE2ETVEventBatch
from vllm.v1.spec_decode.e2etv_event_reservoir import (
    TiDARE2ETVGroupWindowManager,
    TiDARE2ETVPartitionManifest,
    TiDARE2ETVSelectedPartition,
    TiDARE2ETVSelectionPlan,
)

REQUEST_CONTEXT_PREFIX = "e2tv1."


@dataclass(frozen=True)
class TiDARE2ETVRequestContext:
    """Prompt-group context encoded into one otherwise opaque request id."""

    group_id: str
    selection_epoch: int
    opaque_request_id: str

    def validate(self) -> None:
        if len(self.group_id) != 64 or any(
            character not in "0123456789abcdef" for character in self.group_id
        ):
            raise ValueError("E2E-TV group_id must be a lowercase SHA-256 digest")
        if not 0 <= self.selection_epoch < 2**64:
            raise ValueError("E2E-TV selection_epoch must fit uint64")
        encoded = self.opaque_request_id.encode("utf-8")
        if not encoded or len(encoded) > 256:
            raise ValueError("E2E-TV opaque request id must be nonempty and bounded")
        if "." in self.opaque_request_id:
            raise ValueError("E2E-TV opaque request id cannot contain '.'")

    def encode(self) -> str:
        self.validate()
        return (
            f"{REQUEST_CONTEXT_PREFIX}{self.group_id}."
            f"{self.selection_epoch:016x}.{self.opaque_request_id}"
        )

    @classmethod
    def decode(cls, request_id: str) -> TiDARE2ETVRequestContext | None:
        if not request_id.startswith(REQUEST_CONTEXT_PREFIX):
            return None
        parts = request_id[len(REQUEST_CONTEXT_PREFIX) :].split(".", 2)
        if len(parts) != 3 or len(parts[1]) != 16:
            raise ValueError("malformed E2E-TV request context")
        try:
            selection_epoch = int(parts[1], 16)
        except ValueError as error:
            raise ValueError("malformed E2E-TV selection epoch") from error
        context = cls(
            group_id=parts[0],
            selection_epoch=selection_epoch,
            opaque_request_id=parts[2],
        )
        context.validate()
        if context.encode() != request_id:
            raise ValueError("noncanonical E2E-TV request context")
        return context


class TiDARE2ETVRuntimeRecorder:
    """Worker-local same-group event sampler with explicit lifecycle RPCs."""

    def __init__(
        self,
        *,
        partition_id: str,
        max_events_per_group: int,
        seed: int,
        max_open_groups: int,
        installed_policy_version: int,
    ) -> None:
        if not 0 <= installed_policy_version < 2**63:
            raise ValueError("installed_policy_version must fit nonnegative int64")
        self.window_manager = TiDARE2ETVGroupWindowManager(
            partition_id=partition_id,
            max_events_per_group=max_events_per_group,
            seed=seed,
            max_open_groups=max_open_groups,
        )
        self.installed_policy_version = installed_policy_version
        self.observed_events = 0
        self.selected_events = 0

    @property
    def partition_id(self) -> str:
        return self.window_manager.partition_id

    def set_installed_policy_version(self, version: int) -> None:
        if not 0 <= version < 2**63:
            raise ValueError("installed_policy_version must fit nonnegative int64")
        self.installed_policy_version = version

    def record(
        self,
        event_batch: TiDARE2ETVEventBatch,
        *,
        target_logits: torch.Tensor,
        draft_temperature: float,
    ) -> None:
        """Observe exact verifier events, skipping requests without a context."""

        expected = sum(event_batch.num_draft_tokens)
        if event_batch.target_hidden is None:
            raise RuntimeError(
                "E2E-TV event reached rejection sampling without exact target "
                "hidden states"
            )
        tensors = {
            "draft_hidden": event_batch.draft_hidden,
            "target_hidden": event_batch.target_hidden,
            "previous_token_ids": event_batch.previous_token_ids,
            "global_positions": event_batch.global_positions,
            "target_logits": target_logits,
        }
        bad = {
            name: tuple(tensor.shape)
            for name, tensor in tensors.items()
            if tensor.shape[0] != expected
        }
        if bad:
            raise RuntimeError(
                f"E2E-TV runtime tensors do not align to {expected} draft tokens: {bad}"
            )

        start = 0
        for request_id, count in zip(
            event_batch.req_ids, event_batch.num_draft_tokens, strict=True
        ):
            stop = start + count
            context = TiDARE2ETVRequestContext.decode(request_id)
            if context is not None:
                selected = self.window_manager.offer(
                    group_id=context.group_id,
                    selection_epoch=context.selection_epoch,
                    request_id=request_id,
                    installed_policy_version=self.installed_policy_version,
                    draft_temperature=draft_temperature,
                    draft_hidden=event_batch.draft_hidden[start:stop],
                    target_hidden=event_batch.target_hidden[start:stop],
                    previous_token_ids=event_batch.previous_token_ids[start:stop],
                    global_positions=event_batch.global_positions[start:stop],
                    target_logits=target_logits[start:stop],
                )
                self.observed_events += 1
                self.selected_events += int(selected)
            start = stop

    def seal_group(
        self, *, group_id: str, selection_epoch: int
    ) -> TiDARE2ETVPartitionManifest:
        return self.window_manager.seal(
            group_id=group_id,
            selection_epoch=selection_epoch,
        )

    def take_group(
        self, *, group_id: str, plan: TiDARE2ETVSelectionPlan
    ) -> TiDARE2ETVSelectedPartition:
        return self.window_manager.take_selected(group_id=group_id, plan=plan)

    def discard_group(self, group_id: str) -> bool:
        return self.window_manager.discard(group_id)

    def state_dict(self) -> dict[str, Any]:
        return {
            "window_manager": self.window_manager.state_dict(),
            "installed_policy_version": self.installed_policy_version,
            "observed_events": self.observed_events,
            "selected_events": self.selected_events,
        }


class TiDARE2ETVWorkerExtension:
    """Opt-in worker RPC surface for descriptor/payload carrier lifecycle."""

    def _e2etv_rejection_sampler(self):
        model_runner = getattr(self, "model_runner", None)
        rejection_sampler = getattr(model_runner, "rejection_sampler", None)
        if rejection_sampler is None:
            raise RuntimeError("E2E-TV runtime requires the TiDAR rejection sampler")
        return rejection_sampler

    def e2etv_configure_runtime(
        self,
        *,
        partition_prefix: str,
        max_events_per_group: int,
        seed: int,
        max_open_groups: int,
        installed_policy_version: int,
    ) -> dict[str, object]:
        rank = int(getattr(self, "rank", 0))
        partition_id = f"{partition_prefix}.rank-{rank:05d}"
        model_runner = getattr(self, "model_runner", None)
        if model_runner is None:
            raise RuntimeError("E2E-TV runtime requires a GPU model runner")
        sampler = self._e2etv_rejection_sampler()
        sampler.configure_e2etv_runtime(
            partition_id=partition_id,
            max_events_per_group=max_events_per_group,
            seed=seed,
            max_open_groups=max_open_groups,
            installed_policy_version=installed_policy_version,
        )
        model_runner.enable_e2etv_event_inputs()
        return {
            "enabled": True,
            "partition_id": partition_id,
            "max_events_per_group": max_events_per_group,
            "seed": seed,
            "max_open_groups": max_open_groups,
            "installed_policy_version": installed_policy_version,
        }

    def e2etv_set_installed_policy_version(self, version: int) -> int:
        sampler = self._e2etv_rejection_sampler()
        recorder = sampler.e2etv_runtime_recorder
        if recorder is None:
            raise RuntimeError("E2E-TV runtime is not configured")
        recorder.set_installed_policy_version(version)
        return recorder.installed_policy_version

    def e2etv_seal_group(
        self, *, group_id: str, selection_epoch: int
    ) -> TiDARE2ETVPartitionManifest:
        recorder = self._e2etv_rejection_sampler().e2etv_runtime_recorder
        if recorder is None:
            raise RuntimeError("E2E-TV runtime is not configured")
        return recorder.seal_group(
            group_id=group_id,
            selection_epoch=selection_epoch,
        )

    def e2etv_take_group(
        self, *, group_id: str, plan: TiDARE2ETVSelectionPlan
    ) -> TiDARE2ETVSelectedPartition:
        recorder = self._e2etv_rejection_sampler().e2etv_runtime_recorder
        if recorder is None:
            raise RuntimeError("E2E-TV runtime is not configured")
        return recorder.take_group(group_id=group_id, plan=plan)

    def e2etv_discard_group(self, group_id: str) -> bool:
        recorder = self._e2etv_rejection_sampler().e2etv_runtime_recorder
        if recorder is None:
            raise RuntimeError("E2E-TV runtime is not configured")
        return recorder.discard_group(group_id)

    def e2etv_runtime_state_dict(self) -> dict[str, Any]:
        recorder = self._e2etv_rejection_sampler().e2etv_runtime_recorder
        if recorder is None:
            raise RuntimeError("E2E-TV runtime is not configured")
        return recorder.state_dict()
