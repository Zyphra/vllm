# SPDX-License-Identifier: Apache-2.0
"""Deterministic, in-memory sampling of TiDAR verifier events.

This module is the platform-neutral sampling core for DSpARK head training.
It deliberately contains no file writer, RPC, trainer hook, optimizer, or
configuration parser.  When disabled, :func:`build_event_reservoir` returns
``None`` before allocating any state.

The reservoir keeps the lowest seeded content hashes.  That is equivalent to
uniform sampling without replacement under the hash-as-random-oracle model,
is independent of arrival order, and needs no mutable PRNG state.  Tensor
payloads are detached and copied to CPU only after an event wins admission.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

SCHEMA_VERSION = 1


def _sha256(*parts: bytes) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.hexdigest()


def event_identity(request_id: str, global_positions: torch.Tensor) -> str:
    """Return the stable identity of one request-scoped verifier event."""

    request_id = request_id.strip()
    if not request_id:
        raise ValueError("E2E-TV event request_id must be nonempty")
    positions = global_positions.detach().to("cpu", torch.int64).contiguous()
    if positions.ndim != 1 or positions.numel() <= 0:
        raise ValueError("E2E-TV event global_positions must be a nonempty vector")
    return _sha256(
        request_id.encode("utf-8"),
        b"\0",
        positions.numpy().tobytes(),
    )


def event_priority(*, event_id: str, seed: int, selection_epoch: int) -> str:
    """Return a seeded, epoch-scoped uniform priority for ``event_id``."""

    if len(event_id) != 64 or any(c not in "0123456789abcdef" for c in event_id):
        raise ValueError("E2E-TV event_id must be a lowercase SHA-256 digest")
    if not 0 <= seed < 2**64:
        raise ValueError("E2E-TV reservoir seed must fit uint64")
    if not 0 <= selection_epoch < 2**64:
        raise ValueError("E2E-TV selection_epoch must fit uint64")
    return _sha256(
        b"tidar-e2etv-reservoir-v1\0",
        seed.to_bytes(8, "big"),
        selection_epoch.to_bytes(8, "big"),
        bytes.fromhex(event_id),
    )


def _cpu_copy(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", copy=True).contiguous()


def _update_tensor_receipt(digest, tensor: torch.Tensor) -> None:
    """Bind a tensor's dtype, shape, and exact storage bytes to ``digest``."""

    tensor = tensor.detach().to(device="cpu").contiguous()
    dtype = str(tensor.dtype).encode("ascii")
    digest.update(len(dtype).to_bytes(2, "big"))
    digest.update(dtype)
    digest.update(tensor.ndim.to_bytes(2, "big"))
    for dimension in tensor.shape:
        digest.update(int(dimension).to_bytes(8, "big"))
    # NumPy cannot represent every Torch dtype (notably BF16). Viewing the
    # contiguous storage as bytes makes the receipt exact for every dtype.
    digest.update(tensor.view(torch.uint8).numpy().tobytes())


def _event_payload_receipt(
    *,
    request_id: str,
    event_id: str,
    priority_sha256: str,
    installed_policy_version: int,
    draft_temperature: float,
    draft_hidden: torch.Tensor,
    previous_token_ids: torch.Tensor,
    global_positions: torch.Tensor,
    target_logits: torch.Tensor,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"tidar-e2etv-event-payload-v1\0")
    encoded_request_id = request_id.encode("utf-8")
    digest.update(len(encoded_request_id).to_bytes(8, "big"))
    digest.update(encoded_request_id)
    digest.update(bytes.fromhex(event_id))
    digest.update(bytes.fromhex(priority_sha256))
    digest.update(installed_policy_version.to_bytes(8, "big", signed=True))
    digest.update(struct.pack("!d", float(draft_temperature)))
    for tensor in (
        draft_hidden,
        previous_token_ids,
        global_positions,
        target_logits,
    ):
        _update_tensor_receipt(digest, tensor)
    return digest.hexdigest()


def _validate_event_tensors(
    *,
    installed_policy_version: int,
    draft_temperature: float,
    draft_hidden: torch.Tensor,
    previous_token_ids: torch.Tensor,
    global_positions: torch.Tensor,
    target_logits: torch.Tensor,
    require_detached_cpu: bool,
    validate_values: bool = True,
) -> None:
    if not 0 <= installed_policy_version < 2**63:
        raise ValueError("installed_policy_version must fit nonnegative int64")
    if not math.isfinite(draft_temperature) or draft_temperature <= 0.0:
        raise ValueError("draft_temperature must be finite and positive")
    if global_positions.dtype != torch.int64:
        raise ValueError("global_positions must use int64")
    if previous_token_ids.dtype != torch.int64:
        raise ValueError("previous_token_ids must use int64")
    if not draft_hidden.is_floating_point():
        raise ValueError("draft_hidden must use a floating dtype")
    if not target_logits.is_floating_point():
        raise ValueError("target_logits must use a floating dtype")
    token_count = int(global_positions.numel())
    if global_positions.ndim != 1 or token_count <= 0:
        raise ValueError("global_positions must be a nonempty vector")
    if tuple(previous_token_ids.shape) != (token_count,):
        raise ValueError("previous_token_ids do not align with global_positions")
    if draft_hidden.ndim != 2 or draft_hidden.shape[0] != token_count:
        raise ValueError("draft_hidden does not align with global_positions")
    if target_logits.ndim != 2 or target_logits.shape[0] != token_count:
        raise ValueError("target_logits do not align with global_positions")
    if target_logits.shape[1] <= 0:
        raise ValueError("target_logits has an empty vocabulary axis")
    if token_count > 1 and not torch.equal(
        global_positions[1:] - global_positions[:-1],
        torch.ones(
            token_count - 1,
            dtype=torch.int64,
            device=global_positions.device,
        ),
    ):
        raise ValueError("global_positions must be consecutive")
    if validate_values:
        if not torch.isfinite(draft_hidden).all():
            raise ValueError("draft_hidden contains nonfinite values")
        # Negative infinity is a valid sampling mask; NaN and +inf are not.
        if torch.isnan(target_logits).any() or torch.isposinf(target_logits).any():
            raise ValueError("target_logits contains invalid values")
        if not torch.isfinite(target_logits).any(dim=-1).all():
            raise ValueError("target_logits contains an all-masked row")
    if require_detached_cpu:
        for tensor in (
            draft_hidden,
            previous_token_ids,
            global_positions,
            target_logits,
        ):
            if tensor.device.type != "cpu" or tensor.requires_grad:
                raise ValueError(
                    "transported E2E-TV tensors must be detached CPU tensors"
                )


@dataclass(frozen=True)
class TiDARE2ETVEventPayload:
    """Detached tensors required by the exact TiDAR acceptance objective."""

    request_id: str
    event_id: str
    priority_sha256: str
    installed_policy_version: int
    draft_temperature: float
    draft_hidden: torch.Tensor
    previous_token_ids: torch.Tensor
    global_positions: torch.Tensor
    target_logits: torch.Tensor
    payload_sha256: str

    @property
    def num_draft_tokens(self) -> int:
        return int(self.global_positions.numel())

    @property
    def tensor_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self.draft_hidden,
                self.previous_token_ids,
                self.global_positions,
                self.target_logits,
            )
        )

    def validate(self, *, seed: int, selection_epoch: int) -> None:
        _validate_event_tensors(
            installed_policy_version=self.installed_policy_version,
            draft_temperature=self.draft_temperature,
            draft_hidden=self.draft_hidden,
            previous_token_ids=self.previous_token_ids,
            global_positions=self.global_positions,
            target_logits=self.target_logits,
            require_detached_cpu=True,
        )
        if self.event_id != event_identity(self.request_id, self.global_positions):
            raise ValueError("E2E-TV event identity receipt mismatch")
        if self.priority_sha256 != event_priority(
            event_id=self.event_id,
            seed=seed,
            selection_epoch=selection_epoch,
        ):
            raise ValueError("E2E-TV event priority receipt mismatch")
        expected = _event_payload_receipt(
            request_id=self.request_id,
            event_id=self.event_id,
            priority_sha256=self.priority_sha256,
            installed_policy_version=self.installed_policy_version,
            draft_temperature=self.draft_temperature,
            draft_hidden=self.draft_hidden,
            previous_token_ids=self.previous_token_ids,
            global_positions=self.global_positions,
            target_logits=self.target_logits,
        )
        if self.payload_sha256 != expected:
            raise ValueError("E2E-TV event payload receipt mismatch")


@dataclass(frozen=True)
class TiDARE2ETVEventCarrier:
    """One rollout engine's bounded same-window event population sample."""

    schema_version: int
    seed: int
    selection_epoch: int
    reservoir_capacity: int
    observed_event_population: int
    events: tuple[TiDARE2ETVEventPayload, ...]
    lineage_sha256: str

    @property
    def tensor_bytes(self) -> int:
        return sum(event.tensor_bytes for event in self.events)

    def validate(self, *, max_events: int | None = None) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported E2E-TV carrier schema")
        event_priority(
            event_id="0" * 64,
            seed=self.seed,
            selection_epoch=self.selection_epoch,
        )
        if self.reservoir_capacity <= 0:
            raise ValueError("E2E-TV carrier reservoir capacity must be positive")
        if not 0 <= self.observed_event_population < 2**64:
            raise ValueError("E2E-TV carrier population must fit uint64")
        if self.observed_event_population < len(self.events):
            raise ValueError("E2E-TV carrier population is smaller than its sample")
        if len(self.events) > self.reservoir_capacity:
            raise ValueError("E2E-TV carrier exceeds its reservoir capacity")
        if max_events is not None and len(self.events) > max_events:
            raise ValueError("E2E-TV carrier exceeds the configured event budget")
        priorities = [event.priority_sha256 for event in self.events]
        event_ids = [event.event_id for event in self.events]
        if priorities != sorted(priorities):
            raise ValueError("E2E-TV carrier events are not in canonical order")
        if len(priorities) != len(set(priorities)):
            raise ValueError("E2E-TV carrier contains duplicate priorities")
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("E2E-TV carrier contains duplicate event identities")
        for event in self.events:
            event.validate(seed=self.seed, selection_epoch=self.selection_epoch)
        expected = _carrier_lineage(
            seed=self.seed,
            selection_epoch=self.selection_epoch,
            reservoir_capacity=self.reservoir_capacity,
            observed_event_population=self.observed_event_population,
            events=self.events,
        )
        if self.lineage_sha256 != expected:
            raise ValueError("E2E-TV carrier lineage receipt mismatch")


def _carrier_lineage(
    *,
    seed: int,
    selection_epoch: int,
    reservoir_capacity: int,
    observed_event_population: int,
    events: tuple[TiDARE2ETVEventPayload, ...],
) -> str:
    digest = hashlib.sha256()
    digest.update(b"tidar-e2etv-carrier-v1\0")
    digest.update(seed.to_bytes(8, "big"))
    digest.update(selection_epoch.to_bytes(8, "big"))
    digest.update(reservoir_capacity.to_bytes(8, "big"))
    digest.update(observed_event_population.to_bytes(8, "big"))
    for event in events:
        digest.update(bytes.fromhex(event.event_id))
        digest.update(bytes.fromhex(event.priority_sha256))
        digest.update(event.installed_policy_version.to_bytes(8, "big", signed=True))
        digest.update(bytes.fromhex(event.payload_sha256))
    return digest.hexdigest()


def merge_event_carriers(
    carriers: tuple[TiDARE2ETVEventCarrier, ...],
    *,
    max_events: int,
) -> TiDARE2ETVEventCarrier:
    """Exactly compose partition-local bottom-k reservoirs.

    Each input partition must have retained at least ``max_events`` entries.
    Under that condition, taking the global lowest priorities from their union
    is identical to sampling the combined event stream monolithically.
    """

    if not carriers:
        raise ValueError("at least one E2E-TV carrier is required")
    if max_events <= 0:
        raise ValueError("merged E2E-TV max_events must be positive")
    first = carriers[0]
    all_events: list[TiDARE2ETVEventPayload] = []
    observed_event_population = 0
    seen_event_ids: set[str] = set()
    for carrier in carriers:
        carrier.validate()
        if carrier.schema_version != first.schema_version:
            raise ValueError("E2E-TV carrier schema mismatch")
        if (
            carrier.seed != first.seed
            or carrier.selection_epoch != first.selection_epoch
        ):
            raise ValueError("E2E-TV carrier sampling configuration mismatch")
        if carrier.reservoir_capacity < max_events:
            raise ValueError(
                "partition reservoir capacity is too small for exact merge"
            )
        for event in carrier.events:
            if event.event_id in seen_event_ids:
                raise ValueError("E2E-TV carrier partitions overlap")
            seen_event_ids.add(event.event_id)
            all_events.append(event)
        observed_event_population += carrier.observed_event_population
    if observed_event_population >= 2**64:
        raise ValueError("merged E2E-TV carrier population overflows uint64")
    events = tuple(
        sorted(all_events, key=lambda event: event.priority_sha256)[:max_events]
    )
    merged = TiDARE2ETVEventCarrier(
        schema_version=SCHEMA_VERSION,
        seed=first.seed,
        selection_epoch=first.selection_epoch,
        reservoir_capacity=max_events,
        observed_event_population=observed_event_population,
        events=events,
        lineage_sha256=_carrier_lineage(
            seed=first.seed,
            selection_epoch=first.selection_epoch,
            reservoir_capacity=max_events,
            observed_event_population=observed_event_population,
            events=events,
        ),
    )
    merged.validate(max_events=max_events)
    return merged


class TiDARE2ETVEventReservoir:
    """Keep a deterministic uniform sample and drain it exactly once."""

    def __init__(
        self,
        *,
        max_events: int,
        seed: int,
        selection_epoch: int = 0,
    ) -> None:
        if max_events <= 0:
            raise ValueError("E2E-TV max_events must be positive")
        # Validate bounds through the common priority implementation.
        event_priority(event_id="0" * 64, seed=seed, selection_epoch=selection_epoch)
        self.max_events = max_events
        self.seed = seed
        self.selection_epoch = selection_epoch
        self._events: dict[str, TiDARE2ETVEventPayload] = {}
        self._seen_event_ids: set[str] = set()
        self._observed_event_population = 0

    @property
    def observed_event_population(self) -> int:
        return self._observed_event_population

    @property
    def selected_event_count(self) -> int:
        return len(self._events)

    def offer(
        self,
        *,
        request_id: str,
        installed_policy_version: int,
        draft_temperature: float,
        draft_hidden: torch.Tensor,
        previous_token_ids: torch.Tensor,
        global_positions: torch.Tensor,
        target_logits: torch.Tensor,
    ) -> bool:
        """Observe one event and copy its tensors only when it is selected."""

        _validate_event_tensors(
            installed_policy_version=installed_policy_version,
            draft_temperature=draft_temperature,
            draft_hidden=draft_hidden,
            previous_token_ids=previous_token_ids,
            global_positions=global_positions,
            target_logits=target_logits,
            require_detached_cpu=False,
            validate_values=False,
        )

        event_id = event_identity(request_id, global_positions)
        if event_id in self._seen_event_ids:
            raise RuntimeError(f"duplicate E2E-TV event identity {event_id}")
        self._seen_event_ids.add(event_id)
        self._observed_event_population += 1
        priority = event_priority(
            event_id=event_id,
            seed=self.seed,
            selection_epoch=self.selection_epoch,
        )
        if len(self._events) >= self.max_events and priority >= max(self._events):
            return False

        # Full value scans synchronize large logits tensors.  Perform them
        # only for actual local reservoir winners; rejected events still get
        # structural validation and exact population accounting above.
        _validate_event_tensors(
            installed_policy_version=installed_policy_version,
            draft_temperature=draft_temperature,
            draft_hidden=draft_hidden,
            previous_token_ids=previous_token_ids,
            global_positions=global_positions,
            target_logits=target_logits,
            require_detached_cpu=False,
            validate_values=True,
        )

        draft_hidden_cpu = _cpu_copy(draft_hidden)
        previous_token_ids_cpu = _cpu_copy(previous_token_ids)
        global_positions_cpu = _cpu_copy(global_positions)
        target_logits_cpu = _cpu_copy(target_logits)
        payload = TiDARE2ETVEventPayload(
            request_id=request_id,
            event_id=event_id,
            priority_sha256=priority,
            installed_policy_version=installed_policy_version,
            draft_temperature=float(draft_temperature),
            draft_hidden=draft_hidden_cpu,
            previous_token_ids=previous_token_ids_cpu,
            global_positions=global_positions_cpu,
            target_logits=target_logits_cpu,
            payload_sha256=_event_payload_receipt(
                request_id=request_id,
                event_id=event_id,
                priority_sha256=priority,
                installed_policy_version=installed_policy_version,
                draft_temperature=draft_temperature,
                draft_hidden=draft_hidden_cpu,
                previous_token_ids=previous_token_ids_cpu,
                global_positions=global_positions_cpu,
                target_logits=target_logits_cpu,
            ),
        )
        if len(self._events) >= self.max_events:
            del self._events[max(self._events)]
        self._events[priority] = payload
        return True

    def snapshot(self) -> TiDARE2ETVEventCarrier:
        """Return the canonical carrier without mutating the open window."""
        events = tuple(self._events[key] for key in sorted(self._events))
        carrier = TiDARE2ETVEventCarrier(
            schema_version=SCHEMA_VERSION,
            seed=self.seed,
            selection_epoch=self.selection_epoch,
            reservoir_capacity=self.max_events,
            observed_event_population=self._observed_event_population,
            events=events,
            lineage_sha256=_carrier_lineage(
                seed=self.seed,
                selection_epoch=self.selection_epoch,
                reservoir_capacity=self.max_events,
                observed_event_population=self._observed_event_population,
                events=events,
            ),
        )
        carrier.validate(max_events=self.max_events)
        return carrier

    def drain(self) -> TiDARE2ETVEventCarrier:
        """Return the canonical carrier and advance to the next selection epoch."""

        carrier = self.snapshot()
        self.selection_epoch += 1
        self._events.clear()
        self._seen_event_ids.clear()
        self._observed_event_population = 0
        return carrier

    def state_dict(self) -> dict[str, int]:
        """Checkpoint deterministic selection state between update windows."""

        if self._observed_event_population or self._events or self._seen_event_ids:
            raise RuntimeError("cannot checkpoint a nonempty E2E-TV reservoir")
        return {
            "schema_version": SCHEMA_VERSION,
            "max_events": self.max_events,
            "seed": self.seed,
            "selection_epoch": self.selection_epoch,
        }

    def load_state_dict(self, state: Mapping[str, int]) -> None:
        if self._observed_event_population or self._events or self._seen_event_ids:
            raise RuntimeError("cannot restore into a nonempty E2E-TV reservoir")
        expected = {
            "schema_version": SCHEMA_VERSION,
            "max_events": self.max_events,
            "seed": self.seed,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(f"E2E-TV reservoir state mismatch for {key}")
        selection_epoch = state.get("selection_epoch")
        if not isinstance(selection_epoch, int):
            raise ValueError("E2E-TV reservoir state has no integer selection_epoch")
        event_priority(
            event_id="0" * 64,
            seed=self.seed,
            selection_epoch=selection_epoch,
        )
        self.selection_epoch = selection_epoch


def build_event_reservoir(
    *,
    enabled: bool,
    max_events: int,
    seed: int,
    selection_epoch: int = 0,
) -> TiDARE2ETVEventReservoir | None:
    """Construct the sampler only when explicitly enabled."""

    if not enabled:
        return None
    return TiDARE2ETVEventReservoir(
        max_events=max_events,
        seed=seed,
        selection_epoch=selection_epoch,
    )


@dataclass(frozen=True)
class TiDARE2ETVEventDescriptor:
    """Tensor-free receipt for one partition-local reservoir winner."""

    event_id: str
    priority_sha256: str
    payload_sha256: str
    installed_policy_version: int
    tensor_bytes: int

    @classmethod
    def from_payload(cls, payload: TiDARE2ETVEventPayload) -> TiDARE2ETVEventDescriptor:
        return cls(
            event_id=payload.event_id,
            priority_sha256=payload.priority_sha256,
            payload_sha256=payload.payload_sha256,
            installed_policy_version=payload.installed_policy_version,
            tensor_bytes=payload.tensor_bytes,
        )

    def validate(self) -> None:
        for name, value in (
            ("event_id", self.event_id),
            ("priority_sha256", self.priority_sha256),
            ("payload_sha256", self.payload_sha256),
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"invalid E2E-TV descriptor {name}")
        if not 0 <= self.installed_policy_version < 2**63:
            raise ValueError("invalid E2E-TV descriptor policy version")
        if self.tensor_bytes <= 0:
            raise ValueError("invalid E2E-TV descriptor tensor size")


def _partition_manifest_sha256(
    *,
    group_id: str,
    partition_id: str,
    carrier: TiDARE2ETVEventCarrier,
    descriptors: tuple[TiDARE2ETVEventDescriptor, ...],
) -> str:
    digest = hashlib.sha256()
    digest.update(b"tidar-e2etv-partition-manifest-v1\0")
    digest.update(bytes.fromhex(group_id))
    encoded_partition = partition_id.encode("utf-8")
    digest.update(len(encoded_partition).to_bytes(8, "big"))
    digest.update(encoded_partition)
    digest.update(carrier.schema_version.to_bytes(8, "big"))
    digest.update(carrier.seed.to_bytes(8, "big"))
    digest.update(carrier.selection_epoch.to_bytes(8, "big"))
    digest.update(carrier.reservoir_capacity.to_bytes(8, "big"))
    digest.update(carrier.observed_event_population.to_bytes(8, "big"))
    digest.update(bytes.fromhex(carrier.lineage_sha256))
    for descriptor in descriptors:
        digest.update(bytes.fromhex(descriptor.event_id))
        digest.update(bytes.fromhex(descriptor.priority_sha256))
        digest.update(bytes.fromhex(descriptor.payload_sha256))
        digest.update(descriptor.installed_policy_version.to_bytes(8, "big"))
        digest.update(descriptor.tensor_bytes.to_bytes(8, "big"))
    return digest.hexdigest()


@dataclass(frozen=True)
class TiDARE2ETVPartitionManifest:
    """Small first-phase receipt; contains no model tensors."""

    group_id: str
    partition_id: str
    seed: int
    selection_epoch: int
    reservoir_capacity: int
    observed_event_population: int
    carrier_lineage_sha256: str
    descriptors: tuple[TiDARE2ETVEventDescriptor, ...]
    manifest_sha256: str

    @classmethod
    def from_carrier(
        cls,
        *,
        group_id: str,
        partition_id: str,
        carrier: TiDARE2ETVEventCarrier,
    ) -> TiDARE2ETVPartitionManifest:
        carrier.validate()
        _validate_group_and_partition(group_id, partition_id)
        descriptors = tuple(
            TiDARE2ETVEventDescriptor.from_payload(event) for event in carrier.events
        )
        return cls(
            group_id=group_id,
            partition_id=partition_id,
            seed=carrier.seed,
            selection_epoch=carrier.selection_epoch,
            reservoir_capacity=carrier.reservoir_capacity,
            observed_event_population=carrier.observed_event_population,
            carrier_lineage_sha256=carrier.lineage_sha256,
            descriptors=descriptors,
            manifest_sha256=_partition_manifest_sha256(
                group_id=group_id,
                partition_id=partition_id,
                carrier=carrier,
                descriptors=descriptors,
            ),
        )

    def validate(self) -> None:
        _validate_group_and_partition(self.group_id, self.partition_id)
        event_priority(
            event_id="0" * 64,
            seed=self.seed,
            selection_epoch=self.selection_epoch,
        )
        if self.reservoir_capacity <= 0:
            raise ValueError("invalid E2E-TV manifest reservoir capacity")
        if not 0 <= self.observed_event_population < 2**64:
            raise ValueError("invalid E2E-TV manifest event population")
        if len(self.descriptors) > self.reservoir_capacity:
            raise ValueError("E2E-TV manifest exceeds reservoir capacity")
        if len(self.carrier_lineage_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.carrier_lineage_sha256
        ):
            raise ValueError("invalid E2E-TV manifest carrier lineage")
        if len(self.manifest_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.manifest_sha256
        ):
            raise ValueError("invalid E2E-TV partition manifest receipt")
        for descriptor in self.descriptors:
            descriptor.validate()
        priorities = [descriptor.priority_sha256 for descriptor in self.descriptors]
        event_ids = [descriptor.event_id for descriptor in self.descriptors]
        if priorities != sorted(priorities):
            raise ValueError("E2E-TV manifest is not in canonical order")
        if len(priorities) != len(set(priorities)) or len(event_ids) != len(
            set(event_ids)
        ):
            raise ValueError("E2E-TV manifest contains duplicate events")
        carrier = TiDARE2ETVEventCarrier(
            schema_version=SCHEMA_VERSION,
            seed=self.seed,
            selection_epoch=self.selection_epoch,
            reservoir_capacity=self.reservoir_capacity,
            observed_event_population=self.observed_event_population,
            events=(),
            lineage_sha256=self.carrier_lineage_sha256,
        )
        expected = _partition_manifest_sha256(
            group_id=self.group_id,
            partition_id=self.partition_id,
            carrier=carrier,
            descriptors=self.descriptors,
        )
        if expected != self.manifest_sha256:
            raise ValueError("E2E-TV partition manifest receipt mismatch")


def _validate_group_and_partition(group_id: str, partition_id: str) -> None:
    if len(group_id) != 64 or any(
        character not in "0123456789abcdef" for character in group_id
    ):
        raise ValueError("E2E-TV group_id must be a lowercase SHA-256 digest")
    if not partition_id or len(partition_id.encode("utf-8")) > 256:
        raise ValueError("E2E-TV partition_id must be nonempty and bounded")


def _selection_plan_sha256(
    *,
    group_id: str,
    max_events: int,
    manifests: tuple[TiDARE2ETVPartitionManifest, ...],
    selected: tuple[tuple[str, str], ...],
) -> str:
    digest = hashlib.sha256()
    digest.update(b"tidar-e2etv-selection-plan-v1\0")
    digest.update(bytes.fromhex(group_id))
    digest.update(max_events.to_bytes(8, "big"))
    for manifest in manifests:
        digest.update(bytes.fromhex(manifest.manifest_sha256))
    for partition_id, event_id in selected:
        encoded_partition = partition_id.encode("utf-8")
        digest.update(len(encoded_partition).to_bytes(8, "big"))
        digest.update(encoded_partition)
        digest.update(bytes.fromhex(event_id))
    return digest.hexdigest()


@dataclass(frozen=True)
class TiDARE2ETVSelectionPlan:
    """Exact global bottom-k decision over tensor-free manifests."""

    group_id: str
    seed: int
    selection_epoch: int
    max_events: int
    observed_event_population: int
    manifests: tuple[TiDARE2ETVPartitionManifest, ...]
    selected: tuple[tuple[str, str], ...]
    plan_sha256: str

    def validate(self) -> None:
        expected = plan_event_selection(self.manifests, max_events=self.max_events)
        if self != expected:
            raise ValueError("E2E-TV selection plan receipt mismatch")


def plan_event_selection(
    manifests: Sequence[TiDARE2ETVPartitionManifest],
    *,
    max_events: int,
) -> TiDARE2ETVSelectionPlan:
    """Select the exact global bottom-k using descriptor-only manifests."""

    manifests = tuple(sorted(manifests, key=lambda item: item.partition_id))
    if not manifests:
        raise ValueError("at least one E2E-TV partition manifest is required")
    if max_events <= 0:
        raise ValueError("E2E-TV selection max_events must be positive")
    first = manifests[0]
    seen_partitions: set[str] = set()
    seen_events: set[str] = set()
    candidates: list[tuple[str, str, str]] = []
    population = 0
    for manifest in manifests:
        manifest.validate()
        if manifest.partition_id in seen_partitions:
            raise ValueError("duplicate E2E-TV partition manifest")
        seen_partitions.add(manifest.partition_id)
        if (
            manifest.group_id != first.group_id
            or manifest.seed != first.seed
            or manifest.selection_epoch != first.selection_epoch
        ):
            raise ValueError("E2E-TV manifest sampling configuration mismatch")
        if manifest.reservoir_capacity < max_events:
            raise ValueError("partition reservoir capacity is too small")
        population += manifest.observed_event_population
        if population >= 2**64:
            raise ValueError("E2E-TV group population overflows uint64")
        for descriptor in manifest.descriptors:
            if descriptor.event_id in seen_events:
                raise ValueError("E2E-TV partition event populations overlap")
            seen_events.add(descriptor.event_id)
            candidates.append(
                (
                    descriptor.priority_sha256,
                    manifest.partition_id,
                    descriptor.event_id,
                )
            )
    chosen = sorted(candidates)[:max_events]
    selected = tuple((partition_id, event_id) for _, partition_id, event_id in chosen)
    return TiDARE2ETVSelectionPlan(
        group_id=first.group_id,
        seed=first.seed,
        selection_epoch=first.selection_epoch,
        max_events=max_events,
        observed_event_population=population,
        manifests=manifests,
        selected=selected,
        plan_sha256=_selection_plan_sha256(
            group_id=first.group_id,
            max_events=max_events,
            manifests=manifests,
            selected=selected,
        ),
    )


@dataclass(frozen=True)
class TiDARE2ETVSelectedPartition:
    partition_id: str
    manifest_sha256: str
    observed_event_population: int
    events: tuple[TiDARE2ETVEventPayload, ...]


def select_partition_payloads(
    carrier: TiDARE2ETVEventCarrier,
    manifest: TiDARE2ETVPartitionManifest,
    plan: TiDARE2ETVSelectionPlan,
) -> TiDARE2ETVSelectedPartition:
    """Second phase: return only payloads selected for this partition."""

    plan.validate()
    manifest.validate()
    if manifest not in plan.manifests:
        raise ValueError("E2E-TV manifest is absent from the selection plan")
    carrier.validate()
    expected_manifest = TiDARE2ETVPartitionManifest.from_carrier(
        group_id=manifest.group_id,
        partition_id=manifest.partition_id,
        carrier=carrier,
    )
    if expected_manifest != manifest:
        raise ValueError("E2E-TV carrier changed after its manifest was sealed")
    selected_ids = {
        event_id
        for partition_id, event_id in plan.selected
        if partition_id == manifest.partition_id
    }
    events = tuple(event for event in carrier.events if event.event_id in selected_ids)
    if {event.event_id for event in events} != selected_ids:
        raise ValueError("E2E-TV selected partition is missing a payload")
    return TiDARE2ETVSelectedPartition(
        partition_id=manifest.partition_id,
        manifest_sha256=manifest.manifest_sha256,
        observed_event_population=manifest.observed_event_population,
        events=events,
    )


def finalize_event_selection(
    plan: TiDARE2ETVSelectionPlan,
    partitions: Sequence[TiDARE2ETVSelectedPartition],
) -> TiDARE2ETVEventCarrier:
    """Verify second-phase replies and build the globally sampled carrier."""

    plan.validate()
    partitions = tuple(sorted(partitions, key=lambda item: item.partition_id))
    if [item.partition_id for item in partitions] != [
        item.partition_id for item in plan.manifests
    ]:
        raise ValueError("E2E-TV selected partition set is incomplete")
    manifest_by_partition = {
        manifest.partition_id: manifest for manifest in plan.manifests
    }
    payload_by_id: dict[str, TiDARE2ETVEventPayload] = {}
    population = 0
    for partition in partitions:
        manifest = manifest_by_partition[partition.partition_id]
        if partition.manifest_sha256 != manifest.manifest_sha256:
            raise ValueError("E2E-TV selected partition manifest mismatch")
        if partition.observed_event_population != manifest.observed_event_population:
            raise ValueError("E2E-TV selected partition population mismatch")
        population += partition.observed_event_population
        descriptors = {item.event_id: item for item in manifest.descriptors}
        expected_ids = {
            event_id
            for partition_id, event_id in plan.selected
            if partition_id == partition.partition_id
        }
        if {event.event_id for event in partition.events} != expected_ids:
            raise ValueError("E2E-TV selected partition payload set mismatch")
        for event in partition.events:
            event.validate(seed=plan.seed, selection_epoch=plan.selection_epoch)
            descriptor = descriptors[event.event_id]
            if TiDARE2ETVEventDescriptor.from_payload(event) != descriptor:
                raise ValueError("E2E-TV selected payload receipt mismatch")
            payload_by_id[event.event_id] = event
    if population != plan.observed_event_population:
        raise ValueError("E2E-TV selected population does not match the plan")
    events = tuple(
        sorted(
            payload_by_id.values(),
            key=lambda event: event.priority_sha256,
        )
    )
    selected_ids = {event_id for _, event_id in plan.selected}
    if {event.event_id for event in events} != selected_ids:
        raise ValueError("E2E-TV final carrier is missing a selected event")
    carrier = TiDARE2ETVEventCarrier(
        schema_version=SCHEMA_VERSION,
        seed=plan.seed,
        selection_epoch=plan.selection_epoch,
        reservoir_capacity=plan.max_events,
        observed_event_population=plan.observed_event_population,
        events=events,
        lineage_sha256=_carrier_lineage(
            seed=plan.seed,
            selection_epoch=plan.selection_epoch,
            reservoir_capacity=plan.max_events,
            observed_event_population=plan.observed_event_population,
            events=events,
        ),
    )
    carrier.validate(max_events=plan.max_events)
    return carrier


class TiDARE2ETVGroupWindowManager:
    """Bounded partition-local reservoirs for in-flight rollout groups.

    The manager owns no thread, RPC, file, or persistent replay state.  A
    caller opens windows lazily by offering actual verifier events, seals a
    tensor-free manifest once the rollout group has finished, then either
    takes the plan-selected payloads exactly once or discards the window.
    """

    def __init__(
        self,
        *,
        partition_id: str,
        max_events_per_group: int,
        seed: int,
        max_open_groups: int,
    ) -> None:
        _validate_group_and_partition("0" * 64, partition_id)
        if max_events_per_group <= 0:
            raise ValueError("E2E-TV per-group event budget must be positive")
        if max_open_groups <= 0:
            raise ValueError("E2E-TV open-group bound must be positive")
        event_priority(event_id="0" * 64, seed=seed, selection_epoch=0)
        self.partition_id = partition_id
        self.max_events_per_group = max_events_per_group
        self.seed = seed
        self.max_open_groups = max_open_groups
        self._windows: dict[str, TiDARE2ETVEventReservoir] = {}
        self._sealed: dict[str, TiDARE2ETVPartitionManifest] = {}

    @property
    def open_group_count(self) -> int:
        return len(self._windows)

    def _window(self, group_id: str, selection_epoch: int) -> TiDARE2ETVEventReservoir:
        _validate_group_and_partition(group_id, self.partition_id)
        event_priority(
            event_id="0" * 64,
            seed=self.seed,
            selection_epoch=selection_epoch,
        )
        if group_id in self._sealed:
            raise RuntimeError("cannot add an event to a sealed E2E-TV group")
        window = self._windows.get(group_id)
        if window is None:
            if len(self._windows) >= self.max_open_groups:
                raise RuntimeError("E2E-TV open-group bound exceeded")
            window = TiDARE2ETVEventReservoir(
                max_events=self.max_events_per_group,
                seed=self.seed,
                selection_epoch=selection_epoch,
            )
            self._windows[group_id] = window
        elif window.selection_epoch != selection_epoch:
            raise ValueError("E2E-TV group selection epoch changed while open")
        return window

    def offer(
        self,
        *,
        group_id: str,
        selection_epoch: int,
        request_id: str,
        installed_policy_version: int,
        draft_temperature: float,
        draft_hidden: torch.Tensor,
        previous_token_ids: torch.Tensor,
        global_positions: torch.Tensor,
        target_logits: torch.Tensor,
    ) -> bool:
        return self._window(group_id, selection_epoch).offer(
            request_id=request_id,
            installed_policy_version=installed_policy_version,
            draft_temperature=draft_temperature,
            draft_hidden=draft_hidden,
            previous_token_ids=previous_token_ids,
            global_positions=global_positions,
            target_logits=target_logits,
        )

    def seal(
        self, *, group_id: str, selection_epoch: int
    ) -> TiDARE2ETVPartitionManifest:
        """Freeze a descriptor-only manifest for phase-one selection."""

        if group_id in self._sealed:
            manifest = self._sealed[group_id]
            if manifest.selection_epoch != selection_epoch:
                raise ValueError("sealed E2E-TV group selection epoch mismatch")
            return manifest
        window = self._windows.get(group_id)
        if window is None:
            # Every partition participates, including those that observed no
            # verifier events for this group.
            window = self._window(group_id, selection_epoch)
        manifest = TiDARE2ETVPartitionManifest.from_carrier(
            group_id=group_id,
            partition_id=self.partition_id,
            carrier=window.snapshot(),
        )
        self._sealed[group_id] = manifest
        return manifest

    def take_selected(
        self,
        *,
        group_id: str,
        plan: TiDARE2ETVSelectionPlan,
    ) -> TiDARE2ETVSelectedPartition:
        """Return plan-selected tensors and destroy the group window."""

        manifest = self._sealed.get(group_id)
        window = self._windows.get(group_id)
        if manifest is None or window is None:
            raise RuntimeError("E2E-TV group must be sealed before take")
        if plan.group_id != group_id:
            raise ValueError("E2E-TV selection plan targets another group")
        selected = select_partition_payloads(window.snapshot(), manifest, plan)
        del self._sealed[group_id]
        del self._windows[group_id]
        return selected

    def discard(self, group_id: str) -> bool:
        """Destroy one unfinished or unconsumed group without returning data."""

        removed = self._windows.pop(group_id, None) is not None
        self._sealed.pop(group_id, None)
        return removed

    def state_dict(self) -> dict[str, object]:
        if self._windows or self._sealed:
            raise RuntimeError("cannot checkpoint with open E2E-TV groups")
        return {
            "schema_version": SCHEMA_VERSION,
            "partition_id": self.partition_id,
            "max_events_per_group": self.max_events_per_group,
            "seed": self.seed,
            "max_open_groups": self.max_open_groups,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if self._windows or self._sealed:
            raise RuntimeError("cannot restore with open E2E-TV groups")
        if state != self.state_dict():
            raise ValueError("E2E-TV group-window configuration changed")


def build_group_window_manager(
    *,
    enabled: bool,
    partition_id: str,
    max_events_per_group: int,
    seed: int,
    max_open_groups: int,
) -> TiDARE2ETVGroupWindowManager | None:
    if not enabled:
        return None
    return TiDARE2ETVGroupWindowManager(
        partition_id=partition_id,
        max_events_per_group=max_events_per_group,
        seed=seed,
        max_open_groups=max_open_groups,
    )
