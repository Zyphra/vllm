import copy

import pytest
import torch

from vllm.v1.spec_decode.e2etv_event_reservoir import (
    TiDARE2ETVEventCarrier,
    TiDARE2ETVGroupWindowManager,
    TiDARE2ETVPartitionManifest,
    TiDARE2ETVEventReservoir,
    build_event_reservoir,
    build_group_window_manager,
    event_identity,
    finalize_event_selection,
    merge_event_carriers,
    plan_event_selection,
    select_partition_payloads,
)


def _event(index: int, *, version: int = 7) -> dict[str, object]:
    token_count = 3
    return {
        "request_id": f"request-{index}",
        "installed_policy_version": version,
        "draft_temperature": 0.8,
        "draft_hidden": torch.full((token_count, 4), float(index)),
        "target_hidden": torch.full((token_count, 4), float(index + 100)),
        "previous_token_ids": torch.arange(token_count, dtype=torch.int64) + index,
        "global_positions": torch.arange(token_count, dtype=torch.int64) + 10 * index,
        "target_logits": torch.arange(
            token_count * 5, dtype=torch.float32
        ).reshape(token_count, 5)
        + index,
    }


def _sample(order: list[int], *, seed: int = 19, epoch: int = 4):
    reservoir = TiDARE2ETVEventReservoir(
        max_events=3,
        seed=seed,
        selection_epoch=epoch,
    )
    for index in order:
        reservoir.offer(**_event(index))
    return reservoir.drain()


def test_reservoir_is_deterministic_and_arrival_order_independent() -> None:
    forward = _sample(list(range(12)))
    reverse = _sample(list(reversed(range(12))))
    assert [event.event_id for event in forward.events] == [
        event.event_id for event in reverse.events
    ]
    assert forward.lineage_sha256 == reverse.lineage_sha256
    assert forward.observed_event_population == 12
    assert len(forward.events) == 3


def test_seed_and_selection_epoch_change_the_uniform_sample() -> None:
    base = {event.event_id for event in _sample(list(range(32))).events}
    other_seed = {
        event.event_id for event in _sample(list(range(32)), seed=20).events
    }
    other_epoch = {
        event.event_id for event in _sample(list(range(32)), epoch=5).events
    }
    assert base != other_seed
    assert base != other_epoch


def test_selected_payloads_are_detached_cpu_copies() -> None:
    source = _event(1)
    hidden = source["draft_hidden"]
    assert isinstance(hidden, torch.Tensor)
    hidden.requires_grad_(True)
    reservoir = TiDARE2ETVEventReservoir(max_events=1, seed=1)
    assert reservoir.offer(**source)
    hidden.data.fill_(99)
    payload = reservoir.drain().events[0]
    assert payload.draft_hidden.device.type == "cpu"
    assert not payload.draft_hidden.requires_grad
    assert torch.equal(payload.draft_hidden, torch.ones_like(payload.draft_hidden))


def test_rejected_event_is_not_retained_and_population_is_exact() -> None:
    reservoir = TiDARE2ETVEventReservoir(max_events=1, seed=7)
    kept = 0
    for index in range(20):
        kept += int(reservoir.offer(**_event(index)))
    carrier = reservoir.drain()
    assert kept >= 1
    assert carrier.observed_event_population == 20
    assert len(carrier.events) == 1


def test_drain_resets_window_and_advances_checkpointed_epoch() -> None:
    reservoir = TiDARE2ETVEventReservoir(max_events=2, seed=8)
    reservoir.offer(**_event(1, version=3))
    first = reservoir.drain()
    assert first.selection_epoch == 0
    assert first.events[0].installed_policy_version == 3
    assert reservoir.observed_event_population == 0
    assert reservoir.selected_event_count == 0
    state = reservoir.state_dict()
    assert state["selection_epoch"] == 1

    restored = TiDARE2ETVEventReservoir(max_events=2, seed=8)
    restored.load_state_dict(state)
    restored.offer(**_event(2, version=4))
    second = restored.drain()
    assert second.selection_epoch == 1
    assert second.events[0].installed_policy_version == 4


def test_snapshot_is_nonmutating_and_matches_the_following_drain() -> None:
    reservoir = TiDARE2ETVEventReservoir(max_events=2, seed=8)
    reservoir.offer(**_event(1))
    snapshot = reservoir.snapshot()
    assert reservoir.observed_event_population == 1
    assert reservoir.selected_event_count == 1
    assert reservoir.drain() == snapshot


def test_checkpoint_fails_closed_with_pending_events() -> None:
    reservoir = TiDARE2ETVEventReservoir(max_events=1, seed=1)
    reservoir.offer(**_event(1))
    with pytest.raises(RuntimeError, match="nonempty"):
        reservoir.state_dict()
    with pytest.raises(RuntimeError, match="nonempty"):
        reservoir.load_state_dict(
            {
                "schema_version": 1,
                "max_events": 1,
                "seed": 1,
                "selection_epoch": 0,
            }
        )


def test_restore_rejects_config_or_schema_drift() -> None:
    reservoir = TiDARE2ETVEventReservoir(max_events=2, seed=1)
    state = reservoir.state_dict()
    for key, value in (
        ("schema_version", int(state["schema_version"]) + 1),
        ("max_events", 3),
        ("seed", 2),
    ):
        bad = dict(state)
        bad[key] = value
        with pytest.raises(ValueError, match=key):
            reservoir.load_state_dict(bad)


def test_duplicate_event_identity_fails_closed() -> None:
    reservoir = TiDARE2ETVEventReservoir(max_events=2, seed=1)
    event = _event(1)
    reservoir.offer(**event)
    with pytest.raises(RuntimeError, match="duplicate"):
        reservoir.offer(**event)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda event: event.update(request_id=""), "request_id"),
        (
            lambda event: event.update(installed_policy_version=-1),
            "installed_policy_version",
        ),
        (lambda event: event.update(draft_temperature=0.0), "draft_temperature"),
        (
            lambda event: event.update(
                global_positions=torch.tensor([1, 3, 4], dtype=torch.int64)
            ),
            "consecutive",
        ),
        (
            lambda event: event["target_logits"].fill_(float("nan")),
            "target_logits",
        ),
    ],
)
def test_malformed_events_fail_closed(mutation, message: str) -> None:
    event = _event(1)
    mutation(event)
    reservoir = TiDARE2ETVEventReservoir(max_events=1, seed=1)
    with pytest.raises(ValueError, match=message):
        reservoir.offer(**event)


def test_carrier_detects_lineage_or_order_tampering() -> None:
    carrier = _sample(list(range(8)))
    carrier.validate(max_events=3)
    with pytest.raises(ValueError, match="lineage"):
        TiDARE2ETVEventCarrier(
            schema_version=carrier.schema_version,
            seed=carrier.seed,
            selection_epoch=carrier.selection_epoch,
            reservoir_capacity=carrier.reservoir_capacity,
            observed_event_population=carrier.observed_event_population,
            events=carrier.events,
            lineage_sha256="0" * 64,
        ).validate(max_events=3)
    reversed_carrier = copy.copy(carrier)
    object.__setattr__(reversed_carrier, "events", tuple(reversed(carrier.events)))
    with pytest.raises(ValueError, match="canonical order"):
        reversed_carrier.validate(max_events=3)


def test_disabled_factory_allocates_no_reservoir() -> None:
    # Deliberately invalid values establish that disabled mode returns before
    # constructing or validating sampler state.
    assert build_event_reservoir(enabled=False, max_events=0, seed=-1) is None


def test_event_identity_is_stable_and_request_scoped() -> None:
    positions = torch.tensor([10, 11, 12], dtype=torch.int64)
    assert event_identity("a", positions) == event_identity("a", positions.clone())
    assert event_identity("a", positions) != event_identity("b", positions)


def test_tensor_byte_receipt_covers_all_transported_tensors() -> None:
    carrier = _sample([1])
    event = carrier.events[0]
    expected = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (
            event.draft_hidden,
            event.target_hidden,
            event.previous_token_ids,
            event.global_positions,
            event.target_logits,
        )
    )
    assert event.tensor_bytes == expected
    assert carrier.tensor_bytes == expected


def test_carrier_detects_tensor_payload_tampering() -> None:
    carrier = _sample([1])
    event = copy.copy(carrier.events[0])
    target_logits = event.target_logits.clone()
    target_logits[0, 0] += 1.0
    object.__setattr__(event, "target_logits", target_logits)
    tampered = copy.copy(carrier)
    object.__setattr__(tampered, "events", (event,))
    with pytest.raises(ValueError, match="payload receipt"):
        tampered.validate()


def test_partitioned_reservoirs_merge_exactly_like_monolithic_stream() -> None:
    monolithic = TiDARE2ETVEventReservoir(max_events=4, seed=31, selection_epoch=9)
    partitions = [
        TiDARE2ETVEventReservoir(max_events=4, seed=31, selection_epoch=9)
        for _ in range(3)
    ]
    for index in range(30):
        event = _event(index)
        monolithic.offer(**event)
        partitions[index % len(partitions)].offer(**event)
    expected = monolithic.drain()
    merged = merge_event_carriers(
        tuple(partition.drain() for partition in partitions),
        max_events=4,
    )
    assert merged.observed_event_population == 30
    assert [event.event_id for event in merged.events] == [
        event.event_id for event in expected.events
    ]
    assert merged.lineage_sha256 == expected.lineage_sha256


def test_exact_merge_rejects_undersized_partition_reservoirs() -> None:
    partition = TiDARE2ETVEventReservoir(max_events=1, seed=4)
    partition.offer(**_event(1))
    with pytest.raises(ValueError, match="capacity"):
        merge_event_carriers((partition.drain(),), max_events=2)


def _partitioned_selection_fixture():
    group_id = "a" * 64
    reservoirs = [
        TiDARE2ETVEventReservoir(max_events=3, seed=53, selection_epoch=11)
        for _ in range(3)
    ]
    monolithic = TiDARE2ETVEventReservoir(
        max_events=3, seed=53, selection_epoch=11
    )
    for index in range(24):
        event = _event(index)
        reservoirs[index % 3].offer(**event)
        monolithic.offer(**event)
    carriers = tuple(reservoir.snapshot() for reservoir in reservoirs)
    manifests = tuple(
        TiDARE2ETVPartitionManifest.from_carrier(
            group_id=group_id,
            partition_id=f"partition-{index}",
            carrier=carrier,
        )
        for index, carrier in enumerate(carriers)
    )
    return monolithic.snapshot(), carriers, manifests


def test_two_phase_selection_transports_only_exact_global_winners() -> None:
    expected, carriers, manifests = _partitioned_selection_fixture()
    plan = plan_event_selection(reversed(manifests), max_events=3)
    # The first phase is descriptor-only: no Tensor is reachable from a
    # manifest or selection plan.
    assert not any(
        isinstance(value, torch.Tensor)
        for manifest in plan.manifests
        for descriptor in manifest.descriptors
        for value in vars(descriptor).values()
    )
    selected = tuple(
        select_partition_payloads(carrier, manifest, plan)
        for carrier, manifest in zip(carriers, manifests, strict=True)
    )
    result = finalize_event_selection(plan, selected)
    assert result.observed_event_population == 24
    assert [event.event_id for event in result.events] == [
        event.event_id for event in expected.events
    ]
    assert sum(len(partition.events) for partition in selected) == 3


def test_two_phase_selection_supports_empty_partitions() -> None:
    expected, carriers, manifests = _partitioned_selection_fixture()
    # Build the canonical lineage through a real empty reservoir rather than
    # duplicating the receipt implementation in the test.
    empty_carrier = TiDARE2ETVEventReservoir(
        max_events=3, seed=53, selection_epoch=11
    ).snapshot()
    empty_manifest = TiDARE2ETVPartitionManifest.from_carrier(
        group_id="a" * 64,
        partition_id="partition-empty",
        carrier=empty_carrier,
    )
    plan = plan_event_selection((*manifests, empty_manifest), max_events=3)
    selected = tuple(
        select_partition_payloads(carrier, manifest, plan)
        for carrier, manifest in zip(carriers, manifests, strict=True)
    ) + (select_partition_payloads(empty_carrier, empty_manifest, plan),)
    result = finalize_event_selection(plan, selected)
    assert [event.event_id for event in result.events] == [
        event.event_id for event in expected.events
    ]


def test_two_phase_selection_fails_closed_on_manifest_or_payload_drift() -> None:
    _, carriers, manifests = _partitioned_selection_fixture()
    plan = plan_event_selection(manifests, max_events=3)
    selected = [
        select_partition_payloads(carrier, manifest, plan)
        for carrier, manifest in zip(carriers, manifests, strict=True)
    ]
    selected.pop()
    with pytest.raises(ValueError, match="incomplete"):
        finalize_event_selection(plan, selected)

    bad_manifest = copy.copy(manifests[0])
    object.__setattr__(bad_manifest, "observed_event_population", 999)
    with pytest.raises(ValueError, match="receipt"):
        plan_event_selection((bad_manifest, *manifests[1:]), max_events=3)


def test_group_windows_seal_take_and_destroy_exactly_once() -> None:
    group_id = "b" * 64
    managers = [
        TiDARE2ETVGroupWindowManager(
            partition_id=f"rank-{rank}",
            max_events_per_group=2,
            seed=71,
            max_open_groups=4,
        )
        for rank in range(2)
    ]
    expected = TiDARE2ETVEventReservoir(
        max_events=2, seed=71, selection_epoch=15
    )
    for index in range(12):
        managers[index % 2].offer(
            group_id=group_id,
            selection_epoch=15,
            **_event(index),
        )
        expected.offer(**_event(index))
    manifests = tuple(
        manager.seal(group_id=group_id, selection_epoch=15)
        for manager in managers
    )
    plan = plan_event_selection(manifests, max_events=2)
    selected = tuple(
        manager.take_selected(group_id=group_id, plan=plan)
        for manager in managers
    )
    result = finalize_event_selection(plan, selected)
    assert [event.event_id for event in result.events] == [
        event.event_id for event in expected.snapshot().events
    ]
    assert all(manager.open_group_count == 0 for manager in managers)
    with pytest.raises(RuntimeError, match="sealed"):
        managers[0].take_selected(group_id=group_id, plan=plan)


def test_group_window_bounds_sealing_and_checkpoint_contract() -> None:
    manager = TiDARE2ETVGroupWindowManager(
        partition_id="rank-0",
        max_events_per_group=1,
        seed=9,
        max_open_groups=1,
    )
    group_a, group_b = "c" * 64, "d" * 64
    manager.offer(group_id=group_a, selection_epoch=1, **_event(1))
    with pytest.raises(RuntimeError, match="bound"):
        manager.offer(group_id=group_b, selection_epoch=2, **_event(2))
    manifest = manager.seal(group_id=group_a, selection_epoch=1)
    assert manifest == manager.seal(group_id=group_a, selection_epoch=1)
    with pytest.raises(RuntimeError, match="sealed"):
        manager.offer(group_id=group_a, selection_epoch=1, **_event(3))
    with pytest.raises(RuntimeError, match="checkpoint"):
        manager.state_dict()
    assert manager.discard(group_a)
    state = manager.state_dict()
    manager.load_state_dict(state)
    bad = dict(state)
    bad["seed"] = 10
    with pytest.raises(ValueError, match="configuration"):
        manager.load_state_dict(bad)


def test_disabled_group_window_factory_returns_before_validation() -> None:
    assert (
        build_group_window_manager(
            enabled=False,
            partition_id="",
            max_events_per_group=0,
            seed=-1,
            max_open_groups=0,
        )
        is None
    )
