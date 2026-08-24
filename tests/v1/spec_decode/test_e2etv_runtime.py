import types

import pytest
import torch

from vllm.v1.spec_decode.e2etv_event_inputs import (
    TiDARE2ETVEventBatch,
    attach_target_hidden,
)
from vllm.v1.spec_decode.e2etv_event_reservoir import (
    finalize_event_selection,
    plan_event_selection,
)
from vllm.v1.spec_decode.e2etv_runtime import (
    TiDARE2ETVRequestContext,
    TiDARE2ETVRuntimeRecorder,
    TiDARE2ETVWorkerExtension,
)

GROUP = "a" * 64


def _request(group: str = GROUP, epoch: int = 7, opaque: str = "request0") -> str:
    return TiDARE2ETVRequestContext(group, epoch, opaque).encode()


def _batch(request_ids: tuple[str, ...]) -> tuple[TiDARE2ETVEventBatch, torch.Tensor]:
    counts = tuple(2 for _ in request_ids)
    total = sum(counts)
    batch = TiDARE2ETVEventBatch(
        req_ids=request_ids,
        num_draft_tokens=counts,
        draft_hidden=torch.arange(total * 4, dtype=torch.bfloat16).reshape(total, 4),
        previous_token_ids=torch.arange(total, dtype=torch.int64),
        global_positions=torch.cat(
            [torch.arange(10 * i, 10 * i + 2) for i in range(len(request_ids))]
        ).to(torch.int64),
        target_hidden=torch.arange(total * 4, dtype=torch.bfloat16).reshape(total, 4)
        + 100,
    )
    logits = torch.arange(total * 5, dtype=torch.float32).reshape(total, 5)
    return batch, logits


def _recorder(*, version: int = 3, max_events: int = 2):
    return TiDARE2ETVRuntimeRecorder(
        partition_id="replica-0.rank-00000",
        max_events_per_group=max_events,
        seed=19,
        max_open_groups=4,
        installed_policy_version=version,
    )


def test_request_context_round_trip_and_non_context_passthrough():
    encoded = _request(epoch=0x1234, opaque="abcdef")
    assert TiDARE2ETVRequestContext.decode(encoded) == TiDARE2ETVRequestContext(
        GROUP, 0x1234, "abcdef"
    )
    assert TiDARE2ETVRequestContext.decode("ordinary-vllm-request") is None


@pytest.mark.parametrize(
    "request_id",
    [
        "e2tv1.short.0000000000000001.request",
        f"e2tv1.{GROUP}.1.request",
        f"e2tv1.{GROUP}.000000000000000g.request",
        f"e2tv1.{GROUP}.0000000000000001.bad.request",
    ],
)
def test_request_context_rejects_malformed_values(request_id):
    with pytest.raises(ValueError):
        TiDARE2ETVRequestContext.decode(request_id)


def test_recorder_skips_unscoped_requests_and_preserves_compact_event_state():
    recorder = _recorder(max_events=1)
    batch, logits = _batch((_request(), "ordinary-request"))
    recorder.record(batch, draft_temperature=0.8, target_temperature=1.0)

    manifest = recorder.seal_group(group_id=GROUP, selection_epoch=7)
    assert manifest.observed_event_population == 1
    assert len(manifest.descriptors) == 1
    plan = plan_event_selection((manifest,), max_events=1)
    selected = recorder.take_group(group_id=GROUP, plan=plan)
    carrier = finalize_event_selection(plan, (selected,))
    assert carrier.observed_event_population == 1
    assert len(carrier.events) == 1
    assert carrier.events[0].installed_policy_version == 3
    assert carrier.events[0].target_temperature == 1.0
    assert torch.equal(carrier.events[0].target_hidden, batch.target_hidden[:2])
    assert recorder.window_manager.open_group_count == 0


def test_attach_target_hidden_uses_exact_rejection_rows() -> None:
    batch, _ = _batch((_request(),))
    batch.target_hidden = None
    sample_hidden = torch.arange(6 * 4, dtype=torch.bfloat16).view(6, 4)
    attached = attach_target_hidden(
        batch,
        sample_hidden_states=sample_hidden,
        target_logits_indices=torch.tensor([1, 4], dtype=torch.int32),
    )
    assert batch.target_hidden is None
    torch.testing.assert_close(attached.target_hidden, sample_hidden[[1, 4]])
    with pytest.raises(RuntimeError, match="more than once"):
        attach_target_hidden(
            attached,
            sample_hidden_states=sample_hidden,
            target_logits_indices=torch.tensor([1, 4]),
        )


def test_recorder_requires_target_hidden() -> None:
    recorder = _recorder(max_events=1)
    batch, logits = _batch((_request(),))
    batch.target_hidden = None
    with pytest.raises(RuntimeError, match="target hidden"):
        recorder.record(batch, draft_temperature=0.8, target_temperature=1.0)


def test_recorder_preserves_per_request_temperatures() -> None:
    recorder = _recorder(max_events=2)
    batch, _ = _batch(
        (_request(opaque="first"), _request(opaque="second"))
    )
    recorder.record(
        batch,
        draft_temperature=[0.7, 0.8],
        target_temperature=[0.9, 1.0],
    )
    manifest = recorder.seal_group(group_id=GROUP, selection_epoch=7)
    plan = plan_event_selection((manifest,), max_events=2)
    selected = recorder.take_group(group_id=GROUP, plan=plan)
    events = finalize_event_selection(plan, (selected,)).events
    observed = {
        event.request_id: (event.draft_temperature, event.target_temperature)
        for event in events
    }
    assert observed == {
        _request(opaque="first"): (0.7, 0.9),
        _request(opaque="second"): (0.8, 1.0),
    }

    with pytest.raises(ValueError, match="one value per request"):
        recorder.record(
            batch,
            draft_temperature=[0.8],
            target_temperature=[1.0, 1.0],
        )


def test_policy_version_is_sampled_at_the_verifier_event_boundary():
    recorder = _recorder(version=4)
    first_batch, first_logits = _batch((_request(epoch=7, opaque="first"),))
    recorder.record(first_batch, draft_temperature=0.8, target_temperature=1.0)
    recorder.set_installed_policy_version(5)
    second_group = "b" * 64
    second_batch, second_logits = _batch(
        (_request(group=second_group, epoch=8, opaque="second"),)
    )
    recorder.record(second_batch, draft_temperature=0.8, target_temperature=1.0)

    versions = []
    for group, epoch in ((GROUP, 7), (second_group, 8)):
        manifest = recorder.seal_group(group_id=group, selection_epoch=epoch)
        plan = plan_event_selection((manifest,), max_events=1)
        selected = recorder.take_group(group_id=group, plan=plan)
        carrier = finalize_event_selection(plan, (selected,))
        versions.append(carrier.events[0].installed_policy_version)
    assert versions == [4, 5]


def test_discard_destroys_group_and_checkpoint_requires_no_open_windows():
    recorder = _recorder()
    batch, logits = _batch((_request(),))
    recorder.record(batch, draft_temperature=0.8, target_temperature=1.0)
    with pytest.raises(RuntimeError, match="open E2E-TV groups"):
        recorder.state_dict()
    assert recorder.discard_group(GROUP)
    assert not recorder.discard_group(GROUP)
    state = recorder.state_dict()
    assert state["installed_policy_version"] == 3
    assert state["observed_events"] == 1


def test_worker_extension_is_explicit_and_receipt_bound():
    class FakeRejectionSampler:
        e2etv_runtime_recorder = None

        def configure_e2etv_runtime(self, **kwargs):
            self.e2etv_runtime_recorder = TiDARE2ETVRuntimeRecorder(**kwargs)

    worker = TiDARE2ETVWorkerExtension()
    worker.rank = 2
    enabled = []
    worker.model_runner = types.SimpleNamespace(
        rejection_sampler=FakeRejectionSampler(),
        enable_e2etv_event_inputs=lambda: enabled.append(True),
    )
    receipt = worker.e2etv_configure_runtime(
        partition_prefix="replica-3.server-1",
        max_events_per_group=2,
        seed=19,
        max_open_groups=4,
        installed_policy_version=6,
    )
    assert receipt == {
        "enabled": True,
        "partition_id": "replica-3.server-1.rank-00002",
        "max_events_per_group": 2,
        "seed": 19,
        "max_open_groups": 4,
        "installed_policy_version": 6,
    }
    assert enabled == [True]
    assert worker.e2etv_set_installed_policy_version(7) == 7


def test_worker_extension_supports_v2_model_runner_owner():
    class FakeV2ModelRunner:
        e2etv_runtime_recorder = None

        def __init__(self):
            self.enabled = False

        def configure_e2etv_runtime(self, **kwargs):
            self.e2etv_runtime_recorder = TiDARE2ETVRuntimeRecorder(**kwargs)

        def enable_e2etv_event_inputs(self):
            self.enabled = True

    worker = TiDARE2ETVWorkerExtension()
    worker.rank = 3
    worker.model_runner = FakeV2ModelRunner()
    receipt = worker.e2etv_configure_runtime(
        partition_prefix="replica-4.server-2",
        max_events_per_group=3,
        seed=23,
        max_open_groups=5,
        installed_policy_version=8,
    )
    assert receipt["partition_id"] == "replica-4.server-2.rank-00003"
    assert worker.model_runner.enabled
    assert worker.e2etv_set_installed_policy_version(9) == 9


def test_mixed_group_batch_keeps_windows_isolated():
    recorder = _recorder(max_events=1)
    second_group = "c" * 64
    batch, logits = _batch(
        (
            _request(group=GROUP, epoch=7, opaque="first"),
            _request(group=second_group, epoch=8, opaque="second"),
        )
    )
    recorder.record(batch, draft_temperature=0.8, target_temperature=1.0)
    assert recorder.window_manager.open_group_count == 2
    first = recorder.seal_group(group_id=GROUP, selection_epoch=7)
    second = recorder.seal_group(group_id=second_group, selection_epoch=8)
    assert first.observed_event_population == second.observed_event_population == 1
    assert first.descriptors[0].event_id != second.descriptors[0].event_id
