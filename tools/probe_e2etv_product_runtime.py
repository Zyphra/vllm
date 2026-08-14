#!/usr/bin/env python3
"""H100 product-path probe for the default-off exact TiDAR event carrier.

This imports the real vLLM worker/model-runner/TiDAR classes.  It proves that
the dynamically resolved worker extension enables the rejection recorder and
the TiDAR proposal-input path together, and that proposal inputs own their
storage across the following verifier step.
"""

from __future__ import annotations

import hashlib
import json
import types

import torch

from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.spec_decode.e2etv_runtime import (
    TiDARE2ETVWorkerExtension,
)
from vllm.v1.spec_decode.tidar import TiDARProposer
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker


EXTENSION_QUALNAME = (
    "vllm.v1.spec_decode.e2etv_runtime.TiDARE2ETVWorkerExtension"
)


def _sha256_tensor(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def _new_tidar_proposer() -> TiDARProposer:
    proposer = object.__new__(TiDARProposer)
    torch.nn.Module.__init__(proposer)
    proposer.e2etv_event_inputs_enabled = False
    return proposer


def _probe_extension_enablement() -> dict[str, object]:
    extension = resolve_obj_by_qualname(EXTENSION_QUALNAME)
    assert extension is TiDARE2ETVWorkerExtension
    conflicts = [
        name
        for name in dir(extension)
        if not name.startswith("__") and hasattr(Worker, name)
    ]
    assert conflicts == [], f"worker-extension conflicts: {conflicts}"

    sampler = object.__new__(RejectionSampler)
    torch.nn.Module.__init__(sampler)
    sampler.e2etv_runtime_recorder = None
    drafter = _new_tidar_proposer()
    runner = types.SimpleNamespace(
        _e2etv_event_by_req_id=None,
        drafter=drafter,
        rejection_sampler=sampler,
    )
    runner.enable_e2etv_event_inputs = types.MethodType(
        GPUModelRunner.enable_e2etv_event_inputs, runner
    )

    # Disabled construction must not allocate carrier state or enable the
    # TiDAR input capture path.
    assert runner._e2etv_event_by_req_id is None
    assert not drafter.e2etv_event_inputs_enabled
    assert sampler.e2etv_runtime_recorder is None

    worker = TiDARE2ETVWorkerExtension()
    worker.rank = 3
    worker.model_runner = runner
    receipt = worker.e2etv_configure_runtime(
        partition_prefix="probe.replica-0",
        max_events_per_group=2,
        seed=17,
        max_open_groups=4,
        installed_policy_version=11,
    )
    assert receipt["partition_id"] == "probe.replica-0.rank-00003"
    assert runner._e2etv_event_by_req_id == {}
    assert drafter.e2etv_event_inputs_enabled
    assert sampler.e2etv_runtime_recorder is not None
    assert sampler.e2etv_runtime_recorder.installed_policy_version == 11
    return receipt


def _probe_exact_dspark_inputs() -> dict[str, object]:
    assert torch.cuda.is_available(), "product runtime probe requires CUDA"
    device = torch.device("cuda")
    torch.manual_seed(20260814)

    batch_size = 2
    num_speculative_tokens = 16
    hidden_size = 32
    markov_rank = 8
    vocab_size = 257

    proposer = _new_tidar_proposer()
    proposer.num_speculative_tokens = num_speculative_tokens
    proposer.diff_temperature = 0.8
    proposer.e2etv_event_inputs_enabled = True
    proposer.model = types.SimpleNamespace(
        diffusion_output_layer=types.SimpleNamespace(
            weight=torch.randn(
                vocab_size,
                hidden_size,
                device=device,
                dtype=torch.bfloat16,
            )
        ),
        diffusion_markov_head=types.SimpleNamespace(
            w1=torch.randn(
                vocab_size,
                markov_rank,
                device=device,
                dtype=torch.bfloat16,
            ),
            w2=torch.randn(
                markov_rank,
                vocab_size,
                device=device,
                dtype=torch.bfloat16,
            ),
        ),
        dspark_block_len=16,
    )

    hidden = torch.randn(
        batch_size * num_speculative_tokens,
        hidden_size,
        device=device,
        dtype=torch.bfloat16,
    )
    global_positions = torch.stack(
        (
            torch.arange(32, 48, device=device, dtype=torch.long),
            torch.arange(35, 51, device=device, dtype=torch.long),
        )
    )
    committed_tokens = torch.tensor([5, 7], device=device, dtype=torch.long)
    expected_hidden = hidden.clone()
    expected_positions = global_positions.clone()

    sampled = TiDARProposer._dspark_sample_drafts(
        proposer,
        hidden,
        batch_size,
        mask_positions=global_positions,
        prev_token=committed_tokens,
    ).view(batch_size, num_speculative_tokens)

    captured_hidden = proposer.last_e2etv_draft_hidden
    captured_previous = proposer.last_e2etv_previous_token_ids
    captured_positions = proposer.last_e2etv_global_positions
    assert captured_hidden is not None
    assert captured_previous is not None
    assert captured_positions is not None
    assert captured_hidden.data_ptr() != hidden.data_ptr()
    assert captured_positions.data_ptr() != global_positions.data_ptr()

    # Emulate reuse of the source buffers on the following engine step.  The
    # event payload must remain the exact proposal that is about to be
    # verified, not whatever the shared buffers now contain.
    hidden.zero_()
    global_positions.zero_()
    torch.testing.assert_close(captured_hidden, expected_hidden)
    torch.testing.assert_close(captured_positions, expected_positions)

    expected_previous = torch.empty_like(sampled)
    expected_previous[:, 0] = committed_tokens
    for index in range(num_speculative_tokens):
        at_boundary = expected_positions[:, index] % 16 == 0
        if index > 0:
            expected_previous[:, index] = sampled[:, index - 1]
        expected_previous[:, index] = torch.where(
            at_boundary,
            torch.zeros_like(expected_previous[:, index]),
            expected_previous[:, index],
        )
    torch.testing.assert_close(captured_previous, expected_previous)

    return {
        "device": torch.cuda.get_device_name(0),
        "sampled_sha256": _sha256_tensor(sampled),
        "hidden_sha256": _sha256_tensor(captured_hidden),
        "positions_sha256": _sha256_tensor(captured_positions),
        "previous_sha256": _sha256_tensor(captured_previous),
    }


def main() -> None:
    receipt = {
        "extension": _probe_extension_enablement(),
        "dspark": _probe_exact_dspark_inputs(),
    }
    print("E2ETV_PRODUCT_RUNTIME_QUALIFIED")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
