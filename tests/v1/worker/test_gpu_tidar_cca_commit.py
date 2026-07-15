# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.v1.worker.gpu.model_runner import GPUModelRunner


class _FakeRunner:
    def __init__(self) -> None:
        self._tidar_cca_commit_ctx = None
        self.layer = SimpleNamespace(
            _spec_max_P=4,
            _spec_max_S=17,
            _spec_stash_conv=torch.empty(4, 17, 1, 1),
        )
        self.committed_idx = None

    def _use_tidar(self) -> bool:
        return True

    def _get_tidar_cca_layers(self):
        return {"layer": self.layer}

    def _commit_tidar_cca_layers(self, idx_gpu: torch.Tensor) -> None:
        self.committed_idx = idx_gpu.clone()


def test_mixed_cca_commit_uses_prefill_segment_rows() -> None:
    runner = _FakeRunner()
    input_batch = SimpleNamespace(
        # One decode, then verify/prompt/verify/prompt prefill rows.
        num_scheduled_tokens=[1, 17, 75, 5, 130],
        req_ids=["decode", "verify-a", "prompt-a", "verify-b", "prompt-b"],
    )
    scheduler_output = SimpleNamespace(
        scheduled_spec_decode_tokens={
            "verify-a": list(range(16)),
            "verify-b": list(range(4)),
        }
    )

    GPUModelRunner._build_tidar_cca_commit_ctx(
        runner, input_batch, scheduler_output
    )

    ctx = runner._tidar_cca_commit_ctx
    assert ctx is not None
    # Stash rows are relative to the prefill segment. Prompt rows retain
    # their positions instead of making the verify rows compact to [0, 1].
    assert ctx["stash_rows_gpu"].tolist() == [0, 2]
    assert ctx["batch_rows"] == [1, 3]

    GPUModelRunner._commit_tidar_cca_state_from_num_sampled(
        runner, torch.tensor([1, 6, 1, 3, 1])
    )
    # num_sampled includes the final verifier sample, so candidate indices
    # are num_sampled - 1 for the two verify rows only.
    assert runner.committed_idx is not None
    assert runner.committed_idx.tolist() == [5, 2]
