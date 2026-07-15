# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest

from vllm.v1.worker.gpu.model_runner import (
    GPUModelRunner,
    _assert_tidar_prefill_verify_segregation,
)


def test_tidar_prefill_verify_segregation_rejects_mixed_batch() -> None:
    with pytest.raises(
        RuntimeError,
        match=r"prompt_request_ids=\['prompt'\].*verify_request_ids=\['verify'\]",
    ):
        _assert_tidar_prefill_verify_segregation(
            req_ids=["verify", "prompt"],
            draft_tokens={"verify": [1, 2, 3]},
            computed_tokens=np.array([128, 0]),
            prefill_lens=np.array([128, 64]),
        )


def test_tidar_prefill_verify_segregation_allows_warm_nonverify_rows() -> None:
    _assert_tidar_prefill_verify_segregation(
        req_ids=["verify", "warm"],
        draft_tokens={"verify": [1, 2, 3]},
        computed_tokens=np.array([128, 64]),
        prefill_lens=np.array([128, 64]),
    )


def test_sample_tokens_without_execute_state_preserves_original_error() -> None:
    runner = object.__new__(GPUModelRunner)
    runner.execute_model_state = None

    assert GPUModelRunner.sample_tokens(runner, None) is None
