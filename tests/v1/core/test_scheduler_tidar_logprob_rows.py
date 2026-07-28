# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from vllm.v1.core.sched.scheduler import _assert_sampled_logprob_rows
from vllm.v1.outputs import LogprobsLists


def _rows(sampled_ids: list[int]) -> LogprobsLists:
    ids = np.asarray([[token_id, token_id + 1000] for token_id in sampled_ids])
    logprobs = np.asarray([[-0.1, -2.0] for _ in sampled_ids], dtype=np.float32)
    ranks = np.zeros(len(sampled_ids), dtype=np.int32)
    return LogprobsLists(ids, logprobs, ranks)


@pytest.mark.parametrize(
    ("req_id", "emitted"),
    [
        ("reject-all", [11]),
        ("accept-three", [21, 22, 23, 24]),
        ("accept-k", list(range(40, 57))),
    ],
)
def test_tidar_logprob_rows_accept_variable_prefixes(req_id: str, emitted: list[int]):
    _assert_sampled_logprob_rows(req_id, emitted, _rows(emitted))


def test_tidar_logprob_rows_accept_stop_truncation():
    expanded = _rows([51, 52, 53, 54])
    _assert_sampled_logprob_rows("eos", [51, 52], expanded.slice_request(0, 2))


def test_tidar_logprob_rows_reject_request_offset_drift():
    with pytest.raises(RuntimeError, match="token identity mismatch"):
        _assert_sampled_logprob_rows("out-of-order", [71, 72], _rows([81, 82]))


def test_tidar_logprob_rows_reject_shape():
    with pytest.raises(RuntimeError, match="row-count mismatch"):
        _assert_sampled_logprob_rows("short", [1, 2], _rows([1]))


@pytest.mark.parametrize(
    ("logprob", "rank", "subtype"),
    [
        (np.nan, -1, "contains_nan"),
        (-np.inf, -2, "all_negative_infinity"),
        (-np.inf, -3, "positive_infinity_sample_mismatch"),
    ],
)
def test_tidar_logprob_rows_report_nonfinite_subtype(
    logprob: float,
    rank: int,
    subtype: str,
):
    bad = _rows([1, 2])
    bad.logprobs[1, 0] = logprob
    bad.sampled_token_ranks[1] = rank
    expected = (
        rf"request diagnostic at row 1: sampled_token=2, "
        rf"logprob=(nan|-inf), rank={rank}, subtype={subtype}"
    )
    with pytest.raises(RuntimeError, match=expected):
        _assert_sampled_logprob_rows("diagnostic", [1, 2], bad)
