# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu import dp_utils
from vllm.v1.worker.gpu.cudagraph_utils import (
    CudaGraphBatchType,
    CudaGraphKey,
    CudaGraphManager,
)


def _manager() -> CudaGraphManager:
    manager = object.__new__(CudaGraphManager)
    manager._tidar_tokens_per_verify_req = 17
    manager.max_num_reqs = 32
    manager.dp_size = 1
    manager.cudagraph_sizes = {
        17: 17,
        34: 34,
        544: 544,
    }
    manager.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    return manager


def test_tidar_single_verify_uses_geometry_specific_graph() -> None:
    manager = _manager()
    assert manager.get_cudagraph_key(17, [17]) == CudaGraphKey(
        17, CudaGraphBatchType.TIDAR_VERIFY)


def test_tidar_decode_uses_distinct_ambiguous_graph_key() -> None:
    manager = _manager()
    assert manager.get_cudagraph_key(17, [1] * 17) == CudaGraphKey(17)


def test_tidar_multi_verify_uses_exact_geometry() -> None:
    manager = _manager()
    assert manager.get_cudagraph_key(34, [17, 17]) == CudaGraphKey(
        34, CudaGraphBatchType.TIDAR_VERIFY)


def test_tidar_full_verify_population_uses_graph() -> None:
    manager = _manager()
    assert manager.get_cudagraph_key(544, [17] * 32) == CudaGraphKey(
        544, CudaGraphBatchType.TIDAR_VERIFY)


def test_tidar_partial_or_mixed_batches_use_piecewise_fallback() -> None:
    manager = _manager()
    assert manager.get_cudagraph_size(16, [16]) is None
    assert manager.get_cudagraph_size(18, [1, 17]) is None
    assert manager.get_non_full_runtime_mode() == CUDAGraphMode.PIECEWISE


def test_tidar_prefill_uses_piecewise_fallback() -> None:
    manager = _manager()
    assert manager.get_cudagraph_size(257, [257]) is None
    assert manager.get_non_full_runtime_mode() == CUDAGraphMode.PIECEWISE


def test_non_full_runtime_mode_respects_full_only_configuration() -> None:
    manager = _manager()
    manager.cudagraph_mode = CUDAGraphMode.FULL
    assert manager.get_non_full_runtime_mode() == CUDAGraphMode.NONE


def test_non_full_runtime_mode_respects_none_configuration() -> None:
    manager = _manager()
    manager.cudagraph_mode = CUDAGraphMode.NONE
    assert manager.get_non_full_runtime_mode() == CUDAGraphMode.NONE


def test_tidar_capture_keys_distinguish_same_token_count() -> None:
    manager = _manager()
    decode_key = CudaGraphKey(17)
    verify_key = CudaGraphKey(17, CudaGraphBatchType.TIDAR_VERIFY)
    assert {decode_key, verify_key} <= manager._get_capture_keys()
    assert manager._get_num_reqs_for_capture(decode_key) == 17
    assert manager._get_num_reqs_for_capture(verify_key) == 1
    assert manager.get_padded_cudagraph_key(17, verify_key) == verify_key


def test_tidar_dp_verify_full_graphs_stay_disabled() -> None:
    manager = _manager()
    manager.dp_size = 2
    assert manager.get_cudagraph_key(34, [17, 17]) is None
    assert manager.get_non_full_runtime_mode() == CUDAGraphMode.PIECEWISE
    assert all(
        key.batch_type != CudaGraphBatchType.TIDAR_VERIFY
        for key in manager._get_capture_keys()
    )


def test_tidar_dp_decode_verify_mismatch_forces_eager(monkeypatch) -> None:
    def _batch_metadata(*args, **kwargs):
        del args, kwargs
        # Rank 0 selected decode graph 17; rank 1's verifier returned None.
        return (
            torch.tensor([17, 34], dtype=torch.int32),
            torch.tensor([17, -1], dtype=torch.int32),
            torch.tensor([0, 0], dtype=torch.int32),
        )

    monkeypatch.setattr(
        dp_utils, "get_batch_metadata_across_dp", _batch_metadata)
    use_graph, padded, tokens_across, _ = (
        dp_utils.get_cudagraph_and_dp_padding(
            num_tokens=17,
            cudagraph_size=17,
            dp_size=2,
            dp_rank=0,
        )
    )
    assert not use_graph
    assert padded == 17
    assert tokens_across.tolist() == [17, 34]
