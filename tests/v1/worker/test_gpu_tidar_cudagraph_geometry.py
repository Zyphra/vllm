# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu import dp_utils
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager


def _manager() -> CudaGraphManager:
    manager = object.__new__(CudaGraphManager)
    manager._tidar_tokens_per_verify_req = 17
    manager.max_num_reqs = 32
    manager.cudagraph_sizes = {17: 17, 34: 34}
    manager.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    return manager


def test_tidar_single_verify_avoids_ambiguous_decode_graph() -> None:
    manager = _manager()
    assert manager.get_cudagraph_size(17, [17]) is None


def test_tidar_decode_keeps_ambiguous_graph_key() -> None:
    manager = _manager()
    assert manager.get_cudagraph_size(17, [1] * 17) == 17


def test_tidar_multi_verify_stays_eager() -> None:
    manager = _manager()
    assert manager.get_cudagraph_size(34, [17, 17]) is None


def test_tidar_full_verify_population_stays_eager() -> None:
    manager = _manager()
    manager.cudagraph_sizes[544] = 544
    assert manager.get_cudagraph_size(544, [17] * 32) is None


def test_tidar_partial_or_mixed_batches_stay_eager() -> None:
    manager = _manager()
    assert manager.get_cudagraph_size(16, [16]) is None
    assert manager.get_cudagraph_size(18, [1, 17]) is None


def test_tidar_target_capture_geometry_is_decode_only() -> None:
    manager = _manager()
    assert manager._get_num_reqs_for_capture(17) == 17
    assert manager._get_num_reqs_for_capture(32) == 32


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
