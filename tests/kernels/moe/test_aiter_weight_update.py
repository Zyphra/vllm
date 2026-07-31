from unittest.mock import patch

import torch

from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
    _shuffle_aiter_weights_in_place,
)


def test_aiter_weight_update_shuffles_per_expert_in_place() -> None:
    w13 = torch.arange(3 * 4 * 8).reshape(3, 4, 8).float()
    w2 = torch.arange(3 * 8 * 4).reshape(3, 8, 4).float()
    original_w13 = w13.clone()
    original_w2 = w2.clone()
    pointers = (w13.data_ptr(), w2.data_ptr())
    shapes = []

    def shuffle(expert_weight: torch.Tensor) -> torch.Tensor:
        assert expert_weight.ndim == 2
        shapes.append(tuple(expert_weight.shape))
        return expert_weight.flip(-1).clone()

    target = (
        "vllm.model_executor.layers.fused_moe."
        "unquantized_fused_moe_method.rocm_aiter_ops.shuffle_weight"
    )
    with patch(target, side_effect=shuffle):
        _shuffle_aiter_weights_in_place(w13, w2)

    assert (w13.data_ptr(), w2.data_ptr()) == pointers
    assert torch.equal(w13, original_w13.flip(-1))
    assert torch.equal(w2, original_w2.flip(-1))
    assert shapes == [(4, 8)] * 3 + [(8, 4)] * 3


def test_aiter_skips_padding_discarded_by_shuffle() -> None:
    method = object.__new__(UnquantizedFusedMoEMethod)
    method.unquantized_backend = UnquantizedMoeBackend.AITER
    weight = torch.zeros(2, 16, 32)

    with patch("torch.nn.functional.pad") as pad:
        assert method._maybe_pad_weight(weight) is weight
    pad.assert_not_called()
