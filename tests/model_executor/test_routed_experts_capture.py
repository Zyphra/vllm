# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import types

import numpy as np
import pytest
import torch

from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsReader,
    select_routed_experts_kv_group,
)
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
)

pytestmark = pytest.mark.cpu_test


class DummyRouter(BaseRouter):
    @property
    def routing_method_type(self) -> RoutingMethodType:
        return RoutingMethodType.FUSED_TOPK

    def _compute_routing(self, hidden_states, router_logits, indices_type):
        topk_ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
        topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)
        return topk_weights, topk_ids

    def _apply_eplb_mapping(self, topk_ids: torch.Tensor) -> torch.Tensor:
        # Make mapping observable without requiring CUDA EPLB path.
        return topk_ids + 10


def _make_router() -> DummyRouter:
    return DummyRouter(
        top_k=2,
        global_num_experts=16,
        eplb_state=EplbLayerState(),
        enable_eplb=False,
        indices_type_getter=None,
    )


def test_base_router_capture_pre_eplb_mapping():
    router = _make_router()
    captured = []

    def capture_fn(ids):
        captured.append(ids.clone())

    router.set_capture_fn(capture_fn)
    topk_weights, topk_ids = router.select_experts(
        hidden_states=torch.empty(1),
        router_logits=torch.empty(1),
    )

    assert topk_weights.shape == topk_ids.shape
    assert len(captured) == 1
    assert torch.equal(captured[0], torch.tensor([[1, 2], [3, 4]]))
    assert torch.equal(topk_ids, torch.tensor([[11, 12], [13, 14]]))


def test_base_router_capture_with_eplb_enabled():
    router = _make_router()
    router.enable_eplb = True
    router.eplb_state.expert_load_view = torch.zeros(32, dtype=torch.int64)
    router.eplb_state.logical_to_physical_map = torch.arange(32).view(32, 1)
    router.eplb_state.logical_replica_count = torch.ones(32, dtype=torch.int64)

    captured = []

    def capture_fn(ids):
        captured.append(ids.clone())

    router.set_capture_fn(capture_fn)
    _, topk_ids = router.select_experts(
        hidden_states=torch.empty(1),
        router_logits=torch.empty(1),
    )

    assert len(captured) == 1
    # Capture should see logical ids pre-EPLB mapping.
    assert torch.equal(captured[0], torch.tensor([[1, 2], [3, 4]]))
    # Our DummyRouter mapping adds +10.
    assert torch.equal(topk_ids, torch.tensor([[11, 12], [13, 14]]))


def test_gpu_model_runner_binds_router_capture(monkeypatch):
    from vllm.v1.worker import gpu_model_runner as gmr

    class DummyFusedMoE:
        def __init__(self):
            self.layer_id = 7
            self.router = _make_router()

    class DummyCapturer:
        def __init__(self):
            self.calls = []

        def capture(self, layer_id, topk_ids):
            self.calls.append((layer_id, topk_ids))

    dummy_module = DummyFusedMoE()

    # Patch the runtime import inside _bind_routed_experts_capturer.
    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "FusedMoE", DummyFusedMoE)

    dummy_self = types.SimpleNamespace(
        compilation_config=types.SimpleNamespace(
            static_forward_context={"dummy": dummy_module}
        )
    )

    capturer = DummyCapturer()
    gmr.GPUModelRunner._bind_routed_experts_capturer(dummy_self, capturer)

    assert dummy_module.router.capture_fn is not None
    dummy_module.router.capture_fn(torch.tensor([[5, 6]]))

    assert len(capturer.calls) == 1
    layer_id, topk_ids = capturer.calls[0]
    assert layer_id == 7
    assert torch.equal(topk_ids, torch.tensor([[5, 6]]))


def test_gpu_model_runner_binding_stage(monkeypatch):
    from vllm.v1.worker import gpu_model_runner as gmr

    class DummyFusedMoE:
        def __init__(self):
            self.layer_id = 11
            self.router = _make_router()

    class DummyCapturer:
        def __init__(self):
            self.calls = []

        def capture(self, layer_id, topk_ids):
            self.calls.append((layer_id, topk_ids))

    dummy_module = DummyFusedMoE()

    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "FusedMoE", DummyFusedMoE)

    dummy_self = types.SimpleNamespace(
        compilation_config=types.SimpleNamespace(
            static_forward_context={"dummy": dummy_module}
        )
    )

    # Before binding, no capture hook.
    assert dummy_module.router.capture_fn is None

    capturer = DummyCapturer()
    gmr.GPUModelRunner._bind_routed_experts_capturer(dummy_self, capturer)

    # After binding, hook should exist and be callable.
    assert callable(dummy_module.router.capture_fn)
    dummy_module.router.capture_fn(torch.tensor([[9, 10]]))
    assert len(capturer.calls) == 1


def _slots(block_ids: list[int], block_size: int, num_tokens: int) -> np.ndarray:
    offsets = np.arange(block_size, dtype=np.int64)
    return (
        np.asarray(block_ids, dtype=np.int64)[:, None] * block_size + offsets
    ).reshape(-1)[:num_tokens]


def test_route_key_group_uses_dense_smallest_block_space():
    def full_attention_group(block_size: int) -> KVCacheGroupSpec:
        return KVCacheGroupSpec(
            layer_names=[f"attention_{block_size}"],
            kv_cache_spec=FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float16,
            ),
        )

    groups = [
        types.SimpleNamespace(
            kv_cache_spec=types.SimpleNamespace(block_size=33824)
        ),
        full_attention_group(64),
        full_attention_group(32),
    ]
    assert select_routed_experts_kv_group(groups) == (2, 32)
    with pytest.raises(ValueError, match="full-attention"):
        select_routed_experts_kv_group(groups[:1])

    # Distinct live requests own disjoint attention blocks until completion.
    req_a = _slots(list(range(100, 135)), 32, 1098)
    req_b = _slots(list(range(200, 237)), 32, 1153)
    assert not np.intersect1d(req_a, req_b).size


def test_sparse_group0_modulo_witness_is_rejected(tmp_path):
    # Exact sealed-cache witness: five corrupt prompt-75 rows begin with the
    # 34 route rows at token 1120 of a later same-server prompt-130 request.
    # These sparse group-0 addresses differ by an exact multiple of the old
    # compact sidecar capacity, so np.remainder silently aliases them.
    capacity = 493_024
    earlier_first_slot = 23 * 33_824
    later_token_1120_slot = 125 * 33_824 + 1120
    assert later_token_1120_slot - earlier_first_slot == 7 * capacity
    assert earlier_first_slot % capacity == later_token_1120_slot % capacity

    reader = RoutedExpertsReader()
    reader._host_buffer_view = np.zeros((capacity, 1, 1), dtype=np.int32)
    lock_path = tmp_path / "routes.lock"
    lock_path.touch()
    reader._lock_file = str(lock_path)

    # The fixed reader must fail closed instead of returning another live
    # request's routes from a wrapped index.
    with pytest.raises(RuntimeError, match="refusing modulo alias"):
        reader.get_routed_experts(np.asarray([later_token_1120_slot]))


def test_dense_route_slots_survive_concurrent_request_lifetimes(tmp_path):
    block_size = 32
    capacity = 300 * block_size
    req_a = _slots(list(range(10, 45)), block_size, 1098)
    req_b = _slots(list(range(100, 137)), block_size, 1153)
    assert int(max(req_a.max(), req_b.max())) < capacity

    reader = RoutedExpertsReader()
    reader._host_buffer_view = np.zeros((capacity, 1, 1), dtype=np.int32)
    lock_path = tmp_path / "routes.lock"
    lock_path.touch()
    reader._lock_file = str(lock_path)
    reader._host_buffer_view[req_a, 0, 0] = 7
    reader._host_buffer_view[req_b, 0, 0] = 11

    # Writing B cannot change A while both requests own disjoint live blocks.
    routes_a = reader.get_routed_experts(req_a)
    assert np.all(routes_a[:, 0, 0] == 7)
    assert np.all(reader.get_routed_experts(req_b)[:, 0, 0] == 11)

    # Scheduler reads and copies before freeing A. Simulated later reuse of
    # A's blocks therefore cannot mutate the routes already returned.
    reader._host_buffer_view[req_a, 0, 0] = 19
    assert np.all(routes_a[:, 0, 0] == 7)
    assert np.all(reader.get_routed_experts(req_a)[:, 0, 0] == 19)
