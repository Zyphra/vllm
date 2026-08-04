import torch

from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts


class _ExpertMapManager:
    local_num_experts = 2
    placement_strategy = "linear"
    expert_map = torch.tensor([0, 1, -1, -1], dtype=torch.int32)
    expert_mask = torch.tensor([1, 1, 0, 0, 0], dtype=torch.int32)
    routing_tables = (
        torch.tensor([0, 1, -1, -1], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
    )


class _RoutedExpertsBuffers(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.expert_map_manager = _ExpertMapManager()


def test_expert_maps_are_nonpersistent_buffers():
    module = _RoutedExpertsBuffers()
    RoutedExperts.update_expert_map_info(module)

    assert set(dict(module.named_buffers())) == {
        "_expert_map",
        "expert_mask",
        "expert_global_to_physical",
        "expert_physical_to_global",
        "expert_local_to_global",
    }
    assert module.state_dict() == {}
    assert module._expert_map.tolist() == [0, 1, -1, -1]
    assert module.expert_mask.tolist() == [1, 1, 0, 0, 0]
