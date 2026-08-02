from types import SimpleNamespace

import torch

from vllm.model_executor.models import zaya


def test_zaya_expert_weights_use_parameter_reload_wrapper(monkeypatch):
    calls = []

    class FakeRoutedExperts:
        def weight_loader(self, *args, **kwargs):
            raise AssertionError("model loader bypassed the parameter reload wrapper")

    module = FakeRoutedExperts()
    w13 = torch.nn.Parameter(torch.empty(2, 4, 3), requires_grad=False)
    w2 = torch.nn.Parameter(torch.empty(2, 3, 2), requires_grad=False)

    def record(param, loaded_weight, weight_name, shard_id, expert_id):
        calls.append((param, loaded_weight.clone(), weight_name, shard_id, expert_id))

    w13.weight_loader = record
    w2.weight_loader = record
    prefix = "model.layers.0.mlp.experts"
    parameters = {
        f"{prefix}.routed_experts.w13_weight": w13,
        f"{prefix}.routed_experts.w2_weight": w2,
    }
    model = SimpleNamespace(
        named_parameters=lambda: parameters.items(),
        named_buffers=lambda: (),
        named_modules=lambda: ((f"{prefix}.routed_experts", module),),
    )
    monkeypatch.setattr(zaya, "RoutedExperts", FakeRoutedExperts)
    monkeypatch.setattr(zaya, "get_tensor_model_parallel_rank", lambda: 1)

    gate_up = torch.arange(48).reshape(2, 8, 3)
    down = torch.arange(12).reshape(2, 3, 2)
    loaded = zaya.ZayaForCausalLM.load_weights(
        model,
        (
            (f"{prefix}.gate_up_proj", gate_up),
            (f"{prefix}.down_proj", down),
        ),
    )

    assert loaded == set(parameters)
    assert [(call[3], call[4]) for call in calls] == [
        ("w1", 0),
        ("w3", 0),
        ("w1", 1),
        ("w3", 1),
        ("w2", 0),
        ("w2", 1),
    ]
    assert all(call[0] is w13 for call in calls[:4])
    assert all(call[0] is w2 for call in calls[4:])
