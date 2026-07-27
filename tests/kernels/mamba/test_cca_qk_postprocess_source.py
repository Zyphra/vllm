# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]


def _read(*parts: str) -> str:
    return REPO_ROOT.joinpath(*parts).read_text()


def _load_python_function(name: str):
    source = _read("vllm", "_custom_ops.py")
    tree = ast.parse(source)
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    function.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))

    class Tensor:
        pass

    calls = []

    def op(*args):
        calls.append(args)

    torch = SimpleNamespace(
        Tensor=Tensor,
        ops=SimpleNamespace(
            _rocm_C=SimpleNamespace(cca_qk_postprocess=op),
        ),
    )
    namespace = {"torch": torch}
    exec(compile(module, "vllm/_custom_ops.py", "exec"), namespace)
    return namespace[name], calls


def test_python_wrapper_uses_the_mutation_only_four_tensor_abi():
    wrapper, calls = _load_python_function("cca_qk_postprocess")
    tensors = tuple(object() for _ in range(4))

    assert wrapper(*tensors) is None
    assert calls == [tensors]

    fake, fake_calls = _load_python_function("_cca_qk_postprocess_fake")
    assert fake(*tensors) is None
    assert fake_calls == []


def test_native_op_is_built_and_registered_with_a_mandatory_output():
    cmake = _read("CMakeLists.txt")
    header = _read("csrc", "rocm", "ops.h")
    bindings = _read("csrc", "rocm", "torch_bindings.cpp")
    custom_ops = _read("vllm", "_custom_ops.py")

    assert '"csrc/rocm/cca_qk_postprocess.cu"' in cmake
    assert "void cca_qk_postprocess(" in header
    assert (
        '"cca_qk_postprocess(Tensor grouped, Tensor first, Tensor temp, "' in bindings
    )
    assert '"Tensor(a!) out) -> ()"' in bindings
    assert (
        'rocm_ops.impl("cca_qk_postprocess", torch::kCUDA, '
        "&cca_qk_postprocess);" in bindings
    )
    assert '@register_fake("_rocm_C::cca_qk_postprocess")' in custom_ops
    assert 'hasattr(torch.ops._rocm_C, "cca_qk_postprocess")' in custom_ops


def test_kernel_preserves_exact_math_and_fails_closed():
    source = _read("csrc", "rocm", "cca_qk_postprocess.cu")

    for contract in (
        "constexpr int kNumQueryHeads = 8;",
        "constexpr int kNumKeyHeads = 2;",
        "constexpr int kHeadDim = 128;",
        "constexpr int kGqaGroups = 4;",
        "constexpr int kWidth = 1280;",
        "constexpr int kValuesPerLane = 4;",
        "constexpr int kReductionLanes = kHeadDim / kValuesPerLane;",
        "static_assert(kReductionLanes == 32);",
        "rows > 0",
        "std::numeric_limits<int>::max()",
        "grouped.is_cuda()",
        "first.is_cuda()",
        "temp.is_cuda()",
        "out.is_cuda()",
        "grouped.device() == first.device()",
        "grouped.device() == temp.device()",
        "grouped.device() == out.device()",
        "grouped.scalar_type() == torch::kBFloat16",
        "first.scalar_type() == torch::kBFloat16",
        "temp.scalar_type() == torch::kBFloat16",
        "out.scalar_type() == torch::kBFloat16",
        "!out.is_alias_of(grouped)",
        "!out.is_alias_of(first)",
        "!out.is_alias_of(temp)",
        'check_matrix_layout(first, rows, "first")',
        'check_matrix_layout(out, rows, "out")',
        "temp.dim() == 1 && temp.size(0) == kNumKeyHeads",
        'architecture.rfind("gfx942", 0) == 0',
        "properties->maxGridSize[0]",
        "tensor.stride(1) == 1 && tensor.stride(0) >= kWidth",
        "temp.stride(0) == 1",
        "temp.data_ptr<at::BFloat16>()",
        "at::cuda::OptionalCUDAGuard",
        "C10_CUDA_KERNEL_LAUNCH_CHECK();",
    ):
        assert contract in source

    assert "PYBIND11_MODULE" not in source
    assert "torch::empty" not in source
    assert "optional<torch::Tensor>" not in source
    assert "bool clamp_temp" not in source
    assert "double sqrt_head_dim" not in source
    assert ".to(" not in source
    assert source.count("value += 0.5f") == 4

    reduction = (
        source.index("const int channel_base = lane * kValuesPerLane"),
        source.index("const int channel = channel_base + i"),
        source.index("values[i] = value"),
        source.index("volatile float lane_sum = values[0] * values[0]"),
        source.index("for (int i = 1; i < kValuesPerLane; ++i)"),
        source.index("lane_sum = lane_sum + values[i] * values[i]"),
        source.index("for (int offset = 1; offset < kReductionLanes; offset <<= 1)"),
        source.index("const float other = __shfl_down(sum, offset)"),
        source.index("sum = __shfl(sum, 0)"),
    )
    normalization = (
        source.index("sqrtf(sum)"),
        source.index("norm_squared = norm * norm"),
        source.index("stabilized = norm_squared + kNormEps"),
        source.index("rsqrtf(stabilized)"),
        source.index("values[i] * inverse_norm"),
        source.index("normalized * kSqrtHeadDim"),
        source.index("normalized * key_temp"),
    )
    launch = (
        source.index("at::cuda::OptionalCUDAGuard"),
        source.index("at::cuda::getCurrentDeviceProperties"),
        source.index("at::cuda::getCurrentCUDAStream"),
        source.index("cca_qk_postprocess_kernel<<<"),
        source.index("C10_CUDA_KERNEL_LAUNCH_CHECK();"),
    )
    assert reduction == tuple(sorted(reduction))
    assert normalization == tuple(sorted(normalization))
    assert launch == tuple(sorted(launch))
    assert "dim3(kReductionLanes), 0, stream" in source
    assert "__shared__" not in source
