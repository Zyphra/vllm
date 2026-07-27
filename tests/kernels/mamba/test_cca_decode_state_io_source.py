# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
QK_WIDTH = 1280
VALUE_WIDTH = 128
OUTPUT_WIDTH = 1536
PAD_SLOT_ID = -1


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

    calls: list[tuple] = []

    def prepare(*args):
        calls.append(("prepare", *args))

    def commit(*args):
        calls.append(("commit", *args))

    torch = SimpleNamespace(
        Tensor=Tensor,
        ops=SimpleNamespace(
            _rocm_C=SimpleNamespace(
                cca_decode_state_prepare=prepare,
                cca_decode_state_commit=commit,
            ),
        ),
    )
    namespace = {"torch": torch}
    exec(compile(module, "vllm/_custom_ops.py", "exec"), namespace)
    return namespace[name], calls


def _fp32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounding = np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return ((bits + rounding) >> 16).astype(np.uint16)


def _bf16_bits_to_fp32(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16
    return bits.view(np.float32)


def _prepare_reference(
    qk0: np.ndarray,
    v_current: np.ndarray,
    conv_state: np.ndarray,
    recurrent_state: np.ndarray,
    state_idx: np.ndarray,
    qkv_out: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rows = state_idx.size
    conv_window = np.empty((rows, QK_WIDTH, 3), dtype=np.uint16)
    conv_tail = np.empty((rows, QK_WIDTH), dtype=np.float32)
    for row, slot in enumerate(state_idx):
        if slot == PAD_SLOT_ID:
            conv_window[row].fill(0)
            conv_tail[row].fill(0)
            qkv_out[row, QK_WIDTH:].fill(0)
            continue
        conv_window[row, :, 0] = _fp32_to_bf16_bits(conv_state[slot, :, 0])
        conv_window[row, :, 1] = _fp32_to_bf16_bits(conv_state[slot, :, 1])
        conv_window[row, :, 2] = qk0[row]
        conv_tail[row] = conv_state[slot, :, 1]
        qkv_out[row, QK_WIDTH : QK_WIDTH + VALUE_WIDTH] = v_current[row]
        qkv_out[row, QK_WIDTH + VALUE_WIDTH :] = _fp32_to_bf16_bits(
            recurrent_state[slot]
        )
    return conv_window, conv_tail


def _commit_reference(
    qk0: np.ndarray,
    v_delayed_current: np.ndarray,
    state_idx: np.ndarray,
    conv_tail: np.ndarray,
    conv_state: np.ndarray,
    recurrent_state: np.ndarray,
    qkv_out: np.ndarray,
) -> None:
    for row, slot in enumerate(state_idx):
        if slot == PAD_SLOT_ID:
            conv_state[0].fill(0)
            recurrent_state[0].fill(0)
            qkv_out[row].fill(0)
            continue
        conv_state[slot, :, 0] = conv_tail[row]
        conv_state[slot, :, 1] = _bf16_bits_to_fp32(qk0[row])
        recurrent_state[slot] = _bf16_bits_to_fp32(v_delayed_current[row])


def test_python_wrappers_and_fakes_use_mutation_only_abis():
    prepare, calls = _load_python_function("cca_decode_state_prepare")
    prepare_args = tuple(object() for _ in range(8))
    assert prepare(*prepare_args) is None
    assert calls == [("prepare", *prepare_args)]

    commit, calls = _load_python_function("cca_decode_state_commit")
    commit_args = tuple(object() for _ in range(7))
    assert commit(*commit_args) is None
    assert calls == [("commit", *commit_args)]

    prepare_fake, calls = _load_python_function("_cca_decode_state_prepare_fake")
    assert prepare_fake(*prepare_args) is None
    assert calls == []

    commit_fake, calls = _load_python_function("_cca_decode_state_commit_fake")
    assert commit_fake(*commit_args) is None
    assert calls == []


def test_mutation_only_wrappers_compile_as_one_full_graph():
    import torch

    definitions = torch.library.Library("_rocm_C", "FRAGMENT")
    definitions.define(
        "cca_decode_state_prepare("
        "Tensor qk0, Tensor v_current, Tensor conv_state, "
        "Tensor recurrent_state, Tensor state_idx, "
        "Tensor(a!) conv_window, Tensor(b!) conv_tail, "
        "Tensor(c!) qkv_out) -> ()"
    )
    definitions.define(
        "cca_decode_state_commit("
        "Tensor qk0, Tensor v_delayed_current, Tensor state_idx, "
        "Tensor conv_tail, Tensor(a!) conv_state, "
        "Tensor(b!) recurrent_state, Tensor(c!) qkv_out) -> ()"
    )

    implementations = torch.library.Library("_rocm_C", "IMPL", "CPU")
    implementations.impl("cca_decode_state_prepare", lambda *args: None)
    implementations.impl("cca_decode_state_commit", lambda *args: None)

    @torch.library.register_fake("_rocm_C::cca_decode_state_prepare")
    def prepare_fake(*args):
        return None

    @torch.library.register_fake("_rocm_C::cca_decode_state_commit")
    def commit_fake(*args):
        return None

    def load_wrapper(name: str):
        tree = ast.parse(_read("vllm", "_custom_ops.py"))
        function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        function.decorator_list = []
        module = ast.fix_missing_locations(
            ast.Module(body=[function], type_ignores=[])
        )
        namespace = {"torch": torch}
        exec(compile(module, "vllm/_custom_ops.py", "exec"), namespace)
        return namespace[name]

    prepare = load_wrapper("cca_decode_state_prepare")
    commit = load_wrapper("cca_decode_state_commit")

    def state_io(
        qk0,
        v_current,
        v_delayed_current,
        conv_state,
        recurrent_state,
        state_idx,
        conv_window,
        conv_tail,
        qkv_out,
    ):
        prepare(
            qk0,
            v_current,
            conv_state,
            recurrent_state,
            state_idx,
            conv_window,
            conv_tail,
            qkv_out,
        )
        commit(
            qk0,
            v_delayed_current,
            state_idx,
            conv_tail,
            conv_state,
            recurrent_state,
            qkv_out,
        )
        return qkv_out

    tensors = tuple(torch.zeros(1) for _ in range(9))
    explanation = torch._dynamo.explain(state_io)(*tensors)
    assert explanation.graph_count == 1
    assert explanation.graph_break_count == 0
    compiled = torch.compile(state_io, backend="eager", fullgraph=True)
    assert compiled(*tensors) is tensors[-1]


def test_native_ops_are_built_and_registered_with_explicit_mutations():
    cmake = _read("CMakeLists.txt")
    cmake_utils = _read("cmake", "utils.cmake")
    header = _read("csrc", "rocm", "ops.h")
    bindings = _read("csrc", "rocm", "torch_bindings.cpp")
    custom_ops = _read("vllm", "_custom_ops.py")

    assert '"csrc/rocm/cca_decode_state_io.cu"' in cmake
    assert (
        'list(FILTER SRCS EXCLUDE REGEX "[.](cc|cpp|hip)$")' in cmake_utils
    )
    assert (
        'list(FILTER CXX_SRCS INCLUDE REGEX "[.](cc|cpp|hip)$")' in cmake_utils
    )
    assert "void cca_decode_state_prepare(" in header
    assert "void cca_decode_state_commit(" in header
    assert '"Tensor(a!) conv_window, Tensor(b!) conv_tail, "' in bindings
    assert '"Tensor(c!) qkv_out) -> ()"' in bindings
    assert '"Tensor state_idx, Tensor conv_tail, Tensor(a!) conv_state, "' in bindings
    assert '"Tensor(b!) recurrent_state, Tensor(c!) qkv_out) -> ()"' in bindings
    assert 'rocm_ops.impl("cca_decode_state_prepare", torch::kCUDA,' in bindings
    assert 'rocm_ops.impl("cca_decode_state_commit", torch::kCUDA,' in bindings
    assert '@register_fake("_rocm_C::cca_decode_state_prepare")' in custom_ops
    assert '@register_fake("_rocm_C::cca_decode_state_commit")' in custom_ops


def test_kernel_is_allocation_free_fixed_geometry_and_fails_closed():
    source = _read("csrc", "rocm", "cca_decode_state_io.cu")

    for contract in (
        "constexpr int kQkWidth = 1280;",
        "constexpr int kValueWidth = 128;",
        "constexpr int kOutputWidth = 1536;",
        "constexpr int kConvStateWidth = 2;",
        "constexpr int kConvWindowWidth = 3;",
        "constexpr int kPadSlotId = -1;",
        "__builtin_trap();",
        "tensor.stride(1) == 1 && tensor.stride(0) >= width",
        "tensor.stride(2) == 1",
        "tensor.is_contiguous()",
        "state_idx must be int32",
        "mutable outputs must not share storage",
        "__uint_as_float(old0_bits)",
        "conv_tail[tail_offset] = old1_bits;",
        "conv_state[state_offset] =",
        "__float_as_uint(",
        'architecture.rfind("gfx942", 0) == 0',
        "at::cuda::OptionalCUDAGuard",
        "at::cuda::getCurrentCUDAStream",
        "C10_CUDA_KERNEL_LAUNCH_CHECK();",
    ):
        assert contract in source

    for forbidden in (
        "PYBIND11_MODULE",
        "torch::empty",
        "torch::zeros",
        ".contiguous(",
        ".to(",
    ):
        assert forbidden not in source

    prepare = source[
        source.index("__global__ void cca_decode_state_prepare_kernel") : source.index(
            "__global__ void cca_decode_state_commit_kernel"
        )
    ]
    assert "conv_state[" in prepare
    assert "recurrent_state[" in prepare
    assert re.search(r"conv_state\[[^]]+\]\s*=", prepare) is None
    assert re.search(r"recurrent_state\[[^]]+\]\s*=", prepare) is None

    commit = source[
        source.index("__global__ void cca_decode_state_commit_kernel") : source.index(
            "void check_matrix("
        )
    ]
    assert "conv_state[state_offset] =" in commit
    assert "conv_state[state_offset + 1] =" in commit
    assert "recurrent_state[" in commit
    assert "const int safe_slot = is_pad ? 0 : slot;" in commit


def test_cca_integration_is_opt_in_pure_decode_and_uses_static_workspaces():
    source = _read("vllm", "model_executor", "layers", "mamba", "cca.py")

    assert 'os.getenv("VLLM_CCA_NATIVE_DECODE_STATE_IO", "0") == "1"' in source
    assert (
        "self._native_decode_state_io and use_native_qk_postprocess" in source
    )
    assert (
        "and has_decode\n            and not has_prefill"
        in source
    )
    assert (
        "native CCA decode state I/O requires native Q/K postprocess" in source
    )
    assert (
        "max_decode_rows = vllm_config.scheduler_config.max_num_seqs" in source
    )
    assert source.count('"_native_decode_conv_window"') == 1
    assert source.count('"_native_decode_conv_tail"') == 1
    assert source.count("persistent=False") >= 2

    prepare = source.index("ops.cca_decode_state_prepare(")
    convolution = source.index(
        "qk_packed3 = self._conv_qk_decode(conv_window)", prepare
    )
    qk_postprocess = source.index("ops.cca_qk_postprocess(", convolution)
    commit = source.index("ops.cca_decode_state_commit(", qk_postprocess)
    assert prepare < convolution < qk_postprocess < commit

    native_start = source.index(
        "if use_native_decode_state_io:", source.index("if has_decode:")
    )
    native_branch = source[
        native_start : source.index("            else:", native_start)
    ]
    assert "torch.empty(" not in native_branch
    assert "torch.cat(" not in native_branch
    assert ".to(" not in native_branch


def test_cca_integration_keeps_prefill_mixed_and_qk_fallbacks_intact():
    source = _read("vllm", "model_executor", "layers", "mamba", "cca.py")

    assert "if has_prefill:" in source
    assert "self.conv_qk_grouped(" in source
    assert "self.conv_qk_depthwise(qk_packed2_cur)" in source
    assert "recurrent_states[state_indices_tensor_p]" in source
    assert "elif use_compiled_qk_postprocess:" in source
    assert "query, key = self._add_grouped_qk_means_inplace(" in source

    decode = source.index("if has_decode:")
    fallback = source[
        source.index("if not use_native_decode_state_io:", decode)
        : source.index("del qk_packed0_d")
    ]
    for operation in (
        "safe_decode_indices = torch.where(",
        "qk_packed0_cached = conv_states[",
        "qk_packed0_cat = torch.cat(",
        "new_qk_packed0_cache = qk_packed0_cached.roll(",
        "value_delayed_decode = recurrent_states[",
        "recurrent_states[safe_decode_indices] =",
    ):
        assert operation in fallback


def test_cpu_contract_preserves_snapshot_pad_and_fp32_tail_semantics():
    rows = 4
    slots = 5
    state_idx = np.array([2, PAD_SLOT_ID, 2, 3], dtype=np.int32)
    conv_state = np.arange(slots * QK_WIDTH * 2, dtype=np.float32).reshape(
        slots, QK_WIDTH, 2
    ) / np.float32(257.0)
    recurrent_state = np.arange(slots * VALUE_WIDTH, dtype=np.float32).reshape(
        slots, VALUE_WIDTH
    ) / np.float32(31.0)
    low_mantissa = np.nextafter(np.float32(1.0), np.float32(2.0))
    conv_state[2, 0, 1] = low_mantissa
    initial_slot2 = conv_state[2].copy()

    qk0_fp32 = np.arange(rows * QK_WIDTH, dtype=np.float32).reshape(
        rows, QK_WIDTH
    ) / np.float32(101.0)
    v_current_fp32 = np.arange(rows * VALUE_WIDTH, dtype=np.float32).reshape(
        rows, VALUE_WIDTH
    ) / np.float32(43.0)
    v_delayed_fp32 = v_current_fp32 + np.float32(7.0)
    qk0 = _fp32_to_bf16_bits(qk0_fp32)
    v_current = _fp32_to_bf16_bits(v_current_fp32)
    v_delayed = _fp32_to_bf16_bits(v_delayed_fp32)

    # Repeated rows use identical current updates; their final state writer is
    # therefore deterministic even though conflicting duplicates remain undefined.
    qk0[2] = qk0[0]
    v_delayed[2] = v_delayed[0]
    qkv_out = np.full((rows, OUTPUT_WIDTH), 0x3F80, dtype=np.uint16)
    conv_window, conv_tail = _prepare_reference(
        qk0, v_current, conv_state, recurrent_state, state_idx, qkv_out
    )

    np.testing.assert_array_equal(conv_window[0], conv_window[2])
    np.testing.assert_array_equal(conv_tail[0], conv_tail[2])
    np.testing.assert_array_equal(conv_window[1], 0)
    np.testing.assert_array_equal(conv_tail[1], 0)
    np.testing.assert_array_equal(qkv_out[1, QK_WIDTH:], 0)
    assert conv_tail[0, 0].view(np.uint32) == low_mantissa.view(np.uint32)
    assert _bf16_bits_to_fp32(conv_window[0, 0, 1]).item() != low_mantissa

    qk_columns_before_commit = qkv_out[:, :QK_WIDTH].copy()
    _commit_reference(
        qk0,
        v_delayed,
        state_idx,
        conv_tail,
        conv_state,
        recurrent_state,
        qkv_out,
    )

    np.testing.assert_array_equal(conv_state[0], 0)
    np.testing.assert_array_equal(recurrent_state[0], 0)
    np.testing.assert_array_equal(qkv_out[1], 0)
    np.testing.assert_array_equal(
        qkv_out[[0, 2, 3], :QK_WIDTH], qk_columns_before_commit[[0, 2, 3]]
    )
    assert conv_state[2, 0, 0].view(np.uint32) == low_mantissa.view(np.uint32)
    np.testing.assert_array_equal(conv_state[2, :, 1], _bf16_bits_to_fp32(qk0[0]))
    np.testing.assert_array_equal(recurrent_state[2], _bf16_bits_to_fp32(v_delayed[0]))
    np.testing.assert_array_equal(
        conv_window[0, :, 0], _fp32_to_bf16_bits(initial_slot2[:, 0])
    )
