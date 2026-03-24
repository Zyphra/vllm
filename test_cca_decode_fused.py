"""
Unit tests for the CCA decode fused kernel.

Tests the "ultimate" fusion that combines in a single kernel:
  Phase 1 — Depthwise causal conv1d update (K=2)
  Phase 2 — Grouped conv1d (W=2) GEMV
  Phase 3 — qk_mean addition + L2 normalisation + sqrt(head_dim) scaling + temperature
  Phase 4 — Direct output write

Compares:
  1. Stepwise PyTorch reference (gold standard)
  2. C++ HIP fused kernel (JIT-compiled)
"""

import sys
import os
import time
import math
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "CCA_Decode"))


# ── Triton-based stepwise reference (matches vllm unfused decode path) ──

from vllm.model_executor.layers.mamba.ops import (
    run_causal_conv1d_update, grouped_conv1d_decode,
)


def ref_triton_cca_decode_fused(
    new_token,         # [B, E, 1]
    dw_weight,         # [E, 2]
    dw_bias,           # [E] or None
    conv_state,        # [num_cache, E, state_len]  — will be mutated
    state_indices,     # [B] int64
    gw_weight,         # [G, D, D, 2]
    gw_bias,           # [G, D] or None
    qk_mean,           # [B, G, D]
    temp,              # [num_k_heads]
    num_q_heads,       # int
    sqrt_head_dim,     # float
    clamp_temp,        # bool
    pad_slot_id=-1,    # int64 — padding slot to skip (unused by triton ref, handled internally)
):
    """Reference using the original Triton kernels (unfused decode path).

    Phase 1: run_causal_conv1d_update  (Triton depthwise conv1d)
    Phase 2: grouped_conv1d_decode     (Triton grouped conv GEMV)
    Phase 3: qk_mean + L2 norm + scale + temperature  (Python/PyTorch)
    """
    B, E, _ = new_token.shape
    G = gw_weight.size(0)
    D = gw_weight.size(1)
    assert E == G * D

    # ── Phase 1: Triton depthwise causal conv1d update ──
    first_input = new_token.contiguous()               # [B, E, 1]
    phase1_out = run_causal_conv1d_update(
        first_input,
        conv_state,
        dw_weight.contiguous(),
        dw_bias,
        state_indices.contiguous(),
        seqlen=1,
    )                                                   # [B, E, seqlen+1]

    # ── Phase 2: Triton grouped conv1d ──
    b, d, w = phase1_out.shape
    second_conv_input = phase1_out.reshape(b, G, d // G, w).contiguous()
    second_weights = gw_weight.contiguous()             # [G, D, D, 2]
    second_bias = gw_bias.reshape(G, -1).contiguous() if gw_bias is not None else None
    phase2_out = grouped_conv1d_decode(
        second_conv_input,
        second_weights,
        second_bias,
    )                                                   # [B, G, D]

    # ── Phase 3: qk_mean + L2 norm + scale + temperature (Python) ──
    acc = phase2_out.float() + qk_mean.float()

    norm = acc.norm(p=2, dim=-1, keepdim=True)
    acc = acc * (sqrt_head_dim / (norm + 1e-12))

    num_k_heads = G - num_q_heads
    if num_k_heads > 0:
        t = temp.float()
        if clamp_temp:
            t = torch.exp(torch.clamp(t, 1e-7, 2.0))
        acc[:, num_q_heads:, :] *= t[None, :, None]

    return acc.reshape(B, G * D)


# ── Pure PyTorch stepwise reference (for cross-validation) ─────────

def ref_pytorch_cca_decode_fused(
    new_token,         # [B, E, 1]
    dw_weight,         # [E, 2]
    dw_bias,           # [E] or None
    conv_state,        # [num_cache, E, state_len]  — will be mutated
    state_indices,     # [B] int64
    gw_weight,         # [G, D, D, 2]
    gw_bias,           # [G, D] or None
    qk_mean,           # [B, G, D]
    temp,              # [num_k_heads]
    num_q_heads,       # int
    sqrt_head_dim,     # float
    clamp_temp,        # bool
    pad_slot_id=-1,    # int64 — padding slot to skip
):
    """Step-by-step pure PyTorch reference (fp32 accumulation)."""
    B, E, _ = new_token.shape
    G = gw_weight.size(0)
    D = gw_weight.size(1)
    assert E == G * D
    num_cache = conv_state.size(0)

    nt = new_token.float().squeeze(-1)        # [B, E]
    dww = dw_weight.float()                   # [E, 2]
    dwb = dw_bias.float() if dw_bias is not None else None
    cs = conv_state.float()                   # work in fp32

    # ── Phase 1: depthwise causal conv1d update ──
    phase1 = torch.zeros(B, E, 2, device=new_token.device, dtype=torch.float32)
    for b in range(B):
        idx = state_indices[b].item()
        valid = (idx >= 0) and (idx < num_cache) and (idx != pad_slot_id)
        if not valid:
            continue

        s0 = cs[idx, :, 0]                   # [E]
        s1 = cs[idx, :, 1]                   # [E]
        xt = nt[b]                            # [E]

        w0 = dww[:, 0]
        w1 = dww[:, 1]
        bias = dwb if dwb is not None else torch.zeros(E, device=new_token.device)

        y0 = bias + w0 * s0 + w1 * s1
        y1 = bias + w0 * s1 + w1 * xt

        phase1[b, :, 0] = y0
        phase1[b, :, 1] = y1

        conv_state[idx, :, 0] = s1.to(conv_state.dtype)
        conv_state[idx, :, 1] = xt.to(conv_state.dtype)

    # Reshape for grouped conv: [B, G, D, 2]
    phase1_grouped = phase1.reshape(B, G, D, 2)

    # ── Phase 2: grouped conv GEMV ──
    gww = gw_weight.float()
    phase2 = torch.einsum("goiw,bgiw->bgo", gww, phase1_grouped)
    if gw_bias is not None:
        phase2 = phase2 + gw_bias.float().unsqueeze(0)

    # ── Phase 3: qk_mean + L2 norm + scale + temperature ──
    acc = phase2 + qk_mean.float()

    norm = acc.norm(p=2, dim=-1, keepdim=True)
    acc = acc * (sqrt_head_dim / (norm + 1e-12))

    num_k_heads = G - num_q_heads
    if num_k_heads > 0:
        t = temp.float()
        if clamp_temp:
            t = torch.exp(torch.clamp(t, 1e-7, 2.0))
        acc[:, num_q_heads:, :] *= t[None, :, None]

    return acc.reshape(B, G * D)


# ── JIT-compile C++ fused kernel ───────────────────────────────────

def load_fused_kernel():
    try:
        from torch.utils.cpp_extension import load_inline

        kernel_src_path = os.path.join(
            os.path.dirname(__file__),
            "csrc", "rocm", "cca_decode_fused.cu",
        )
        if not os.path.exists(kernel_src_path):
            print(f"[WARN] Fused kernel source not found: {kernel_src_path}")
            return None

        with open(kernel_src_path, "r") as f:
            cuda_src = f.read()

        cpp_src = r"""
#include <torch/extension.h>

torch::Tensor cca_decode_fused(
    const torch::Tensor& new_token,
    const torch::Tensor& dw_weight,
    const std::optional<torch::Tensor>& dw_bias,
    torch::Tensor& conv_state,
    const torch::Tensor& state_indices,
    const torch::Tensor& gw_weight,
    const std::optional<torch::Tensor>& gw_bias,
    const torch::Tensor& qk_mean,
    const torch::Tensor& temp,
    int64_t num_q_heads,
    double sqrt_head_dim,
    bool clamp_temp,
    int64_t pad_slot_id);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cca_decode_fused", &cca_decode_fused,
          "CCA decode fused kernel v2 (HIP/CUDA)");
}
"""

        extra_cflags = ["-O3"]
        extra_cuda_cflags = ["-O3"]

        if torch.version.hip is not None:
            extra_cflags.append("-DUSE_ROCM")
            extra_cuda_cflags.append("-DUSE_ROCM")

        print("[INFO] JIT-compiling fused kernel (may take a minute) …")
        t0 = time.time()
        module = load_inline(
            name="cca_decode_fused_ext",
            cpp_sources=[cpp_src],
            cuda_sources=[cuda_src],
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            verbose=False,
        )
        print(f"[INFO] Fused kernel compiled in {time.time() - t0:.1f}s")
        return module.cca_decode_fused
    except Exception as e:
        print(f"[WARN] Cannot JIT-compile fused kernel: {e}")
        import traceback
        traceback.print_exc()
        return None


def transpose_gw_weight(gw_weight):
    """Transpose gw_weight from [G, D, D, 2] to [G, D*2, D] for coalesced access."""
    G, D_out, D_in, K = gw_weight.shape
    return gw_weight.permute(0, 2, 3, 1).contiguous().view(G, D_in * K, D_out)


# ── Test helpers ────────────────────────────────────────────────────

def allclose(a, b, name_a, name_b, atol=1e-2, rtol=1e-2):
    a_f = a.float()
    b_f = b.float()
    diff = (a_f - b_f).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rel_err = max_diff / (b_f.abs().max().item() + 1e-12)
    ok = torch.allclose(a_f, b_f, atol=atol, rtol=rtol)
    status = "PASS ✓" if ok else "FAIL ✗"
    print(f"  [{status}] {name_a} vs {name_b}:  "
          f"max_abs={max_diff:.6e}  mean_abs={mean_diff:.6e}  rel={rel_err:.6e}")
    if not ok:
        idx = diff.argmax()
        flat_a = a_f.reshape(-1)
        flat_b = b_f.reshape(-1)
        print(f"         worst element @{idx.item()}: "
              f"got={flat_a[idx].item():.6f}  expected={flat_b[idx].item():.6f}")
    return ok


def run_single_test(
    B, G, D, num_q_heads, dtype, has_dw_bias, has_gw_bias, clamp_temp,
    fused_fn, device, state_len=2, pad_slot_id=-1
):
    E = G * D
    num_k_heads = G - num_q_heads
    head_dim = D
    sqrt_hd = math.sqrt(head_dim)

    tag = (f"B={B} G={G} D={D} qh={num_q_heads} kh={num_k_heads} "
           f"dtype={dtype} dw_bias={has_dw_bias} gw_bias={has_gw_bias} "
           f"clamp={clamp_temp} state_len={state_len} pad_slot={pad_slot_id}")
    print(f"\n── Test: {tag} ──")

    torch.manual_seed(42)
    num_cache = max(B, 4)

    new_token    = torch.randn(B, E, 1,    device=device, dtype=dtype) * 0.1
    dw_weight    = torch.randn(E, 2,       device=device, dtype=dtype) * 0.5
    dw_bias      = torch.randn(E,          device=device, dtype=dtype) * 0.1 if has_dw_bias else None
    conv_state   = torch.randn(num_cache, E, state_len, device=device, dtype=dtype) * 0.1
    state_indices = torch.arange(B, device=device, dtype=torch.int64) % num_cache
    gw_weight    = torch.randn(G, D, D, 2, device=device, dtype=dtype) * 0.02
    gw_bias      = torch.randn(G, D,       device=device, dtype=dtype) * 0.1 if has_gw_bias else None
    qk_mean      = torch.randn(B, G, D,    device=device, dtype=dtype) * 0.1
    temp_vec     = torch.randn(num_k_heads, device=device, dtype=torch.float32).abs() + 0.5

    # Inject pad_slot_id into some state_indices entries when testing
    if pad_slot_id >= 0 and B > 1:
        state_indices[0] = pad_slot_id

    all_pass = True

    # clamp_temp + bf16 amplifies rounding through exp(); use wider tolerance
    out_atol = 0.1 if clamp_temp else 1e-2
    out_rtol = 0.1 if clamp_temp else 1e-2

    # run_causal_conv1d_update hard-codes pad_slot_id=-1 internally,
    # so Triton ref is only valid when pad_slot_id == -1.
    triton_ref_valid = (pad_slot_id == -1)

    # ── Pure PyTorch reference (always valid, gold standard) ──
    cs_pytorch = conv_state.clone()
    y_pytorch = ref_pytorch_cca_decode_fused(
        new_token, dw_weight, dw_bias, cs_pytorch, state_indices,
        gw_weight, gw_bias, qk_mean, temp_vec,
        num_q_heads, sqrt_hd, clamp_temp, pad_slot_id,
    )

    # ── Triton reference (only when pad_slot_id == -1) ──
    if triton_ref_valid:
        cs_triton = conv_state.clone()
        y_triton = ref_triton_cca_decode_fused(
            new_token, dw_weight, dw_bias, cs_triton, state_indices,
            gw_weight, gw_bias, qk_mean, temp_vec,
            num_q_heads, sqrt_hd, clamp_temp, pad_slot_id,
        )
        all_pass &= allclose(y_triton, y_pytorch, "ref_triton", "ref_pytorch",
                             atol=out_atol, rtol=out_rtol)
    else:
        print("  [SKIP] Triton ref (pad_slot_id != -1, hardcoded in triton kernel)")

    # Pick the best available ref for comparing against fused kernel
    y_ref = y_triton if triton_ref_valid else y_pytorch
    cs_ref = cs_triton if triton_ref_valid else cs_pytorch
    ref_name = "ref_triton" if triton_ref_valid else "ref_pytorch"

    # ── Fused C++ kernel vs reference ──
    if fused_fn is not None:
        cs_fused = conv_state.clone()
        gw_weight_T = transpose_gw_weight(gw_weight)
        y_fused = fused_fn(
            new_token.contiguous(), dw_weight.contiguous(),
            dw_bias.contiguous() if dw_bias is not None else None,
            cs_fused,
            state_indices.contiguous(),
            gw_weight_T,
            gw_bias.contiguous() if gw_bias is not None else None,
            qk_mean.contiguous(),
            temp_vec.contiguous(),
            num_q_heads, sqrt_hd, clamp_temp, pad_slot_id,
        )
        all_pass &= allclose(y_fused, y_ref, "fused_cpp", ref_name,
                             atol=out_atol, rtol=out_rtol)
        all_pass &= allclose(cs_fused[:, :, :2].float(), cs_ref[:, :, :2].float(),
                             "conv_state_cpp", f"conv_state_{ref_name.split('_')[1]}",
                             atol=1e-3, rtol=1e-3)
    else:
        print("  [SKIP] Fused C++ kernel not available")

    return all_pass


# ── Performance benchmark ───────────────────────────────────────────

def benchmark_fused(fused_fn, ref_fn, device,
                    B=128, G=24, D=128, num_q_heads=16,
                    warmup=20, iters=200):
    E = G * D
    num_k_heads = G - num_q_heads
    sqrt_hd = math.sqrt(D)
    num_cache = max(B, 4)

    torch.manual_seed(0)
    new_token    = torch.randn(B, E, 1,    device=device, dtype=torch.bfloat16) * 0.1
    dw_weight    = torch.randn(E, 2,       device=device, dtype=torch.bfloat16) * 0.5
    dw_bias      = torch.randn(E,          device=device, dtype=torch.bfloat16) * 0.1
    conv_state   = torch.randn(num_cache, E, 2, device=device, dtype=torch.bfloat16) * 0.1
    state_indices = torch.arange(B, device=device, dtype=torch.int64) % num_cache
    gw_weight    = torch.randn(G, D, D, 2, device=device, dtype=torch.bfloat16) * 0.02
    gw_bias      = torch.randn(G, D,       device=device, dtype=torch.bfloat16) * 0.1
    qk_mean      = torch.randn(B, G, D,    device=device, dtype=torch.bfloat16) * 0.1
    temp_vec     = torch.randn(num_k_heads, device=device, dtype=torch.float32).abs() + 0.5

    # Pre-compute contiguous tensors for triton / fused benchmarks
    nt_c = new_token.contiguous()
    dww_c = dw_weight.contiguous()
    dwb_c = dw_bias.contiguous()
    gww_c = gw_weight.contiguous()
    gww_T = transpose_gw_weight(gww_c)                      # [G, D*2, D]
    gwb_c = gw_bias.contiguous()
    qkm_c = qk_mean.contiguous()
    tv_c  = temp_vec.contiguous()
    si_c  = state_indices.contiguous()
    groups = G
    second_weights = gww_c                                  # [G, D, D, 2]
    second_bias = gwb_c.reshape(groups, -1).contiguous()    # [G, D]
    dw_weights_flat = dww_c                                 # [E, 2]

    def run_triton():
        cs = conv_state.clone()
        phase1 = run_causal_conv1d_update(
            nt_c, cs, dw_weights_flat, dw_bias, si_c, seqlen=1)
        b, d, w = phase1.shape
        second_in = phase1.reshape(b, groups, d // groups, w).contiguous()
        phase2 = grouped_conv1d_decode(second_in, second_weights, second_bias)
        acc = phase2.float() + qkm_c.float()
        norm = acc.norm(p=2, dim=-1, keepdim=True)
        acc = acc * (sqrt_hd / (norm + 1e-12))
        return acc.reshape(b, -1)

    def run_fused():
        cs = conv_state.clone()
        return fused_fn(
            nt_c, dww_c, dwb_c, cs, si_c,
            gww_T, gwb_c, qkm_c, tv_c,
            num_q_heads, sqrt_hd, False, -1,
        )

    def time_fn(fn, name):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
        avg = sum(times) / len(times)
        med = sorted(times)[len(times) // 2]
        mn = min(times)
        print(f"  {name:25s}  avg={avg:.3f}ms  med={med:.3f}ms  min={mn:.3f}ms")
        return avg

    print(f"\n  Benchmark: B={B} G={G} D={D} qh={num_q_heads} bf16")
    time_fn(run_triton, "triton_unfused")
    if fused_fn is not None:
        time_fn(run_fused, "fused_cpp_kernel")
    else:
        print("  [SKIP] Fused C++ kernel not available")


# ── Main ────────────────────────────────────────────────────────────

def main():
    device = "cuda"
    if not torch.cuda.is_available():
        print("CUDA/HIP device not available, exiting.")
        return

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    if torch.version.hip:
        print(f"ROCm/HIP: {torch.version.hip}")
    print()

    fused_fn = load_fused_kernel()

    # ────────────────────────────────────────
    # Correctness tests
    # ────────────────────────────────────────
    print("=" * 70)
    print("  CORRECTNESS TESTS — CCA Decode Fused Kernel")
    print("=" * 70)

    test_configs = [
        # (B, G, D, num_q_heads, dtype, has_dw_bias, has_gw_bias, clamp_temp, state_len, pad_slot_id)
        (1,   24, 128, 16, torch.bfloat16, True,  True,  False, 2, -1),
        (4,   24, 128, 16, torch.bfloat16, True,  True,  False, 2, -1),
        (32,  24, 128, 16, torch.bfloat16, True,  True,  False, 2, -1),
        (128, 24, 128, 16, torch.bfloat16, True,  True,  False, 2, -1),
        # Without biases
        (4,   24, 128, 16, torch.bfloat16, False, False, False, 2, -1),
        (4,   24, 128, 16, torch.bfloat16, True,  False, False, 2, -1),
        (4,   24, 128, 16, torch.bfloat16, False, True,  False, 2, -1),
        # With temperature clamping
        (4,   24, 128, 16, torch.bfloat16, True,  True,  True,  2, -1),
        (32,  24, 128, 16, torch.bfloat16, True,  True,  True,  2, -1),
        # Different group/dim configs
        (1,   16, 64,  8,  torch.bfloat16, True,  True,  False, 2, -1),
        (4,   8,  128, 4,  torch.bfloat16, True,  True,  False, 2, -1),
        (4,   32, 64,  24, torch.bfloat16, True,  True,  True,  2, -1),
        # fp16
        (4,   24, 128, 16, torch.float16,  True,  True,  False, 2, -1),
        (4,   24, 128, 16, torch.float16,  True,  True,  True,  2, -1),
        # fp32
        (4,   24, 128, 16, torch.float32,  True,  True,  False, 2, -1),
        # state_len > 2: verify correct stride handling
        (4,   24, 128, 16, torch.bfloat16, True,  True,  False, 4, -1),
        (4,   24, 128, 16, torch.bfloat16, True,  True,  True,  4, -1),
        (1,   24, 128, 16, torch.bfloat16, True,  True,  False, 8, -1),
        (32,  24, 128, 16, torch.float16,  True,  True,  False, 3, -1),
        # Production shape: B=512 G=10 D=128 qh=8 kh=2
        (512, 10, 128, 8,  torch.bfloat16, True,  True,  False, 2, -1),
        (512, 10, 128, 8,  torch.bfloat16, True,  True,  True,  2, -1),
        # pad_slot_id tests: negative-one default
        (4,   24, 128, 16, torch.bfloat16, True,  True,  False, 2, -1),
        # pad_slot_id = 0 (first cache slot is padding)
        (4,   24, 128, 16, torch.bfloat16, True,  True,  False, 2,  0),
        # pad_slot_id = valid non-negative index
        (8,   24, 128, 16, torch.bfloat16, True,  True,  False, 2,  2),
        (8,   10, 128, 8,  torch.bfloat16, True,  True,  True,  2,  3),
        # pad_slot_id with production shape
        (512, 10, 128, 8,  torch.bfloat16, True,  True,  False, 2,  0),
    ]

    total_pass = 0
    total_tests = 0
    for cfg in test_configs:
        total_tests += 1
        B, G, D, qh, dt, dwb, gwb, ct, sl, psid = cfg
        if run_single_test(B, G, D, qh, dt, dwb, gwb, ct,
                           fused_fn, device, state_len=sl,
                           pad_slot_id=psid):
            total_pass += 1

    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {total_pass}/{total_tests} test groups passed")
    print(f"{'=' * 70}")

    # ────────────────────────────────────────
    # Performance benchmark
    # ────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  PERFORMANCE BENCHMARK")
    print(f"{'=' * 70}")

    # # Large batch
    # benchmark_fused(fused_fn, ref_cca_decode_fused, device,
    #                 B=128, G=24, D=128, num_q_heads=16)
    # # Small batch (typical decode)
    # benchmark_fused(fused_fn, ref_cca_decode_fused, device,
    #                 B=1, G=24, D=128, num_q_heads=16)
    # benchmark_fused(fused_fn, ref_cca_decode_fused, device,
    #                 B=4, G=24, D=128, num_q_heads=16)

    print("\n  --- Production shape: G=10 D=128 ---")
    for bs in [128, 64, 32, 16]:
        benchmark_fused(fused_fn, ref_triton_cca_decode_fused, device,
                        B=bs, G=10, D=128, num_q_heads=8)

    print("\n  --- Large shape: G=24 D=128 ---")
    for bs in [128, 64, 32, 16]:
        benchmark_fused(fused_fn, ref_triton_cca_decode_fused, device,
                        B=bs, G=24, D=128, num_q_heads=16)

    print("\nDone.")


if __name__ == "__main__":
    main()
