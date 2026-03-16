"""Correctness test & profiling for cca_prefill_fused vs. the original Python loop."""
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["VLLM_CCA_PREFILL_FUSED_KERNEL_PATH"] = "csrc/rocm/cca_prefill_fused.cu"
os.environ["VLLM_CCA_FUSED_ENABLED"] = "1"

from vllm.model_executor.layers.mamba.ops.cca import cca_prefill_fused
from vllm.model_executor.layers.mamba.ops.cca import (
    cca_prefill_fused_hip_available, cca_prefill_fused_hip,
)

def reference_prefill_loop(
    hs_p,                # [T, 1, H]
    qk_packed0_p,        # [T, 1, E]
    prev_hs,             # [num_cache, H]
    conv_states,         # [num_cache, E, state_len]
    query_start_loc_p,   # [num_prefills + 1]
    has_initial_states_p,# [num_prefills]
    state_indices_p,     # [num_prefills]
    conv_qk,             # nn.Sequential(Conv1d, Conv1d)
    total_padding,
    cca_time0,
):
    """Original Python for-loop implementation (reference)."""
    T = hs_p.shape[0]
    H = hs_p.shape[2]
    E = qk_packed0_p.shape[2]

    hs2 = torch.zeros((T, 1, H), device=hs_p.device, dtype=hs_p.dtype)
    qk_packed3_p = torch.zeros((T, 1, E), device=hs_p.device, dtype=hs_p.dtype)

    for i in range(len(query_start_loc_p) - 1):
        start_i = query_start_loc_p[i].item()
        end_i = query_start_loc_p[i + 1].item()
        hs2_cur = hs_p[start_i:end_i, :, :]
        qk_packed0_cur = qk_packed0_p[start_i:end_i, :, :]
        qk_packed1_cur = qk_packed0_cur.permute(1, 2, 0)

        if has_initial_states_p[i]:
            hs2_cached = prev_hs[state_indices_p[i]].unsqueeze(0).unsqueeze(0)
            hs2_cur = torch.cat([hs2_cached, hs2_cur[:-1]], dim=0)
            qk_packed0_cached = conv_states[state_indices_p[i]].unsqueeze(0)
            qk_packed2_cur = torch.cat([qk_packed0_cached, qk_packed1_cur],
                                       dim=-1)
        else:
            hs2_cur = F.pad(hs2_cur[:-1], pad=(0, 0, 0, 0, 1, 0))
            qk_packed2_cur = F.pad(qk_packed1_cur, (total_padding, 0))

        hs2[start_i:end_i] = hs2_cur

        conv_states_cur = F.pad(
            qk_packed2_cur,
            (cca_time0 - qk_packed2_cur.shape[-1], 0),
        )
        conv_states[state_indices_p[i]] = conv_states_cur.to(
            conv_states.device)

        qk_packed3_cur = conv_qk(qk_packed2_cur).permute(2, 0, 1)
        qk_packed3_p[start_i:end_i] = qk_packed3_cur

    prev_hs[state_indices_p] = hs_p[
        query_start_loc_p[1:] - 1, 0, :
    ].to(prev_hs.device)

    return hs2, qk_packed3_p


def make_test_data(
    num_prefills, seq_lens, H, E, G, D, num_cache, device, dtype,
    has_initial_all=True,
):
    T = sum(seq_lens)
    state_len = 2

    hs_p = torch.randn(T, 1, H, device=device, dtype=dtype)
    qk_packed0_p = torch.randn(T, 1, E, device=device, dtype=dtype)
    prev_hs = torch.randn(num_cache, H, device=device, dtype=dtype)
    conv_states = torch.randn(num_cache, E, state_len, device=device,
                              dtype=dtype)

    query_start_loc_p = torch.zeros(num_prefills + 1, device=device,
                                    dtype=torch.int64)
    for i, sl in enumerate(seq_lens):
        query_start_loc_p[i + 1] = query_start_loc_p[i] + sl

    if has_initial_all:
        has_initial_states_p = torch.ones(num_prefills, device=device,
                                          dtype=torch.bool)
    else:
        has_initial_states_p = torch.zeros(num_prefills, device=device,
                                           dtype=torch.bool)

    perm = torch.randperm(num_cache, device=device)[:num_prefills]
    state_indices_p = perm.to(torch.int64)

    cca_time0 = 2
    cca_time1 = 2
    total_padding = (cca_time0 - 1) + (cca_time1 - 1)

    conv_qk = nn.Sequential(
        nn.Conv1d(E, E, kernel_size=cca_time0, groups=E, padding=0, stride=1),
        nn.Conv1d(E, E, kernel_size=cca_time1, groups=G, padding=0, stride=1),
    ).to(device=device, dtype=dtype)

    dim = E
    kernel_width = cca_time0
    dw_weight = conv_qk[0].weight.reshape(dim, kernel_width).contiguous()
    dw_bias = conv_qk[0].bias
    gw_weight = conv_qk[1].weight.reshape(G, D, -1, cca_time1).contiguous()
    gw_bias = conv_qk[1].bias.reshape(G, -1).contiguous()

    return dict(
        hs_p=hs_p,
        qk_packed0_p=qk_packed0_p,
        prev_hs=prev_hs,
        conv_states=conv_states,
        query_start_loc_p=query_start_loc_p,
        has_initial_states_p=has_initial_states_p,
        state_indices_p=state_indices_p,
        conv_qk=conv_qk,
        dw_weight=dw_weight,
        dw_bias=dw_bias,
        gw_weight=gw_weight,
        gw_bias=gw_bias,
        total_padding=total_padding,
        cca_time0=cca_time0,
    )


def run_test(
    label, num_prefills, seq_lens, H, E, G, D, num_cache, device, dtype,
    has_initial_all=True, atol=1e-2, rtol=1e-2,
):
    data = make_test_data(num_prefills, seq_lens, H, E, G, D, num_cache,
                          device, dtype, has_initial_all)

    prev_hs_ref = data["prev_hs"].clone()
    conv_states_ref = data["conv_states"].clone()
    prev_hs_fused = data["prev_hs"].clone()
    conv_states_fused = data["conv_states"].clone()

    hs2_ref, qk3_ref = reference_prefill_loop(
        data["hs_p"], data["qk_packed0_p"],
        prev_hs_ref, conv_states_ref,
        data["query_start_loc_p"], data["has_initial_states_p"],
        data["state_indices_p"],
        data["conv_qk"], data["total_padding"], data["cca_time0"],
    )

    hs2_fused, qk3_fused = cca_prefill_fused(
        data["hs_p"], data["qk_packed0_p"],
        prev_hs_fused, conv_states_fused,
        data["query_start_loc_p"], data["has_initial_states_p"],
        data["state_indices_p"],
        data["dw_weight"], data["dw_bias"],
        data["gw_weight"], data["gw_bias"],
    )

    ok = True
    for name, ref, fused in [
        ("hs2", hs2_ref, hs2_fused),
        ("qk_packed3_p", qk3_ref, qk3_fused),
        ("prev_hs", prev_hs_ref, prev_hs_fused),
        ("conv_states", conv_states_ref, conv_states_fused),
    ]:
        if not torch.allclose(ref.float(), fused.float(), atol=atol, rtol=rtol):
            max_diff = (ref.float() - fused.float()).abs().max().item()
            print(f"  FAIL {label}/{name}: max_diff={max_diff:.6f}")
            ok = False
        else:
            max_diff = (ref.float() - fused.float()).abs().max().item()
            print(f"  PASS {label}/{name}: max_diff={max_diff:.6f}")

    return ok


def main():
    device = "cuda"
    dtype = torch.bfloat16
    G = 10
    D = 128
    E = G * D
    H = 2048
    num_cache = 64

    all_pass = True

    print("=== Test 1: single request, with initial state ===")
    all_pass &= run_test("single_init", 1, [32], H, E, G, D, num_cache,
                         device, dtype, has_initial_all=True)

    print("\n=== Test 2: single request, no initial state ===")
    all_pass &= run_test("single_noinit", 1, [32], H, E, G, D, num_cache,
                         device, dtype, has_initial_all=False)

    print("\n=== Test 3: multiple requests, with initial state ===")
    all_pass &= run_test("multi_init", 4, [16, 64, 8, 128], H, E, G, D,
                         num_cache, device, dtype, has_initial_all=True)

    print("\n=== Test 4: multiple requests, no initial state ===")
    all_pass &= run_test("multi_noinit", 4, [16, 64, 8, 128], H, E, G, D,
                         num_cache, device, dtype, has_initial_all=False)

    print("\n=== Test 5: single token per request ===")
    all_pass &= run_test("single_tok", 3, [1, 1, 1], H, E, G, D,
                         num_cache, device, dtype, has_initial_all=True)

    print("\n=== Test 6: mixed initial states ===")
    data = make_test_data(4, [16, 64, 8, 128], H, E, G, D, num_cache,
                          device, dtype, has_initial_all=True)
    data["has_initial_states_p"][0] = False
    data["has_initial_states_p"][2] = False

    prev_hs_ref = data["prev_hs"].clone()
    conv_states_ref = data["conv_states"].clone()
    prev_hs_fused = data["prev_hs"].clone()
    conv_states_fused = data["conv_states"].clone()

    hs2_ref, qk3_ref = reference_prefill_loop(
        data["hs_p"], data["qk_packed0_p"],
        prev_hs_ref, conv_states_ref,
        data["query_start_loc_p"], data["has_initial_states_p"],
        data["state_indices_p"],
        data["conv_qk"], data["total_padding"], data["cca_time0"],
    )
    hs2_fused, qk3_fused = cca_prefill_fused(
        data["hs_p"], data["qk_packed0_p"],
        prev_hs_fused, conv_states_fused,
        data["query_start_loc_p"], data["has_initial_states_p"],
        data["state_indices_p"],
        data["dw_weight"], data["dw_bias"],
        data["gw_weight"], data["gw_bias"],
    )
    atol, rtol = 1e-2, 1e-2
    mixed_ok = True
    for name, ref, fused in [
        ("hs2", hs2_ref, hs2_fused),
        ("qk_packed3_p", qk3_ref, qk3_fused),
        ("prev_hs", prev_hs_ref, prev_hs_fused),
        ("conv_states", conv_states_ref, conv_states_fused),
    ]:
        if not torch.allclose(ref.float(), fused.float(), atol=atol, rtol=rtol):
            max_diff = (ref.float() - fused.float()).abs().max().item()
            print(f"  FAIL mixed/{name}: max_diff={max_diff:.6f}")
            mixed_ok = False
        else:
            max_diff = (ref.float() - fused.float()).abs().max().item()
            print(f"  PASS mixed/{name}: max_diff={max_diff:.6f}")
    all_pass &= mixed_ok

    if all_pass:
        print("\n*** ALL TRITON TESTS PASSED ***")
    else:
        print("\n*** SOME TRITON TESTS FAILED ***")

    # ------------------------------------------------------------------
    # HIP C++ version correctness
    # ------------------------------------------------------------------
    if cca_prefill_fused_hip_available():
        print("\n" + "=" * 60)
        print("  HIP C++ Prefill Fused — Correctness Tests")
        print("=" * 60)
        hip_pass = True

        def run_hip_test(label, data, atol=1e-2, rtol=1e-2):
            prev_hs_ref = data["prev_hs"].clone()
            conv_states_ref = data["conv_states"].clone()
            prev_hs_hip = data["prev_hs"].clone()
            conv_states_hip = data["conv_states"].clone()

            hs2_ref, qk3_ref = reference_prefill_loop(
                data["hs_p"], data["qk_packed0_p"],
                prev_hs_ref, conv_states_ref,
                data["query_start_loc_p"], data["has_initial_states_p"],
                data["state_indices_p"],
                data["conv_qk"], data["total_padding"], data["cca_time0"],
            )
            hs2_hip, qk3_hip = cca_prefill_fused_hip(
                data["hs_p"], data["qk_packed0_p"],
                prev_hs_hip, conv_states_hip,
                data["query_start_loc_p"], data["has_initial_states_p"],
                data["state_indices_p"],
                data["dw_weight"], data["dw_bias"],
                data["gw_weight"], data["gw_bias"],
            )
            ok = True
            for name, ref, fused in [
                ("hs2", hs2_ref, hs2_hip),
                ("qk_packed3_p", qk3_ref, qk3_hip),
                ("prev_hs", prev_hs_ref, prev_hs_hip),
                ("conv_states", conv_states_ref, conv_states_hip),
            ]:
                if not torch.allclose(ref.float(), fused.float(),
                                      atol=atol, rtol=rtol):
                    max_diff = (ref.float() - fused.float()).abs().max().item()
                    print(f"  FAIL {label}/{name}: max_diff={max_diff:.6f}")
                    ok = False
                else:
                    max_diff = (ref.float() - fused.float()).abs().max().item()
                    print(f"  PASS {label}/{name}: max_diff={max_diff:.6f}")
            return ok

        print("\n=== HIP Test 1: single request, with init ===")
        hip_pass &= run_hip_test("single_init",
            make_test_data(1, [32], H, E, G, D, num_cache, device, dtype, True))

        print("\n=== HIP Test 2: single request, no init ===")
        hip_pass &= run_hip_test("single_noinit",
            make_test_data(1, [32], H, E, G, D, num_cache, device, dtype, False))

        print("\n=== HIP Test 3: multi request, with init ===")
        hip_pass &= run_hip_test("multi_init",
            make_test_data(4, [16, 64, 8, 128], H, E, G, D, num_cache,
                           device, dtype, True))

        print("\n=== HIP Test 4: single token ===")
        hip_pass &= run_hip_test("single_tok",
            make_test_data(3, [1, 1, 1], H, E, G, D, num_cache,
                           device, dtype, True))

        print("\n=== HIP Test 5: mixed initial states ===")
        mixed_data = make_test_data(4, [16, 64, 8, 128], H, E, G, D,
                                    num_cache, device, dtype, True)
        mixed_data["has_initial_states_p"][0] = False
        mixed_data["has_initial_states_p"][2] = False
        hip_pass &= run_hip_test("mixed", mixed_data)

        if hip_pass:
            print("\n*** ALL HIP C++ TESTS PASSED ***")
        else:
            print("\n*** SOME HIP C++ TESTS FAILED ***")
        all_pass &= hip_pass
    else:
        print("\n[SKIP] HIP C++ prefill fused kernel not available.")


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------

def bench_one(label, fn, warmup=10, repeats=100):
    """GPU-timed benchmark. Returns median ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]

    for i in range(repeats):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times.sort()
    median = times[repeats // 2]
    p10 = times[max(0, repeats // 10)]
    p90 = times[min(repeats - 1, repeats * 9 // 10)]
    print(f"  {label:30s}  median={median:.3f} ms  p10={p10:.3f} ms  p90={p90:.3f} ms")
    return median


def profile_scenario(
    tag, num_prefills, seq_lens, H, E, G, D, num_cache, device, dtype,
    warmup=10, repeats=100,
):
    T = sum(seq_lens)
    data = make_test_data(num_prefills, seq_lens, H, E, G, D, num_cache,
                          device, dtype, has_initial_all=True)
    hip_ok = cca_prefill_fused_hip_available()

    def run_ref():
        ph = data["prev_hs"].clone()
        cs = data["conv_states"].clone()
        return reference_prefill_loop(
            data["hs_p"], data["qk_packed0_p"], ph, cs,
            data["query_start_loc_p"], data["has_initial_states_p"],
            data["state_indices_p"],
            data["conv_qk"], data["total_padding"], data["cca_time0"],
        )

    def run_fused():
        ph = data["prev_hs"].clone()
        cs = data["conv_states"].clone()
        return cca_prefill_fused(
            data["hs_p"], data["qk_packed0_p"], ph, cs,
            data["query_start_loc_p"], data["has_initial_states_p"],
            data["state_indices_p"],
            data["dw_weight"], data["dw_bias"],
            data["gw_weight"], data["gw_bias"],
        )

    def run_hip():
        ph = data["prev_hs"].clone()
        cs = data["conv_states"].clone()
        return cca_prefill_fused_hip(
            data["hs_p"], data["qk_packed0_p"], ph, cs,
            data["query_start_loc_p"], data["has_initial_states_p"],
            data["state_indices_p"],
            data["dw_weight"], data["dw_bias"],
            data["gw_weight"], data["gw_bias"],
        )

    print(f"\n--- {tag}  (N={num_prefills}, T={T}, seqlens={seq_lens}) ---")
    t_ref = bench_one("reference (Python loop)", run_ref, warmup, repeats)
    t_tri = bench_one("fused (Triton 3-kernel)", run_fused, warmup, repeats)
    t_hip = None
    if hip_ok:
        t_hip = bench_one("fused (HIP C++ 3-kernel)", run_hip, warmup, repeats)
    tri_speedup = t_ref / t_tri if t_tri > 0 else float("inf")
    print(f"  {'Triton speedup':30s}  {tri_speedup:.2f}x")
    if hip_ok and t_hip:
        hip_speedup = t_ref / t_hip if t_hip > 0 else float("inf")
        print(f"  {'HIP C++ speedup':30s}  {hip_speedup:.2f}x")
    return t_ref, t_tri, t_hip


def profile_main():
    device = "cuda"
    dtype = torch.bfloat16
    G = 10
    D = 128
    E = G * D
    H = 2048
    num_cache = 512

    print("=" * 70)
    print("  CCA Prefill Fused — Performance Profile")
    print("=" * 70)

    profile_scenario("1-req short", 1, [32], H, E, G, D, num_cache,
                     device, dtype)
    profile_scenario("1-req medium", 1, [256], H, E, G, D, num_cache,
                     device, dtype)
    profile_scenario("1-req long", 1, [1024], H, E, G, D, num_cache,
                     device, dtype)
    profile_scenario("4-req mixed", 4, [16, 64, 128, 256], H, E, G, D,
                     num_cache, device, dtype)
    profile_scenario("8-req uniform", 8, [64] * 8, H, E, G, D, num_cache,
                     device, dtype)
    profile_scenario("16-req uniform", 16, [32] * 16, H, E, G, D, num_cache,
                     device, dtype)
    profile_scenario("32-req uniform", 32, [16] * 32, H, E, G, D, num_cache,
                     device, dtype)
    profile_scenario("64-req uniform", 64, [8] * 64, H, E, G, D, num_cache,
                     device, dtype)
    profile_scenario("4-req long", 4, [512, 1024, 256, 2048], H, E, G, D,
                     num_cache, device, dtype)

    print("\n" + "=" * 70)
    print("  Per-kernel breakdown (4-req mixed)")
    print("=" * 70)
    breakdown_scenario(4, [16, 64, 128, 256], H, E, G, D, num_cache,
                       device, dtype)


def breakdown_scenario(
    num_prefills, seq_lens, H, E, G, D, num_cache, device, dtype,
    warmup=10, repeats=100,
):
    """Profile each of the 3 fused kernels individually."""
    import triton
    from vllm.model_executor.layers.mamba.ops.cca import (
        _cca_prefill_hs2_shift_kernel,
        _cca_prefill_dw_conv1d_kernel,
        _cca_prefill_grouped_conv1d_kernel,
    )

    T = sum(seq_lens)
    data = make_test_data(num_prefills, seq_lens, H, E, G, D, num_cache,
                          device, dtype, has_initial_all=True)

    query_start_loc_p = data["query_start_loc_p"].to(device)
    has_initial_states_p = data["has_initial_states_p"].to(device)
    state_indices_p = data["state_indices_p"].to(device=device, dtype=torch.int64)

    seq_lens_t = query_start_loc_p[1:] - query_start_loc_p[:-1]
    req_idx = torch.repeat_interleave(
        torch.arange(num_prefills, device=device, dtype=torch.int32),
        seq_lens_t.int(),
    )

    hs_p_2d = data["hs_p"][:, 0, :].contiguous()
    qk_p_2d = data["qk_packed0_p"][:, 0, :].contiguous()
    hs2_2d = torch.empty((T, H), device=device, dtype=dtype)

    dw_start_loc = query_start_loc_p[:-1] + torch.arange(
        num_prefills, device=device, dtype=query_start_loc_p.dtype)
    total_dw = T + num_prefills
    dw_out = torch.empty((total_dw, E), device=device, dtype=dtype)
    dw_token_offset = torch.arange(T, device=device, dtype=torch.int64) + req_idx.long()
    qk3_2d = torch.empty((T, E), device=device, dtype=dtype)

    BLOCK_H = 128
    BLOCK_E = 128

    def run_k1():
        ph = data["prev_hs"].clone()
        _cca_prefill_hs2_shift_kernel[(T, triton.cdiv(H, BLOCK_H))](
            hs_p_2d, ph, hs2_2d,
            query_start_loc_p, has_initial_states_p, state_indices_p,
            req_idx,
            T=T, H=H, stride_hs_t=hs_p_2d.stride(0),
            stride_prev_row=ph.stride(0), BLOCK_H=BLOCK_H,
        )

    def run_k2():
        cs = data["conv_states"].clone()
        cs_strides = cs.stride()
        _cca_prefill_dw_conv1d_kernel[(num_prefills, triton.cdiv(E, BLOCK_E))](
            qk_p_2d, data["dw_weight"], data["dw_bias"], cs, dw_out,
            query_start_loc_p, has_initial_states_p, state_indices_p,
            dw_start_loc,
            num_prefills=num_prefills, E=E, state_len=2,
            stride_qk_t=qk_p_2d.stride(0),
            stride_cs_row=cs_strides[0], stride_cs_ch=cs_strides[1],
            stride_cs_tok=cs_strides[2], stride_dw_t=dw_out.stride(0),
            HAS_BIAS=True, BLOCK_E=BLOCK_E,
        )

    def run_k3():
        _cca_prefill_grouped_conv1d_kernel[(T, G)](
            dw_out, data["gw_weight"], data["gw_bias"], qk3_2d,
            dw_token_offset,
            G=G, D=D, E=E,
            stride_dw_t=dw_out.stride(0), stride_out_t=qk3_2d.stride(0),
            HAS_BIAS=True, T=T,
        )

    bench_one("K1: hs2_shift", run_k1, warmup, repeats)
    bench_one("K2: dw_conv1d", run_k2, warmup, repeats)
    bench_one("K3: grouped_conv1d", run_k3, warmup, repeats)


if __name__ == "__main__":
    import sys
    if "--profile" in sys.argv:
        profile_main()
    else:
        main()
        print("\n(run with --profile for performance benchmarks)")
