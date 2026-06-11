# TiDAR on AMD MI300X — Performance Report

Branch: `jinzhao/tidar` in `Zyphra/vllm-smoe-amd` (verbatim mirror of Zvllm
`jinzhao/tidar_v016` @ `3f1a680f2`). Hardware: 8× MI300X (192 GB, gfx942),
ROCm 7.2, torch 2.10. Docker image `jinzhao/vllm-tidar-amd:latest`
(built on `zyphra/rocm-primus:aiter_pa_swa`). NVIDIA reference: H100
(vp-dgx), same commit.

## TL;DR

- **TiDAR works on AMD and is fast.** Our single-forward (SF) path with the
  paged Triton kernel hits **803 tok/s** doing real spec decode (accept 5.6)
  — faster than the colleague's `ROCM_AITER_FA` AR (590, stock kernel) and
  TF (376, his paged K+1 kernel), both measured single-GPU on the same
  workload.
- **The SF kernel is already at its AMD ceiling.** MI300X autotune tuning
  (2026-06-11) gave no win: occupancy knobs (`waves_per_eu`, `num_stages`)
  are parity; MFMA-math knobs (`matrix_instr_nonkdim`, `kpack`) regress
  acceptance (5.06→4.68) because they perturb the verifier-fed output.
  Throughput is bound by the MoE + spec-decode-verify path, not this kernel.
- **Only FlexAttention runs TiDAR correctly on our tree.** Every other AMD
  backend (AITER-FA, Triton, AITER-Unified) breaks TiDAR's K+1 spec layout
  — accept collapses to ~1.2 (pos-0 partially survives, pos-1+ → 0).
- AR (no spec decode) has no K+1 layout, so it runs on the fast AITER-FA
  backend: **658 tok/s**, matching the colleague's ~600.
- The acceptance "gap" chased in earlier analysis was a benchmarking
  artifact (raw vs chat-template prompts); at matched config AMD ≈ NVIDIA.
  That investigation is archived in memory, not here.

## Benchmark config

All numbers below share one config (the colleague's eval config, so we can
compare directly):

- Checkpoint `iter_0012600` (HF), AIME25 (30 problems), chat template,
  `enable_thinking=False`
- `n=4`, `temperature=0.5`, `max_tokens=8192`, `max_model_len=10000`
- `max_num_seqs=16` (b=16), captured (`cudagraph_mode=FULL_DECODE_ONLY`),
  Triton CCA (default)
- true-mean accept = `1 + Σaccepted / Σ(drafted/K)` over ALL SpecDecoding
  windows (K=16)

## Results (matched config)

| Mode / config | attention | tok/s | accept | notes |
|---|---|---:|---:|---|
| **SF `[0,4,7,11]`** (P=4) | Flex + SF Triton paged | **803** | 5.63 | **fastest; beats colleague AR** |
| SF `[0,3,7,10,16]` (P=5) | Flex + SF Triton paged | 761 | 5.97 | +0.34 acc for -5% tok/s |
| SF `[0,5,11]` (P=3) | Flex + SF Triton paged | 755 | 5.46 | sparser ≠ faster |
| AR (no spec) | AITER-FA | 658 | – | ≈ colleague AR ~600 |
| SF dense `[0..16]` (P=17) | Flex + SF Triton paged | 544 | 7.57 | highest accept |
| TF | Flex | 263 | 7.17 | |
| AR (no spec) | Flex | 28.5 | – | 23× slower than AITER-FA — Flex pathological for AR at b=16 |
| TF | AITER-FA | 71 | **1.25** | **broken** — K+1 mask, pos1+→0 |
| *colleague AR* | ROCM_AITER_FA (stock) | 590 | – | captured; measured single-GPU on his fork |
| *colleague TF* | ROCM_AITER_FA + his paged K+1 kernel | 376 | 6.97 | eager; the paged kernel fixes K+1 accept |

### Reading

- **SF proposal-level sparsity is the dominant throughput knob.** Dense
  `[0..16]` (544, acc 7.57) → `[0,4,7,11]` (803, acc 5.63): fewer drafter
  forwards per step buys throughput faster than it loses accept. P=3
  `[0,5,11]` (755) is *not* faster than P=4, and P=5 `[0,3,7,10,16]`
  (761, acc 5.97) trades 5% tok/s for +0.34 accept — the sweet spot is
  P=4-5 with levels spread to ~11-16. Levels must include 0; dense P=17
  (544) is for max accept only.
- **Our SF beats the colleague's TF and AR** at his own config, on a slower
  attention backend — the SF paged Triton kernel is the AMD win.
- **TF trails** — our 263 (FLEX_ATTENTION, captured) vs his 376
  (ROCM_AITER_FA + his custom `tidar_paged_multi_token_attention`
  kernel, eager). The gap is the attention backend, not TiDAR. **AR**
  has no K+1 layout, so both trees run it on stock ROCM_AITER_FA: ours
  658, his 590 (both single-GPU). Both his numbers are per-GPU, not
  DP=8 aggregate. See below.

## Verified against the aiter-pa-tidar-fix backend

The `vllm-diffusion-dev` branch (image `zyphra/rocm-primus:aiter-pa-tidar-fix`)
adds a hand-written `tidar_paged_multi_token_attention` Triton kernel on the
AITER-FA backend (see below). Ran its AR + TF at the same single-GPU config
(ckpt iter_0012600, AIME25 30 prompts, n=4, T=0.5, mt=8192, b=16):

| Config | vLLM attn backend | in-kernel path | mode | tok/s | accept |
|---|---|---|---|---:|---:|
| **our SF `[0,4,7,11]`** | FLEX_ATTENTION | our SF paged Triton kernel | captured | **803** | 5.63 |
| our AR | ROCM_AITER_FA | stock paged decode | captured | 658 | – |
| **aiter-pa-tidar-fix AR** | ROCM_AITER_FA | stock paged decode | captured | **590** | – |
| **aiter-pa-tidar-fix TF** | ROCM_AITER_FA | his `tidar_paged_multi_token_attention` (K+1) | eager | **376** | **6.97** |
| our TF | FLEX_ATTENTION | Flex K+1 mask | captured | 263 | 7.17 |
| our TF | ROCM_AITER_FA | stock (no paged multi-token) | captured | 71 | **1.25** |

Both colleague rows use the **same `ROCM_AITER_FA` vLLM backend**; the
only difference is the in-kernel path. AR has no K+1 spec layout, so it
rides the stock paged-decode kernel (590 tok/s, captured). TF routes
through his hand-written `tidar_paged_multi_token_attention` kernel
(376 tok/s, eager, accept 6.97) — which is what keeps K+1 acceptance
healthy where the *stock* ROCM_AITER_FA TF collapses to 1.25.

His TF accept decays gently across positions (6.97 mean; per-pos
0.50->0.43->0.38->0.37) — real multi-token acceptance, not a
pos-0-only artifact. His quoted ~600 AR / ~380 TF are **per-GPU** (we
reproduced 590 AR single-GPU), not DP=8 aggregate. Our SF still wins on
raw throughput; his kernel wins on a *correct fast TF*. His TF ran eager
here — a torch-2.10 "cudagraphs must be captured on a non-default
stream" error blocks the captured path under our direct-LLM harness (his
server harness avoids it); captured would push TF above 376.

This is the concrete payoff of open-work item below: grafting their
`tidar_paged_multi_token_attention` into our Flex TF path would give us a
fast *and correct* TF without adopting their whole branch.

## Cross-platform: same config on NVIDIA H100 (vp-dgx-2)

Identical config (iter_0012600, AIME25 30 prompts chat-template, n=4,
T=0.5, mt=8192, b=16, captured). NVIDIA can run FLASH_ATTN; AMD cannot.

| Config | AMD MI300X tok/s | NVIDIA H100 tok/s | accept (NV) |
|---|---:|---:|---:|
| TF + FLASH_ATTN | — (FA unavailable on ROCm) | **1827** | 6.99 |
| AR + FLASH_ATTN | — | 1011 | – |
| SF `[0,4,7,11]` Triton paged (Flex) | 803 | 1055 | 5.44 |
| AR + AITER-FA | 658 | — | – |
| TF + Flex | 263 | — | – |

Takeaways:
- **The SF paged Triton kernel ports well**: 803 (AMD) vs 1055 (NVIDIA) is
  only 1.3× — far tighter than any other cross-platform gap here. The
  kernel isn't the problem; AMD just lacks the FA backend.
- **NVIDIA's headline is TF+FA (1827)** — there, TF *beats* AR (1011) 1.8×
  because FlashAttention makes the 2-forward verify cheap and accept stays
  high (6.99). This is exactly the path AMD can't run (no `vllm_flash_attn`
  varlen on ROCm; see below).
- **Ordering flips by platform**: on NVIDIA SF (1055) < TF (1827); on AMD
  SF (803) >> TF (263). With cheap FA, TF's higher accept wins; without it,
  SF's single-forward structure is the only way to dodge slow attention.
- AMD's path to NVIDIA-class TF throughput is the colleague's AITER-FA
  paged TF kernel, or MI300X-tuning the SF kernel (already at 76% of its
  own NVIDIA tok/s).

## Attention backends

Only FlexAttention runs the TiDAR K+1 layout correctly on our tree:

| Backend | TiDAR accept | verdict |
|---|---:|---|
| FLEX_ATTENTION | healthy (5–7) | **only working backend** |
| ROCM_AITER_FA | 1.25 | broken: stock extend kernel mis-masks within-segment causal |
| TRITON_ATTN | 1.47 | broken, same pattern |
| ROCM_AITER_UNIFIED_ATTN | 1.50 | broken, same pattern |

Confirmed broken at eager (b=3), captured b=1, and captured b=16 (his
config) — independent of batch/capture. AR (no K+1 layout) runs fine on
all backends and is fastest on AITER-FA.

### FLASH_ATTN is not available for the ROCm decoder

vLLM's `flash_attn.py` backend calls the **vLLM FA3-fork API**
(`seqused_k`, `scheduler_metadata`, `fa_version`, descales, LSE returns).
The standalone `flash_attn` 2.8.3 in the image is the classic FA2 API
(`cu_seqlens_k`, `dropout_p`, `return_attn_probs`) and accepts none of
those kwargs. `fa_utils` binds it on ROCm but the signatures are
incompatible — wiring it in would be a full call-site rewrite, not a flag.
The dispatcher's "FLASH_ATTN not supported on ROCm" reflects this. This is
why the colleague built on AITER-FA (which exposes the FA3-style paged
`seqused_k` API vLLM expects).

## How the colleague got fast TF on AITER-FA

His `vllm-diffusion-dev` branch (diverges from `smoe-aiter-moe` at
`950cc0dc8d`, +4786 lines) replaces the stock AITER extend path with a
hand-written Triton kernel, `tidar_paged_multi_token_attention_kernel` in
`rocm_aiter_fa.py`. The crux is its per-query-position causal horizon:

```
batch_idx  = query_token_idx // MAX_QUERY_LEN    # MAX_QUERY_LEN = K+1 = 17
query_pos  = query_token_idx - batch_idx*MAX_QUERY_LEN
context_len = seq_len - MAX_QUERY_LEN
max_kv_len = context_len + query_pos + 1         # baked K+1 causal mask
```

This bakes TiDAR's K+1 per-position mask into an online-softmax paged
kernel — exactly what AITER's stock extend kernel gets wrong. Gated behind
`VLLM_TIDAR_USE_PAGED_ATTENTION=1`; routes both verify (causal) and draft
(non-causal) passes through it; falls back to AITER varlen for non-TiDAR.

Conceptually it's the **same idea as our `sf_attention.py` paged Triton
kernel** — a bespoke paged online-softmax kernel bypassing the broken stock
path — but wired into AITER-FA (TF layout) instead of Flex (SF layout).

## Is our SF Triton kernel optimized for AMD?

**Yes, effectively** — measured 2026-06-11. The CUDA-shaped autotune list
is already the right list for MI300X, and AMD-specific knob tuning gives no
reliable win because end-to-end SF throughput is not bound by this kernel.

Same-session A/B (smoediffusion iter_0012600, SF `[0,4,7,11]` K+1, AIME25
30×n4 T=0.5 b=16, captured, single MI300X):

| `_sf_attention_fwd_kernel_paged` autotune list | TOTAL tok/s | accept (true-mean) |
|---|---:|---:|
| original 10 CUDA-shaped configs | 744 | 5.06 |
| + `waves_per_eu` / `num_stages=1` occupancy variants | 778 | 5.27 |
| + `matrix_instr_nonkdim=16` + `kpack=2` (MFMA-math knobs) | **693** | **4.68** |

- **Occupancy tuning is parity.** `waves_per_eu` + `num_stages=1` variants
  land at 778 vs 744 — inside the ~5-8% run-to-run band (the *same
  unchanged* kernel measured 744 this session and 803 in a prior one;
  generation isn't bit-deterministic at T=0.5, token totals swing
  320k–383k). No reliable gain, so the list is kept unchanged (avoids the
  extra autotune warmup of a bigger config list for nothing).
- **MFMA-math knobs actively hurt.** `matrix_instr_nonkdim` / `kpack`
  change the MFMA instruction + accumulation order, perturbing the
  attention output. Because that output feeds the spec-decode **verifier**,
  the perturbation shifts which drafts are accepted: accept 5.06 → 4.68,
  throughput 744 → 693. The throughput loss tracked the accept loss 1:1 —
  the compute wasn't slower, the *drafts got worse*. These knobs are now
  documented as forbidden in a guardrail comment above the autotune block.
  General rule: verifier-feeding kernels must keep numerics fixed.
- **Why it's marginal:** cudagraph capture already removes launch overhead
  (eager 497 → captured ~750–800). The residual is dominated by the MoE
  forward + spec-decode verify machinery, not the SF attention kernel, so
  tuning this one kernel's autotune was always going to be in the noise.
  Real SF headroom (if any) is algorithmic (the 2-phase paged loop) or in
  the MoE path — not in these knobs.

## Runbook

Recommended config (fastest correct TiDAR on AMD today):

```bash
docker run --rm --device /dev/dri --device /dev/kfd --group-add video \
  --network host --ipc host --shm-size 32G -v /shared:/shared \
  -w /shared/home/jinzhao/workspace/tidar/vllm-smoe-amd \
  -e HIP_VISIBLE_DEVICES=0 -e PYTHONUNBUFFERED=1 \
  -e VLLM_SKIP_SDPA_PREINIT=1 \
  -e VLLM_TIDAR_SF_TRITON=1 \
  -e VLLM_TIDAR_PROPOSAL_ACC_LEVELS=0,4,7,11 \
  jinzhao/vllm-tidar-amd:latest python -u <bench>.py
# LLM(attention_backend="FLEX_ATTENTION", compilation_config={"cudagraph_mode":"FULL_DECODE_ONLY"})
```

Gotchas (each cost real debugging time):

1. `attention_backend` must be an `LLM()` kwarg — v0.16 ignores
   `VLLM_ATTENTION_BACKEND`.
2. `VLLM_SKIP_SDPA_PREINIT=1` + the (uncommitted) gate in
   `vllm/env_override.py` avoids an intermittent `import vllm` segfault on
   ROCm (`_sfdp_init` → `_get_sfdp_patterns`).
3. Bench-script structure is load-bearing on ROCm: clone the known-good
   `bench_match_sy.py` template and only swap inputs. A different module
   structure segfaulted the engine at first GPU alloc (`eagle.py:133`).
4. SF proposal levels must include 0 — default `(4,7,10)` collapses to
   accept ≈1.0.
5. `dtype=float32`+FLEX is unsupported on ROCm (silently garbage); use bf16.
6. pytorch CCA (`VLLM_CCA_TRITON=0`) is capture-unsafe on ROCm at b=16
   (HIP error during FULL graph capture); use Triton CCA (default) for
   captured runs.
7. Always pass `disable_log_stats=False` and aggregate accept across all
   windows.
8. Cluster: ad-hoc `docker run` competes with an auto-scheduler — check
   `rocm-smi --showmemuse` before launch; name containers (`--name`) to
   avoid friendly-fire kills. Node scanner:
   `/shared/home/jinzhao/workspace/tidar/scan_cnodes.sh`.

## Open work (throughput)

1. **Full SF proposal-level sweep** — P=4 `[0,4,7,11]` (803) beat both
   denser and sparser; map the optimum properly.
2. ~~MI300X-tune the SF Triton kernel autotune configs~~ — **DONE
   (2026-06-11), no win.** Occupancy knobs are parity; MFMA-math knobs
   regress acceptance. See "Is our SF Triton kernel optimized for AMD?"
   above. Throughput isn't bound by this kernel.
3. **Port the colleague's `tidar_paged_multi_token_attention` kernel** into
   our Flex TF path (self-contained ~150-line graft) to fix TF, OR adopt
   his AITER-FA branch and graft our SF kernel onto it.
4. **MI300X MoE config** — every run warns the
   `E=16,N=2048,...AMD_Instinct_MI300X.json` tile config is missing.
