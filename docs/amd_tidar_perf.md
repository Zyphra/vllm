# TiDAR on AMD MI300X — Performance Report

Branch: `jinzhao/tidar` in `Zyphra/vllm-smoe-amd` (verbatim mirror of Zvllm
`jinzhao/tidar_v016` @ `3f1a680f2`). Hardware: 8× MI300X (192 GB, gfx942),
ROCm 7.2, torch 2.10. Docker image `jinzhao/vllm-tidar-amd:latest`
(built on `zyphra/rocm-primus:aiter_pa_swa`). NVIDIA reference: H100
(vp-dgx), same commit.

## TL;DR

- **TiDAR works on AMD and is fast.** Our single-forward (SF) path with the
  paged Triton kernel hits **803 tok/s** doing real spec decode (accept 5.6)
  — faster than a colleague's AITER-FA-paged AR (~600) and TF (~380) on the
  same workload.
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
| *colleague AR* (ref) | AITER-FA paged | *~600* | – | his vllm-diffusion-dev fork |
| *colleague TF* (ref) | AITER-FA paged | *~380* | – | his fork |

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
- **TF trails** (263 vs his 380): we're on FlexAttention; he wrote a
  custom AITER-FA paged kernel (see below). The gap is the attention
  backend, not TiDAR.

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

No — it runs correctly and auto-adapts, but the tuning space is
NVIDIA-shaped. `_sf_attention_fwd_kernel_paged` uses `@triton.autotune`
over 10 configs keyed on `(verify_len, Kp1, P_props, num_heads, head_dim,
block_size)`. Issues on MI300X:
- `num_stages=2/3` is a CUDA software-pipelining concept; largely inert on
  CDNA, so 3 of the 10 configs are effective duplicates.
- `num_warps`/`BLOCK_Q`/`BLOCK_KV` were picked for H100 occupancy, not
  MI300X (304 CUs, 64-lane wavefronts, different LDS/regfile). No AMD knobs
  (`waves_per_eu`, `matrix_instr_nonkdim`).

Autotune still picks the best available config, so the 803 tok/s is with
this CUDA-shaped space — an MI300X-specific config list is **unexplored
headroom** and a cheap, self-contained edit (just the `configs=[...]` list).

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
2. **MI300X-tune the SF Triton kernel autotune configs** — cheap edit,
   unexplored headroom.
3. **Port the colleague's `tidar_paged_multi_token_attention` kernel** into
   our Flex TF path (self-contained ~150-line graft) to fix TF, OR adopt
   his AITER-FA branch and graft our SF kernel onto it.
4. **MI300X MoE config** — every run warns the
   `E=16,N=2048,...AMD_Instinct_MI300X.json` tile config is missing.
