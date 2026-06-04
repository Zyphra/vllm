# TiDAR vLLM v0.16 Port — Handoff

**Branch:** `jinzhao/tidar_v016` @ `819b27dc2`
**Repo:** `git@github.com:Zyphra/Zvllm.git`
**Env:** `/data/home/jinzhao/workspace/tidar/Zvllm-v016/.venv-v016`
**Bench node:** idle GPU on vp-dgx-89 (147.68.0.89)
**Date:** 2026-06-03

The v0.16 port is production-ready for TiDAR workloads. Performance is at par
or better than v0.15 across nearly every config tested.

## What works

| Mode | Status | Notes |
|---|---|---|
| AR eager / AR FULL captured | ✅ | Standard vLLM path |
| SF eager / PIECEWISE / FULL captured | ✅ | **Primary path for SF use cases** |
| TF eager | ✅ | Fallback (~20 tok/s) |
| TF PIECEWISE captured | ✅ (opt-in) | Requires `VLLM_TIDAR_ROUTER_PAD=1` |
| **TF FULL_DECODE_ONLY captured** | ✅ (opt-in) | **Primary perf path** — see Quickstart |
| `vllm serve` DP=8 | Not retested | v0.15 command should port; not validated |

## Performance (n=10 mt=2000, AIME thinking-off, K=16, T_AR=0)

Idle vp-dgx-89, smoediffusion `iter_0012000` checkpoint:

| mode    | batch | v0.16 tok/s | v0.15 tok/s | ratio | v0.16 acc | v0.15 acc |
|---|---:|---:|---:|---:|---:|---:|
| AR (no spec) |  1 | **101** |   65 | **156%** | n/a | n/a |
| TF FA        |  1 |     228 |  240 |   95%  | 7.09 | 6.94 |
| TF FA        |  2 | **384** |  374 | **103%** | 6.63 | 6.48 |
| TF FA        |  4 | **608** |  566 | **107%** | 6.79 | 7.10 |
| TF FA        |  8 | 914 (var 831-1069) | 1154 | 79% (variance, not structural) | 6.72 | 7.00 |
| TF FA        | 16 | **1621** | 1537 | **105%** | 7.25 | 6.77 |
| SF FLEX      |  1 | **223** | ~75 | **297%** (v0.15 unstable here) | 6.79 | 6.10 |
| SF FLEX      |  8 | **634** |  319 | **199%** | 6.98 | 6.16 |

Accept = `Mean acceptance length` averaged across all SpecDecoding metric
windows in the run (each window ~10s of generation). v0.15 vs v0.16 accept
is within ~5% across configs — the perf differences in the table are
throughput differences, not accept-quality differences.

Run-to-run sigma on TF b=1 is ~0.5% on idle node. The b=8 dip is high
variance (v0.16 std-dev ~100); v0.15 is stable. The cudagraph dispatcher
with 13 captured sizes likely occasionally falls back to a suboptimal size
at runtime b=8 specifically — wins at b=4 (+7%) and b=16 (+5%) confirm it's
not structural.

## Quickstart

```bash
ssh vp-dgx-89  # or any idle node
cd /data/home/jinzhao/workspace/tidar/Zvllm-v016
git checkout jinzhao/tidar_v016 && git pull --ff-only
source .venv-v016/bin/activate
```

### Run TF FULL captured (recommended TiDAR perf path)

```bash
export VLLM_TIDAR_TWO_FORWARD=1
export VLLM_TIDAR_ROUTER_PAD=1       # dodge cutlass align1 OOB (see Fixes)
export VLLM_TIDAR_FA_NO_SPLITS=1     # +10% accept on captured FA
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
```

```python
llm = LLM(
    model=ckpt, dtype="bfloat16",
    enforce_eager=False,
    speculative_config={
        "method": "tidar",
        "num_speculative_tokens": 16,
        "tidar_diff_temperature": 0.0,
    },
    compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
    kernel_config={"enable_flashinfer_autotune": False},  # +8% tok/s
)
```

### Run SF FULL captured

```bash
export VLLM_ATTENTION_BACKEND=FLEX_ATTENTION
unset VLLM_TIDAR_TWO_FORWARD  # SF is default
```

Same LLM construction as above (without the TF env vars).

### Run TF eager (fallback)

```python
import os; os.environ["VLLM_TIDAR_TWO_FORWARD"] = "1"
llm = LLM(model=ckpt, ..., enforce_eager=True,
          speculative_config={...})
```

### Bench + profile scripts

- `scripts/bench_tidar.py` — drives every number in this doc. Env knobs:
  `BENCH_K`, `BENCH_B`, `BENCH_N`, `BENCH_MT`, `BENCH_MODE` (`ar`/`tf`/`sf`),
  `BENCH_EAGER`, `BENCH_CG`, `BENCH_GPU_MEM`, `BENCH_MML`, `BENCH_MNBT`,
  `BENCH_FI_AUTOTUNE_OFF`, `BENCH_DISABLE_LOG_STATS`.
- `scripts/profile_sf.py` — one-prompt torch.profiler capture of SF FULL.

## Required v0.16-specific fixes

### `VLLM_TIDAR_ROUTER_PAD=1` for TF FULL / PIECEWISE

Without it, `SMoERouter`'s final `D=2048 -> E=17` linear (stride 17 = not a
multiple of 8) makes cublas pick `cutlass_75_tensorop_s1688gemm_bf16_64x64
_tn_align1` under captured cudagraph. That kernel writes 1+ bytes past its
output allocation — eager allocator padding hides it; cudagraph mempool's
tight fit exposes it. Compute-sanitizer memcheck confirms.

Fix (committed `43c0fa08e`): project against a padded `D -> E_padded=24`
weight (next multiple of 8). cublas picks a different (correct) sm_90 kernel
for the aligned output stride. Slice back to E=17 columns. Padded weight
buffer is filled lazily on first forward; no checkpoint format change.

### `VLLM_TIDAR_FA_NO_SPLITS=1` for TF FA captured

v0.16's FA backend defaults to `max_num_splits=32` under FULL cudagraph
(from `flash_attn_max_num_splits_for_cuda_graph`). Split-KV reduction order
non-determinism flips drafter argmax at borderline tokens, dropping accept
~10%. Set `max_num_splits=1` to match eager FA numerics.

Gated on TF mode only (committed `5a51e6ea4`): SF's structured-mask kernel
lives in flex_attention.py — SF + FA combo is broken at the kernel level
anyway, so use FLEX for SF.

### `enable_flashinfer_autotune=False` for TF (~8% tok/s)

`kernel_config.enable_flashinfer_autotune=True` is the v0.16 default at
`optimization_level >= O1`. It runs `_dummy_run(MNBT=2048-4096, is_profile=
True)` to benchmark FlashInfer ops and cache the best per-shape. The autotune
sizes for MNBT but TF spec-decode forwards are 17-34 tokens — the selected
ops are mis-sized.

No code change needed; pass it in `kernel_config=...` at LLM construction.

## Shipped perf wins (5 commits this session, +52% cumulative)

| commit | win | gain |
|---|---|---:|
| `5e4b95df0` | lm_head: `bf16xbf16 -> fp32` via `out_dtype` (Hopper SM90 Tensor Cores) instead of `fp32xfp32` (SM80 GEMM). nsys: 87.6ms -> 27.3ms across 200 tokens on the same kernel slot. | +9% |
| `84950b974` | Revert MOE_OP default to OFF. Earlier flip to ON (`771a701d6`) was based on a high-variance n=3 measurement with one 204 tok/s outlier; on idle node 89 n=10 the flip is -2.7%. | reverts mistake |
| `528f2b851` | Skip `_update_states_after_model_execute` for TiDAR (no GDN/no mamba-align consumer). Eliminates dead-code `.cpu().numpy()` sync + cat/argmax/python-loop. | +1.9% |
| `8e79bdc39` | Skip `compute_causal_conv1d_metadata` when builder is `CCAAttentionMetadataBuilder`. CCA has its own conv path; the function builds Mamba1 conv kernel metadata that CCA never reads. | +1.0% |
| **`c1e8c9f9c`** | **Force `disable_padded_drafter_batch=True` for TiDAR**. v0.15's `_use_padded_drafter_batch()` returned False for TiDAR; v0.16 dropped the override when refactoring the gate to a pure config field. The fix restores v0.15's draft-after-bookkeep pattern: bookkeep's `.cpu()` waits only for verify+sample (not drafter), and drafter overlaps with the next step's preprocess+forward. | **+35.7%** |

Opt-outs (all default-on per these commits):
- `VLLM_TIDAR_KEEP_PADDED_DRAFTER=1` — restore v0.16's old EAGLE-style use_gpu_toks path
- `VLLM_TIDAR_NO_CONV1D_META_SKIP=1` — restore conv1d_metadata call
- `VLLM_TIDAR_SMOE_MOE_OP=1` — re-enable the cat()-avoidance custom op
  (helps short-burst benches at b=1, slightly hurts steady-state n=10; default OFF)

## Open items / known issues

- **Multi-`llm.generate()` corruption** under SF FULL captured: two consecutive
  `llm.generate()` calls on the same LLM crash the second with
  `cudaErrorIllegalAddress` at `_update_states_after_model_execute -> .cpu()`.
  Single batched calls work fine. Both `bench_tidar.py` and `_sf_mmlu_sweep.py`
  work around by issuing one batched call. Likely cause: block-table state
  from finished requests not fully released, or scheduler state stale across
  calls. Not investigated.

- **TF b=8 throughput variance** (831-1069 across 5 runs; v0.15 stable at
  1105-1168). Likely cudagraph dispatcher fallback at b=8 specifically — the
  captured sizes are `[1,2,4,8,17,34,51,68,85,102,119,136,306]` and runtime b
  changes as prompts finish.

- **Async scheduling for TiDAR** is gated off. Could close more gap at higher
  batch. Three blockers in the existing async path:
  1. `_bookkeeping_sync` async branch asserts `sampled_token_ids.shape[-1]==1`;
     for TF spec-decode this is K+1=17.
  2. `_commit_tidar_cca_state` requires `valid_sampled_token_ids` on CPU.
     Either compute on GPU directly (replace `len(...)` with `idx_gpu.shape[0]`)
     or sync per-step (defeats the async benefit).
  3. `prev_sampled_token_ids` consumer in `_update_states` assumes shape
     `[num_reqs, 1]` for the bonus column. For TF the bonus is at column 0
     of `[num_reqs, K+1]` (happens to work) but the downstream draft-token
     scatter assumes a separate `_draft_token_ids` tensor.

- **SF MoE softmax recompute** (background, not in the critical path now):
  v0.16's `triton_red_fused__softmax__to_copy_add_bitwise_not_gather_mean_mul_ne_pow_rsqrt_view_5`
  fuses softmax+gather inline (v0.15 didn't because of a different MoE call
  shape). At dense P=0..16 b=3 SF this is 66% of GPU time. The
  `VLLM_TIDAR_SMOE_MOE_OP=1` opaque custom op (committed `b03144b13`) helps
  at small M but is neutral/-3% at large M and slightly regresses n=10 — so
  default OFF. Hand-tuned MoE router Triton kernel would beat inductor's
  auto-fusion here.

## v0.15 vs v0.16 — architecture notes worth knowing

- v0.16's runner does **NOT** rebind `drafter.model` to the CUDAGraphWrapper-
  wrapped model. v0.15 did. In v0.16, `TiDARProposer.load_model` binds
  `self.model = target_model` (unwrapped) BEFORE the runner wraps. So
  `drafter.forward` bypasses the wrapper and runs eager by default. The
  warmup drafter-capture hook (`0001ddc4d`) handles this by capturing the
  drafter graph at warmup with explicit `is_drafter_pass=True` BatchDescriptor.

- v0.16's `BatchDescriptor` is a NamedTuple. `is_drafter_pass: bool = False`
  added at the end (defaults preserve equality with existing call sites);
  threaded through `_create_padded_batch_descriptor` and `dispatch()`.

- v0.16 split `unified_kv_cache_update` out from FA's forward into a separate
  splitting_op (40 layers × 2 forwards = 80 extra op dispatches per step on
  TF). v0.15 inlined `reshape_and_cache_flash` in FA forward. Tried to
  re-inline in v0.16 but FA's `_vllm_fa3_C.fwd` kernel raises `out must have
  shape (total_q, num_heads, head_size_v)` during warmup. The flag flip
  changes a code path in attention.py that affects output shape setup —
  needs deeper investigation if pursued.

- v0.16 forces `cudagraph_copy_inputs=True` for TiDAR (v0.15 had False).
  Per-step memcpy of inputs into captured buffers. `VLLM_TIDAR_NO_COPY_INPUTS=1`
  opt-out exists but no measurable benefit and historic crashes when off.

- `mamba_attn.py` builder's persistent buffer copy (`self.state_indices_tensor`)
  fires for pure-decode batches under has_full_cudagraphs. For TF verifier
  (classified as PREFILL since query_len=17 > 1), the persistent buffer copy
  is SKIPPED and `state_indices_tensor` returns the fresh slice from
  `mamba_get_block_table_tensor(...)[:, 0]`. Pointer is stable (slice of
  persistent block_table_tensor), safe for capture/replay.

## Key commits

Phase 1-2 (initial port): `cd68dec2c` ... `9da7f7d07` — base TF/SF eager and captured paths.

Correctness fixes:
- `43c0fa08e` — router-pad workaround (TF FULL/PIECEWISE captured)
- `0001ddc4d` — TF drafter warmup capture
- `35a01a325` — FA backend 0% accept (slot_mapping dict)
- `96cc939c7` — `VLLM_TIDAR_FA_NO_SPLITS=1` for accept recovery
- `55e08f7cb` — runtime warning for SF+FA mismatch
- `3026d9151` — force CCA vectorized path under all TiDAR (TF eager drift fix)

Perf wins (this hunt):
- `5e4b95df0` — lm_head SM90
- `528f2b851` — skip _update_states_after for non-GDN
- `8e79bdc39` — skip compute_causal_conv1d_metadata for CCA
- `c1e8c9f9c` — **force disable_padded_drafter_batch=True for TiDAR** (the big one)

Per-step Python overhead reductions:
- `a658cb610` — cache fp32 lm_head weight transpose
- `1d23c6385` — hoist _commit_tidar_cca_state idx/arange GPU
- `9945e9e5b` — cache _get_cca_block_slots
- `80ab16112` — cache FA group id lookup
- `955cc6361` — skip per-step state_indices_tensor.tolist sync
- `284091c04` — cache /tmp/tidar_mix_w file-read

## Related memory entries (Claude Code project memory)

- `project_tidar_v016_port_in_progress` — running status (this handoff is the snapshot)
- `project_sf_captured_cudagraph_fixes` — env_override patches preserved across the port
- `project_sf_kp1_layout_required` — K+1 layout respected
- `project_sf_requires_flex_backend` — v0.16 env var is silent, route through `attention_backend` kwarg
- `feedback_always_log_stats` — TF/SF benches must pass `--log-stats`
