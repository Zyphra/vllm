# TiDAR-on-AMD Handoff

_Last updated: 2026-07-07. Owner: jinzhao. Scope: TiDAR two-forward (TF) decode on MI300X for the 80B SMoE/Zaya checkpoint._

## 1. Current state

The old production TiDAR TF path on AMD was correct after the AITER-FA causal-mask fix, but it was host-overhead bound and only reached **177 tok/s** at b8/MT512. The current working-tree path is **Path B**: TiDAR TF runs on the fork's V2 GPU runner (`vllm/v1/worker/gpu/model_runner.py`, selected by `VLLM_USE_V2_MODEL_RUNNER=1`) and uses native V2 async scheduling plus cudagraph replay for both verify and draft forwards.

**Production-comparable headline:** **553.839 tok/s** on MI300X at b8/MT512, `ROCM_AITER_FA`, V2 async, `FULL_AND_PIECEWISE`, K=16. This is **3.13x** over the old 177 tok/s production baseline.

Path B is not "stock vLLM 0.24"; it is a v0.24-style adoption inside the v0.16-based fork. v0.24/DiffusionGemma helped by showing the clean architecture: V2 runner + spec-decode data path + model-state-style per-request state + async/cudagraph-friendly execution.

2026-07-07 AMD/NVIDIA parity update: low AMD TF acceptance was traced to runs that missed the TiDAR TF paged AITER attention path and fell back to generic ROCm AITER extend attention. The branch now defaults that route on for TiDAR TF; older trees should set `VLLM_TIDAR_USE_TF_PAGED_ATTENTION=1` explicitly.

## 2. TF throughput numbers

### Current V2 TiDAR TF, single GPU, async captured

Run context: `ibm-cnode-19`, GPU 3, `FULL_AND_PIECEWISE`, `ROCM_AITER_FA`, K=16, AIME thinking-on prompts, `num_speculative_tokens=16`, no EPLB.

**Best measured V2 TF throughput by batch size:**

| bsz | Backend / runner | Best tput | Mean accept len | Run shape | Notes |
|---:|---|---:|---:|---|---|
| 1 | V2 + async + `ROCM_AITER_FA` + `FULL_AND_PIECEWISE` + verify/drafter graphs | `108.635 tok/s` | `4.000` | MT128, offset 0, seed 0 | Current V2 |
| 3 | V2 + async + `ROCM_AITER_FA` + `FULL_AND_PIECEWISE` + verify/drafter graphs | `298.601 tok/s` | `4.000` | MT256, offset 12, seed 1 | Best of two b3 cases |
| 8 | V2 + async + `ROCM_AITER_FA` + `FULL_AND_PIECEWISE` + verify/drafter graphs | **`553.839 tok/s`** | `3.103` | MT512, offset 0, seed 0 | Current headline |
| 16 | V2 + async + `ROCM_AITER_FA` + `FULL_AND_PIECEWISE` + verify/drafter graphs | `940.229 tok/s` | `6.548` | MT2000, AIME25 thinking-off, `iter_0012600`, seed 0 | Long v0.16-like config; corrected accept |
| 64 | V2 + async + `ROCM_AITER_FA` + `FULL_AND_PIECEWISE` + verify/drafter graphs | `2743.088 tok/s` | `6.120` | MT2000, AIME25 thinking-off, `iter_0012600`, seed 0 | Cap fixed; single full-sweep repeat. Separate b64 rerun best: `2699.349 tok/s`, accept `6.552` |

Other measured V2 TF cases:

| Case | Throughput | Mean accept len | Notes |
|---|---:|---:|---|
| b3, MT128, offset 0, seed 0 | `224.316 tok/s` | `3.048` | Lower than b3/MT256 offset12 |
| b8, MT256, offset 8, seed 1 | `438.955 tok/s` | `2.485` | Harder prompt slice |
| b1, MT2000, AIME25 thinking-off | `100.146 tok/s` | `5.208` | Long v0.16-like config, `iter_0012600` |
| b8, MT2000, AIME25 thinking-off | `482.468 tok/s` | `5.400` | Long v0.16-like config; repeat 1 was `459.985 tok/s`, accept `6.499` |

Acceptance caveats from the 2026-07-06 follow-up:

- Current V2 acceptance probes use **target temperature 0.0** and **`tidar_diff_temperature=0.0`**. The V2 drafter takes argmax logits, and the V2 rejection path accepts while target argmax equals draft argmax.
- Do not compare these short V2 probes directly with the v0.16 6-7 table without matching workload. The v0.16 table used `iter_0012000`, AIME thinking-off, `MT=2000`, and long metric windows. The available current checkpoint is `iter_0012600`.
- The old probe formula `total_tokens / (batch * propose_calls)` undercounts acceptance when requests finish early. `/shared/home/jinzhao/tidar_m5_logs/probe_v2_tidar_accept.py` now also accumulates actual V2 `num_sampled` across active requests and reports the old formula as `legacy_mean_accept_len`.
- `vllm/attention/ops/tf_attention.py` now honors `VLLM_TIDAR_TF_PAGED_NO_SPLITS=1` and the legacy `VLLM_TIDAR_FA_NO_SPLITS=1`. `vllm/v1/attention/backends/rocm_aiter_fa.py` now uses a gated TiDAR K+1 decode branch instead of the older unconditional `_dmql > 1` branch.
- Corrected b16/MT256 eager diagnostic on `ibm-head -> Slurm cnode-26`, GPU3, `iter_0012600`, AIME first16, `VLLM_TIDAR_USE_TF_PAGED_ATTENTION=1`, `PATCH_PROBE_REPEATS=2`: the log confirms `Using TiDAR TF paged attention in ROCm AITER-FA` for both draft (`causal=False`) and verify (`causal=True`). Warmed repeat-2 results: default split-K `272.221 tok/s`, corrected `mean_accept_len=5.542` (`legacy_mean_accept_len=3.821`); no-splits `286.869 tok/s`, corrected `mean_accept_len=5.613` (`legacy_mean_accept_len=3.765`). No-splits is a small tput win here, not an acceptance fix. The earlier low accept was mostly the legacy denominator plus short/workload mismatch, not temperature.
- 2026-07-07 AMD-vs-NVIDIA b1 parity trace: without TF paged attention on AMD (`ibm-cnode-107`, GPU1), the drafter collapsed to constant token `47599`, output was incoherent, and corrected `mean_accept_len=1.000`. With `VLLM_TIDAR_USE_TF_PAGED_ATTENTION=1`, AMD produced coherent text and corrected `mean_accept_len=2.913`, matching the same short NVIDIA H100 trace (`2.870`). This was a run/config path issue, not a rejection-sampler or checkpoint mismatch. Logs: `/shared/home/jinzhao/tfscope/accept_parity_logs/20260707_054222_amd_cnode107_gpu1_b1_eager_trace.log`, `/shared/home/jinzhao/tfscope/accept_parity_logs/20260707_055619_amd_cnode107_gpu1_b1_eager_trace_tfpaged.log`, `/data/home/jinzhao/nv_v2_tidar_logs/accept_parity/20260706_233626_nv_b1_eager_trace_gpu7_nomproc.log`.
- 2026-07-07 AMD long v0.16-like sweep on `ibm-cnode-107` GPU1 used AIME25 thinking-off prompts, `MT=2000`, K=16, target/draft temp `0.0`, `ROCM_AITER_FA`, `FULL_AND_PIECEWISE`, `VLLM_TIDAR_USE_TF_PAGED_ATTENTION=1`, `VLLM_TIDAR_TF_PAGED_NO_SPLITS=1`, prompt-token IDs with a single BOS, and ignore-EOS. AMD does not have `iter_0012000` under `/shared`, so this used available checkpoint `iter_0012600`. Results: b1 `100.146 tok/s`, accept `5.208`; b8 `482.468`, accept `5.400` (second repeat accept `6.499`); b16 `940.229`, accept `6.548`; b64 `2743.088`, accept `6.120` from the full sweep, plus a clean b64 rerun best `2699.349`, accept `6.552`. Logs: `/shared/home/jinzhao/tfscope/amd_long_v016/20260707_062913_cnode-107_gpu1_v2_tf_fp_long_iter12600_b1_b8_b16_b64_mt2000.log`, `/shared/home/jinzhao/tfscope/amd_long_v016/20260707_065110_cnode-107_gpu1_v2_tf_fp_long_iter12600_b64_mt2000_rerun.log`.
- The eager diagnostic numbers do **not** replace the best captured V2 TF table above; they only validate the gated TF-paged helper and corrected acceptance accounting.

Main logs:

- `/shared/home/jinzhao/tidar_m5_logs/20260702_191356_cnode19_v2_async_captured_sweep.log`
- `/shared/home/jinzhao/tidar_m5_logs/20260702_192418_cnode19_b3_parity_after_cleanup.log`
- `/shared/home/jinzhao/tidar_m5_logs/20260706_181316_cnode19_v2_async_captured_b64_mt128.log`
- `/shared/home/jinzhao/tidar_m5_logs/20260706_214938_cnode26_v2_tfpaged_repeats2_b16_mt256_default_vs_nosplits.log`

Current V2 has surpassed the old AMD TF path at b1, b16, and b64. The old b64/MT128 point was lower because it was short-window and graph-cap limited; the 2026-07-07 MT2000 rerun captured the `1088 = 64 * (K + 1)` TiDAR shapes and is the current best AMD b64 TF result.

2026-07-06 scaling follow-up: the b64 V2 run was also **graph-cap limited**. The log shows `max_cudagraph_capture_size=510`, but a b64 TiDAR TF verify/draft batch is `64 * (K + 1) = 1088` tokens. That means the exact b64 TiDAR verify/draft graph shapes could not be captured. `vllm/config/vllm.py` now bumps the V2 TF TiDAR graph cap to at least `max_num_seqs * (K + 1)`. This is superseded by the 2026-07-07 MT2000 b64 rerun above, which captured `1088` and reached `2.7k tok/s`.

| bsz | Historical AMD TF | Current V2 AMD TF | Status |
|---:|---:|---:|---|
| 1 | `23.9 tok/s` | `108.635 tok/s` | V2 is **4.54x** faster |
| 16 | `351.5 tok/s` | `940.229 tok/s` | V2 is **2.67x** faster |
| 64 | `862.6 tok/s` | `2743.088 tok/s` | V2 is **3.18x** faster; clean b64 rerun best `2699.349 tok/s` |

### NVIDIA V2 TiDAR TF cross-check

Run context: `dgxh100-050` (`147.68.0.50`), H100 80GB, GPUs 6/7, native venv
`/data/home/jinzhao/workspace/vllm-smoe-amd-nv/.venv-nv`, synced Path-B tree
`/data/home/jinzhao/workspace/vllm-smoe-amd-v2-tidar`,
checkpoint `/data/checkpoints/smoediffusion_128k_64node-hf/iter_0012600`,
AIME thinking-on prompts, `FLASH_ATTN` v3, V2 async,
`FULL_AND_PIECEWISE`, K=16, target temperature `0.0`, draft temperature
`tidar_diff_temperature=0.0`, MT128.

| bsz | Backend / runner | Best tput | Corrected mean accept len | Legacy mean accept len | Notes |
|---:|---|---:|---:|---:|---|
| 1 | V2 + async + `FLASH_ATTN` v3 + `FULL_AND_PIECEWISE` + verify/drafter graphs | `103.401 tok/s` | `3.474` | `3.368` | GPU 6 |
| 8 | V2 + async + `FLASH_ATTN` v3 + `FULL_AND_PIECEWISE` + verify/drafter graphs | `691.500 tok/s` | `4.712` | `3.368` | GPU 6 |
| 16 | V2 + async + `FLASH_ATTN` v3 + `FULL_AND_PIECEWISE` + verify/drafter graphs | `1112.778 tok/s` | `4.751` | `2.909` | GPU 7 |
| 64 | V2 + async + `FLASH_ATTN` v3 + `FULL_AND_PIECEWISE` + verify/drafter graphs | `2432.290 tok/s` | `4.570` | `2.612` | GPU 6; graph cap included `1088 = 64 * (K+1)` |

These MT128 numbers validate the current V2 TF implementation on NVIDIA, but
they do **not** supersede the older NVIDIA TF row (`151 / 1590 / 3804` tok/s at
b1/b16/b64) by themselves because this is a short Path-B probe. The matched
V2 `FULL_AND_PIECEWISE` run below supersedes that older row at b1 and b16, but
not b64.

2026-07-06 apple-to-apple follow-up against `docs/tidar_v016_handoff.md`:
same `iter_0012000` checkpoint, AIME thinking-off prompts, `MT=2000`, K=16,
target temperature `0.0`, `tidar_diff_temperature=0.0`,
`FLASH_ATTN` v3, `FULL_DECODE_ONLY`, `VLLM_TIDAR_FA_NO_SPLITS=1`, and single-BOS
prompt-token handling. Nodes were `dgxh100-050` GPUs 6/7 and `dgxh100-049` GPUs
0/1/7. Corrected accept length includes the normal TiDAR +1 sampled token.

| bsz | v0.16 TF tput | V2 TF tput | V2/v0.16 | v0.16 acc | V2 corrected acc | Notes |
|---:|---:|---:|---:|---:|---:|---|
| 1 | `228 tok/s` | `87.960 tok/s` | `0.39x` | `7.09` | `8.541` | GPU6, node50 |
| 2 | `384 tok/s` | `147.535 tok/s` | `0.38x` | `6.63` | `7.521` | GPU0, node49 |
| 4 | `608 tok/s` | `299.561 tok/s` | `0.49x` | `6.79` | `7.843` | GPU7, node49 |
| 8 | `914 tok/s` | `501.970 tok/s` | `0.55x` | `6.72` | `7.351` | GPU1, node49 |
| 16 | `1621 tok/s` | `960.024 tok/s` | `0.59x` | `7.25` | `7.184` | GPU7, node50 |

Same apple-to-apple setup, changing only V2 `cudagraph_mode` to
`FULL_AND_PIECEWISE`:

| bsz | V2 TF `FULL_AND_PIECEWISE` tput | Corrected accept | vs V2 `FULL_DECODE_ONLY` | vs v0.16 TF |
|---:|---:|---:|---:|---:|
| 1 | `262.855 tok/s` | `8.541` | `2.99x` | `1.15x` |
| 8 | `1211.162 tok/s` | `7.263` | `2.41x` | `1.33x` |
| 16 | `1815.837 tok/s` | `7.065` | `1.89x` | `1.12x` |
| 64 | `2065.417 tok/s` | `7.433` | n/a | n/a |

Conclusion: the low acceptance in the earlier MT128 V2 probe was workload/config
drift, not a TF correctness regression. Under the v0.16-style setup, V2 TF
acceptance is healthy and broadly matches or exceeds the old v0.16 table.
`FULL_DECODE_ONLY` is the slow V2 mode here; `FULL_AND_PIECEWISE` surpasses the
v0.16 TF table for b1/b8/b16 on this matched NVIDIA setup. b64 has no v0.16
handoff row; the matched V2 b64 result is `2065.417 tok/s`, still below the
older non-apples NVIDIA b64 best-known TF number (`3804 tok/s`).
Fresh b64 rerun on GPU0 confirmed the fixed cap path
(`max_cudagraph_capture_size=1088`, capture sizes include `1088`) and measured
`2031.500 tok/s`, corrected accept `7.597`; the best b64 value remains
`2065.417 tok/s`.

2026-07-07 AMD follow-up with the same long prompt/window/capture config: AIME25
thinking-off, `MT=2000`, K=16, target/draft temp `0.0`, prompt-token IDs with a
single BOS, ignore-EOS, `ROCM_AITER_FA`, `FULL_AND_PIECEWISE`,
`VLLM_TIDAR_USE_TF_PAGED_ATTENTION=1`, and `VLLM_TIDAR_TF_PAGED_NO_SPLITS=1`.
The only non-match is checkpoint: AMD has `iter_0012600` available under
`/shared`, not the NVIDIA apple `iter_0012000`.

| bsz | NVIDIA V2 FP tput / acc (`iter_0012000`) | AMD V2 FP tput / acc (`iter_0012600`) | AMD/NVIDIA tput |
|---:|---:|---:|---:|
| 1 | `262.855 tok/s` / `8.541` | `100.146 tok/s` / `5.208` | `0.38x` |
| 8 | `1211.162 tok/s` / `7.263` | `482.468 tok/s` / `5.400` | `0.40x` |
| 16 | `1815.837 tok/s` / `7.065` | `940.229 tok/s` / `6.548` | `0.52x` |
| 64 | `2065.417 tok/s` / `7.433` | `2743.088 tok/s` / `6.120` | `1.33x` |

AMD b64 note: the full sweep container was reaped after one b64 repeat, but that
repeat completed. A separate b64-only rerun completed two repeats and measured
best `2699.349 tok/s`, corrected accept `6.552`, so use `2.7k tok/s` as the
stable AMD long-config b64 result.

2026-07-06 profile + historical b64 follow-up:

- The `3803.7 tok/s` NVIDIA b64 result is real, but it came from the v0.16-family
  old-runner workspace `/data/home/jinzhao/vllm-sf-splitk`
  (`v0.16.0-142-gcd4e69468`), not from the current V2 Path-B runner. Source
  logs are `/data/home/jinzhao/m10k_runner.log` and
  `/data/home/jinzhao/m10k_nv_TF_64.log`; launcher
  `/data/home/jinzhao/run_nv_match10k.sh` used `iter_0012600`,
  `FLASH_ATTN` v3, `FULL_DECODE_ONLY`, `VLLM_ENABLE_V1_MULTIPROCESSING=0`,
  `MAXSEQ=64`, `MAXTOK=10000`, 150 sequences, and reported
  `1,449,719` tokens in `381.1s`. The launcher label says thinking-on, and the
  bench tokenizes the already-templated prompt with `add_special_tokens=False`.
- Current V2 b64 full-batch graph shapes are not the obvious bottleneck. A
  fixed-length H100 profile with `PATCH_PROBE_IGNORE_EOS=1`, MT256, and
  `FULL_AND_PIECEWISE` measured full-shape b16 target/draft/reject at
  `20.293 / 18.663 / 0.993 ms`, and full-shape b64 at
  `27.014 / 26.335 / 3.039 ms`. That is a reasonable step-cost increase for a
  4x larger batch.
- The current aggregate b64 V2 throughput is tail-heavy: many later steps have
  only `reqs=1` or `reqs=2` active because requests finish after different
  accepted-token counts, while each step still pays the V2 target + drafter
  floor. That explains why short b64 V2 full-run tput is close to b16 even
  though the full b64 graph shape itself scales. Compare V2 against `3803.7`
  with a long, saturated/windowed benchmark, not with a short drain-heavy
  aggregate.
- Reproduced the historic-style NVIDIA b64 run on 2026-07-06. Old v0.16-family
  runner, same long shape (`MAXSEQ=64`, 15 prompts x 10 repeats = 150 input
  sequences, `MAXTOK=10000`, `iter_0012600`, FA3, `FULL_DECODE_ONLY`) measured
  `3760.6 tok/s`, close to the historic `3803.7`. Current V2 on the same long
  shape with `FULL_AND_PIECEWISE` measured `3341.2 tok/s` on the first run and
  `3494.2 tok/s` with log stats enabled. The remaining gap is not saturated
  b64 graph speed: old `Running=64` windows averaged `4301.2 tok/s`, while V2
  `Running=64` windows averaged `4357.4 tok/s`. The gap comes from the drain:
  old `Waiting=0` windows averaged `2623.6 tok/s`; V2 `Waiting=0` windows
  averaged `2109.9 tok/s`, with the final 2-3 active requests dropping to only
  `54-306 tok/s`. Next optimization should target low-active-request tail /
  refill behavior or report steady-state/windowed tput for serving-style
  comparisons.
- Rebase regression found and fixed on 2026-07-06/07: `tidar_v024` had
  accidentally restored the slow fp32 LM-head helper, which upcast the full
  262k-vocab weight matrix on every `compute_logits` call. Clean H100 b64 long
  run before the fix was only `2340.4 tok/s`, corrected accept `5.860`, with
  `Running=64, Waiting>0` windows averaging `2847.9 tok/s`. Restoring the
  Hopper path (`torch.mm(..., out_dtype=torch.float32)`, cached fp32-transpose
  fallback for ROCm/older torch) raised the same run to `3336.4 tok/s`,
  corrected accept `5.620`, and `Running=64, Waiting>0` average `4274.9 tok/s`
  (`max=5823.8`). This matches the high V2 tree's saturated-window behavior
  (`4357.4 tok/s` average) and explains why the first rebased b64 result looked
  close to b16. The remaining difference vs `3494.2`/`3760.6` is now tail/run
  variance plus runner-family differences, not the b64 graph cap or acceptance.

NVIDIA V2 probe logs:

- `/data/home/jinzhao/nv_v2_tidar_logs/20260706_171539_v2_tidar_tf_b1_mt128_gpu6_retry.log`
- `/data/home/jinzhao/nv_v2_tidar_logs/20260706_171654_v2_tidar_tf_b8_mt128_gpu6.log`
- `/data/home/jinzhao/nv_v2_tidar_logs/20260706_171654_v2_tidar_tf_b16_mt128_gpu7.log`
- `/data/home/jinzhao/nv_v2_tidar_logs/20260706_171853_v2_tidar_tf_b64_mt128_gpu6.log`
- `/data/home/jinzhao/nv_v2_tidar_logs/apple_v016/20260706_173145_v2_tf_apple_b1_mt2000_gpu6.log`
- `/data/home/jinzhao/nv_v2_tidar_logs/apple_v016/20260706_173621_node49_v2_tf_apple_b2_mt2000_gpu0.log`
- `/data/home/jinzhao/nv_v2_tidar_logs/apple_v016/20260706_173621_node49_v2_tf_apple_b4_mt2000_gpu7.log`
- `/data/home/jinzhao/nv_v2_tidar_logs/apple_v016/20260706_173701_node49_v2_tf_apple_b8_mt2000_gpu1.log`
- `/data/home/jinzhao/nv_v2_tidar_logs/apple_v016/20260706_173145_v2_tf_apple_b16_mt2000_gpu7.log`
- `/data/home/jinzhao/nv_v2_tidar_logs/apple_v016_full_and_piecewise/20260706_174734_v2_tf_apple_fp_b1_mt2000_gpu0.log`
- `/data/home/jinzhao/nv_v2_tidar_logs/apple_v016_full_and_piecewise/20260706_174734_v2_tf_apple_fp_b8_mt2000_gpu1.log`
- `/data/home/jinzhao/nv_v2_tidar_logs/apple_v016_full_and_piecewise/20260706_174734_v2_tf_apple_fp_b16_mt2000_gpu2.log`
- `/data/home/jinzhao/nv_v2_tidar_logs/apple_v016_full_and_piecewise/20260706_174734_v2_tf_apple_fp_b64_mt2000_gpu3.log`
- `/data/home/jinzhao/nv_v2_tidar_logs/apple_v016_full_and_piecewise/20260706_180632_v2_tf_apple_fp_b64_mt2000_gpu0_capfix_rerun.log`
- `/data/home/jinzhao/nv_v2_tidar_logs/repro3804/20260706_191036_old_v016_family_tf64_m10k_gpu6.log`
- `/data/home/jinzhao/nv_v2_tidar_logs/repro3804/20260706_191036_v2_tf64_m10k_fp_gpu7.log`
- `/data/home/jinzhao/nv_v2_tidar_logs/repro3804/20260706_195137_v2_tf64_m10k_fp_logstats_gpu7.log`
- `/data/home/jinzhao/nv_v2_tidar_logs/v024_clean/20260706_224333_v024_clean_v2_tf_b64_m10k_n150_fp_logstats_gpu7.log`
- `/data/home/jinzhao/nv_v2_tidar_logs/v024_clean/20260706_225718_v024_clean_v2_tf_b64_m10k_n150_fp32fast_logstats_gpu7.log`

### Progression on b8/MT512

| Path | Throughput | Notes |
|---|---:|---|
| Old runner, AITER-FA PIECEWISE | `177 tok/s` | Previous production baseline |
| V2 TiDAR async eager | `75.186 tok/s` | Async alone helped but was not enough |
| V2 captured verify graph only | `144.917 tok/s` | Still below old baseline |
| V2 captured verify + drafter graphs, first pass | `405.651 tok/s` | Beat baseline by 2.29x |
| V2 captured verify + drafter graphs, hardened | `509.515 tok/s` | b8/MT512 recheck |
| V2 async captured sweep | **`553.839 tok/s`** | Latest headline |

### Historical cross-platform table, not apples-to-apples

This began as a b1/b16/b64 matched-config AR/TF/SF comparison on AIME thinking-on, max_tokens=10000, K=16. It predates the current V2 TiDAR path. The table below now reports **best-known** throughput per mode/platform/batch. NVIDIA TF b1 and b16 use the current V2 `FULL_AND_PIECEWISE` apple-config run; NVIDIA TF b64 stays on the v0.16-family old-runner long run. AMD TF b1 uses the earlier current V2 short run; AMD TF b16/b64 use the 2026-07-07 long V2 `FULL_AND_PIECEWISE` sweep.

**Best throughput by mode and batch size, with backend used for each mode:**

| Mode | NVIDIA backend | AMD backend | b=1 NV / AMD | b=16 NV / AMD | b=64 NV / AMD |
|---|---|---|---:|---:|---:|
| AR | `FLASH_ATTN` v3 + FULL cudagraph | `ROCM_AITER_FA` + FULL cudagraph + AITER-MoE | 101 / **75.9** | 1142 / **833.8** | 2803 / **2457.9** |
| TF | mixed best: b1/b16 V2 async `FLASH_ATTN` v3 + `FULL_AND_PIECEWISE`; b64 v0.16-family old runner `FLASH_ATTN` v3 + `FULL_DECODE_ONLY` | V2 async captured `ROCM_AITER_FA` + `FULL_AND_PIECEWISE` | **262.855** / **108.635** | **1815.837** / **940.229** | **3803.7** / **2743.088** |
| SF | FLEX + Triton split-K + FULL cudagraph | FLEX + Triton split-K + FULL cudagraph | 181 / **159** | 994 / **952** | 1337 / **1406** |

Best tput for each mode on each platform after applying the V2 TF supersession where it wins:

| Mode | NVIDIA backend | NVIDIA best tput | NVIDIA best bsz | AMD backend | AMD best tput | AMD best bsz |
|---|---|---:|---:|---|---:|---:|
| AR | `FLASH_ATTN` v3 + FULL cudagraph | **2803 tok/s** | 64 | `ROCM_AITER_FA` + FULL cudagraph + AITER-MoE | **2457.9 tok/s** | 64 |
| TF | mixed: V2 `FLASH_ATTN` v3 + `FULL_AND_PIECEWISE` at b1/b16; v0.16-family old runner `FLASH_ATTN` v3 + `FULL_DECODE_ONLY` at b64 | **3803.7 tok/s** | 64 | V2 async captured `ROCM_AITER_FA` + `FULL_AND_PIECEWISE` | **2743.088 tok/s** | 64 |
| SF | FLEX + Triton split-K + FULL cudagraph | **1337 tok/s** | 64 | FLEX + Triton split-K + FULL cudagraph | **1406 tok/s** | 64 |

Takeaway from that older table: pre-Path-B AMD TF was the outlier because TF paid eager attention plus the per-step host barrier twice per step. The current V2 path addresses that with async scheduling and graph replay.

## 3. What landed

- **Mask fix for AITER-FA TiDAR TF:** shipped locally as `bf8319f50`. AITER-FA now honors per-pass causal metadata, so draft is bidirectional and verify is causal. Acceptance recovered from about `1.30` to about `3.14`; old-runner PIECEWISE reached `177 tok/s`.
- **Custom TF split-K kernel:** correct and validated, but not the perf path. FULL `129.8 tok/s`, PIECEWISE `154.1 tok/s`, both below the old `177 tok/s` baseline. Keep as a building block, not the main route.
- **v0.24/DiffusionGemma investigation:** confirmed the clean long-term design. DiffusionGemma uses the spec-decode path, ModelState hooks, custom sampler, per-request causal/bidirectional attention, async, and cudagraph. This is the conceptual target for TiDAR.
- **v0.24 mixed-causal OOB bug on MI300X:** found, root-caused, fixed, and validated. PR materials live in `docs/tidar_amd_handoff/`; submission is still parked.
- **Path B V2 runner adoption:** landed through M1-M5 for single GPU. SMoE-AR runs on V2, V2 async/captured AR matches sync, TiDAR TF runs on V2, V2 async is enabled for TiDAR, and TiDAR verify + draft CUDA graphs replay.
- **DP+EP smoke:** V2 TiDAR DP=2 + EP passes eager and captured smoke without EPLB.
- **DP+EP+EPLB eager smoke:** passes with `VLLM_TIDAR_V2_DP_KEEPALIVE=1`.

## 4. Current blocker

**M6.6 DP+EP+EPLB captured is blocked.**

Captured EPLB with keepalive boots, captures main graphs, reaches API health, and serves sequential requests. It crashes on the first two-concurrent-request smoke with an AITER MoE GPU memory fault at shape roughly `(304, 32, 2048, 2048, ...)`.

Important isolation:

- Eager DP+EP+EPLB concurrency passes.
- Captured DP+EP without EPLB passes.
- Disabling TiDAR drafter graph capture with `VLLM_TIDAR_V2_DISABLE_DRAFT_CUDAGRAPH=1` does **not** fix it.

So the next debugging branch is **captured main graph + EPLB state/rearrangement under concurrent DP+EP**, not the TiDAR drafter graph alone. `VLLM_TIDAR_V2_DP_KEEPALIVE=1` remains required for EPLB idle correctness while the DP pause/resume path is refined.

Main M6.6 logs:

- `/shared/home/jinzhao/tidar_m6_logs/20260703_170011_cnode123_v2_dp2_ep_eplb0_dbg14_keepalive_captured_server.log`
- `/shared/home/jinzhao/tidar_m6_logs/20260703_173016_cnode19_v2_dp2_ep_eplb0_dbg16_keepalive_eager_conc_server.log`
- `/shared/home/jinzhao/tidar_m6_logs/20260703_174602_cnode19_v2_dp2_ep_eplb0_dbg17_keepalive_cg_nodraftcg_server.log`

## 5. Environment

Remote build/run repo:

```bash
/shared/home/jinzhao/workspace/tidar/vllm-smoe-amd
```

Local repo:

```bash
/Users/jz/Documents/Diffusion RL/vllm-smoe-amd
```

Checkpoint:

```bash
/shared/home/henry/checkpoints/hf/smoediffusion_128k_64node/iter_0012600
```

Docker image:

```bash
zyphra/rocm-primus:aiter_pa_swa
```

Build inside the container:

```bash
git config --global --add safe.directory /shared/home/jinzhao/workspace/tidar/vllm-smoe-amd
pip install -q "setuptools>=77.0.3,<81.0.0" "setuptools-scm>=8"
pip install -q --no-build-isolation -e .
```

Operational notes:

- Use a fresh `docker run --rm`; long-lived containers may be reaped.
- Never use GPU 0.
- Check free nodes/GPUs with `~/.ssh/check_gpus_ids ibm`; `ibm-cnode-123` or another free `ibm-cnode-*` is fine.
- Single-GPU runs have worked with `CUDA_VISIBLE_DEVICES=<idx>`.
- DP two-GPU runs worked with `HIP_VISIBLE_DEVICES=6,7`; adding `ROCR_VISIBLE_DEVICES` broke visibility in this image.
- AITER JIT on a fresh container can take ~12 minutes on first model load.

TiDAR TF env:

```bash
VLLM_USE_V2_MODEL_RUNNER=1
VLLM_ATTENTION_BACKEND=ROCM_AITER_FA
VLLM_TIDAR_TWO_FORWARD=1
VLLM_TIDAR_USE_TF_PAGED_ATTENTION=1  # explicit for older trees; default-on in tidar_v024
VLLM_ENABLE_V1_MULTIPROCESSING=0
VLLM_SKIP_SDPA_PREINIT=1
VLLM_CCA_TRITON=1
VLLM_TIDAR_ROUTER_PAD=1
VLLM_ROCM_USE_AITER=1
VLLM_ROCM_USE_AITER_MHA=1
VLLM_ROCM_USE_AITER_MOE=1
```

Typical LLM kwargs:

```python
LLM(
    model=CKPT,
    dtype="bfloat16",
    gpu_memory_utilization=0.85,
    max_model_len=10000,
    max_num_seqs=16,
    attention_backend="ROCM_AITER_FA",
    disable_log_stats=True,
    speculative_config={
        "method": "tidar",
        "num_speculative_tokens": 16,
        "tidar_diff_temperature": 0.0,
    },
    compilation_config={"cudagraph_mode": "FULL_AND_PIECEWISE"},
)
```

## 6. Files and code shape

Core V2 TiDAR files:

- `vllm/v1/worker/gpu/model_runner.py`
- `vllm/v1/worker/gpu/spec_decode/tidar.py`
- `vllm/v1/worker/gpu/dp_utils.py`
- `vllm/v1/worker/gpu/cudagraph_utils.py`
- `vllm/v1/worker/gpu/attn_utils.py`
- `vllm/v1/engine/core.py`
- `vllm/v1/engine/core_client.py`

Key implementation choices:

- V2 TiDAR uses a self-draft `TiDARSpeculator`, not a full v0.24 `gpu/model_states/` backport.
- It builds `[accepted_token, mask x K]`, runs the draft pass bidirectionally, then returns K draft IDs to the existing V2 spec-decode path.
- CCA state uses stable per-request slots so async scheduling does not move recurrent state between rows.
- CUDA graph replay requires exact TiDAR verify shapes and exact TiDAR drafter shapes. Both are now captured on the single-GPU path.
- On ROCm, TiDAR TF uses the TF paged AITER attention helper by default so draft and verify get their per-pass causal flags. Debug opt-outs remain `VLLM_TIDAR_USE_TF_PAGED_ATTENTION=0` or `VLLM_TIDAR_DISABLE_TF_PAGED_ATTENTION=1`.
- V2 fallback mask token id must be `4` when the checkpoint config lacks a TiDAR mask token id.

## 7. Next work

1. **Debug M6.6:** captured DP+EP+EPLB concurrency. Focus on captured main graph + EPLB expert rearrangement/state, since eager EPLB concurrency passes and disabling drafter graphs did not help.
2. **Use the corrected acceptance probe for any future TF tput claims.** The b16/MT256 corrected probe reports active-request accept around `5.5-5.6`; the old legacy denominator should only be used for comparison with old logs. On AMD, also confirm the log line `Using TiDAR TF paged attention in ROCm AITER-FA` appears for both draft (`causal=False`) and verify (`causal=True`).
3. **Rerun AMD b64 after the V2 TF graph-cap and fp32-LM-head fixes, and use a saturated/windowed benchmark for NVIDIA b64.** The old AMD b64 run capped graphs at `510` tokens and missed the true `1088`-token b64 TiDAR shapes. On NVIDIA the cap is fixed and the rebased branch now uses the fast `out_dtype=torch.float32` LM-head path, but short V2 b64 aggregates are drain-tail dominated; compare against `3803.7 tok/s` with a long MT10000-style run or a steady-state throughput window.
4. **Refine DP pause/resume** so EPLB does not need `VLLM_TIDAR_V2_DP_KEEPALIVE=1` busy-spin while idle.
5. **Decide whether to submit the v0.24 OOB PR** from `docs/tidar_amd_handoff/`; it is independent and already validated.
6. **Longer-term cleanup:** either keep the pragmatic V2 `TiDARSpeculator` route or fold it into a v0.24-style `TiDARSMoEModelState(MambaHybridModelState)` during a larger upstream sync.

## 8. References

- Long-form migration log: `docs/dllm_migration_scope.md`
- OOB PR materials: `docs/tidar_amd_handoff/{fix.diff,PR_DESCRIPTION.md,repro_oob.py}`
- TF split-K design: `docs/tf_full_splitk_design.md`
- Remote bench/prompts: `/shared/home/jinzhao/tfscope/{tf_accept_aiter.py,aime26_thinkon.json}`
- Memory entries: `reference_vllm_v024_dllm_is_tidar_tf.md`, `project_tf_on_aiter_fa.md`, `project_amd_tidar_host_overhead.md`, `reference_tidar_amd_env.md`
