# TiDAR-on-AMD Handoff

_Last updated: 2026-07-13. Owner: jinzhao. Scope: TiDAR two-forward (TF) decode on MI300X for the 80B SMoE/Zaya checkpoint._

## 1. Current state

The old production TiDAR TF path on AMD was correct after the AITER-FA causal-mask fix, but it was host-overhead bound and only reached **177 tok/s** at b8/MT512. The current working-tree path is **Path B**: TiDAR TF runs on the fork's V2 GPU runner (`vllm/v1/worker/gpu/model_runner.py`, selected by `VLLM_USE_V2_MODEL_RUNNER=1`) and uses native V2 async scheduling plus cudagraph replay for both verify and draft forwards.

**Production-comparable headline:** **553.839 tok/s** on MI300X at b8/MT512, `ROCM_AITER_FA`, V2 async, `FULL_AND_PIECEWISE`, K=16. This is **3.13x** over the old 177 tok/s production baseline.

Path B is not "stock vLLM 0.24"; it is a v0.24-style adoption inside the v0.16-based fork. v0.24/DiffusionGemma helped by showing the clean architecture: V2 runner + spec-decode data path + model-state-style per-request state + async/cudagraph-friendly execution.

2026-07-13 batch-shape diagnostic update: raw b1/b64 layer hashes identify
ROCm unquantized dense GEMMs as the first acceptance-trajectory divergence and
CCA convolution as the second; AITER MoE is not causal. A new opt-in
fixed-reduction CCA path (`VLLM_CCA_BATCH_INVARIANT_CONV=1`) is exact across
b1/b64 and cudagraph replay. In a matched MI300X full-model A/B it lowers the
b64 step from `65.25` to `60.20 ms` and raises steady TF from `5139.7` to
`5571.2 tok/s` (`+8.4%`) with unchanged acceptance and final token hash. Making
all dense projections fixed-reduction as well gives identical outputs at all
80 layers and an identical complete acceptance trajectory, but the diagnostic
dense kernel is too slow. Default/forced hipBLASLt, rocBLAS, and disabled
rocBLAS atomics produce the same isolated dense mismatch; the remaining
numerical need is a fast fixed-reduction BF16 GEMM. Reproducers and full tables
are in `docs/amd_nvidia_tidar_tput_tests.md`.

2026-07-07 AMD/NVIDIA parity update: low AMD TF acceptance was traced to runs that missed the TiDAR TF paged AITER attention path and fell back to generic ROCm AITER extend attention. The branch now defaults that route on for TiDAR TF; older trees should set `VLLM_TIDAR_USE_TF_PAGED_ATTENTION=1` explicitly. A same-checkpoint `iter_0012600` long sweep now shows AMD acceptance is roughly on par with NVIDIA, while AMD throughput is lower because the target/draft forwards are slower on `ROCM_AITER_FA`.

2026-07-07 checkpoint-control update: the exact current NVIDIA V2 TF long script run back-to-back shows `iter_0012000` is materially more TiDAR-accept-friendly than `iter_0012600` (b1/b8/b16/b64 accept `6.34/6.62/6.48/6.56` vs `4.98/5.55/5.63/5.80`). This explains most of the apparent regression against the older v0.16 handoff accept table; it is checkpoint/workload drift, not evidence of a V2 rejection-sampler bug.

2026-07-09 measurement policy update: use **`iter_0012600`** for new
collaborator-facing AMD/NVIDIA throughput measurements because that is the
checkpoint available to the AMD team. Pass prompt token IDs with exactly one
forced BOS (`--prompt-token-ids --force-bos`) and verify
`leading4=[2,105,9731,107]`, `leading_bos_count=1`. Keep `iter_0012000` only
for historical v0.16/checkpoint-drift comparisons. The focused reference table
and repo-local AR/TF reproducer are in `docs/amd_nvidia_tidar_tput_tests.md`.

2026-07-09 AMD MT10000 update: the matched `iter_0012000` AMD rerun completed
on `cnode-83` with `ROCM_AITER_FA`, V2 async, `FULL_AND_PIECEWISE`, EOS
allowed, prompt-token IDs with single BOS, target/draft temp `0.0`, and K=16.
The b64 rerun confirmed the full TiDAR graph cap fix (`1088 = 64 * (K + 1)`)
and used the TF paged AITER path for both verify and draft. Acceptance is
healthy and broadly NVIDIA-comparable, but AMD TF is slower than AMD AR at
b8/b16/b64 under this historical `iter_0012000` long workload. Keep it only as
checkpoint/workload history; the collaborator-facing `iter_0012600` table is
in `docs/amd_nvidia_tidar_tput_tests.md`.

2026-07-09 BF16 LM-head update: a paired b16 profile showed no throughput win.
FP32 `out_dtype` reached `843.313 tok/s`, accept `4.981`; BF16 reached
`836.250 tok/s`, accept `4.672`. Torch 2.10/ROCm 7.2 already supports the fast
BF16 x BF16 -> FP32 `torch.mm` path, so the suspected explicit-FP32 fallback
was not active. BF16 remains opt-in via
`VLLM_SMOE_ROCM_BF16_LM_HEAD=1`; default stays FP32. Full-shape LM-head calls
were below 1 ms versus roughly `29.8 / 27.3 ms` target/draft model forwards.
Focus next on the generic untuned AITER MoE path, then the CCA uniform K+1
path. This was an `iter_0012000` diagnostic; use the `iter_0012600` profile in
`docs/amd_nvidia_tidar_tput_tests.md` for collaborator-facing comparisons.

2026-07-09 AMD kernel-screen update: a matched, unprofiled b16/MT1024 screen on
`cnode-163` found no backend/configuration win over the current padded AITER
MoE plus cuDNN CCA path (`949.805 tok/s`, accept `5.117`). Triton MoE reached
`846.252`, alternate AITER SMoE-op `911.650`, AITER without router padding
`909.478`, CCA Triton fusion `858.596`, and CCA unfold/einsum `885.883 tok/s`.
Keep all of those alternatives off. The next useful kernel work is a tuned
AITER unquantized-MoE configuration for the TiDAR shapes or a TiDAR-aware HIP
CCA uniform-K+1 kernel that preserves candidate-state stashing and separate
draft writes.

2026-07-12 steady-state measurement update: fixed full-batch H100 engine
iterations, timed from target-forward start through sampling, CCA commit, and
draft completion, reach `209/1091/2184/5270 tok/s` at b1/b8/b16/b64. These are
`2.53x/1.96x/1.94x/1.59x` over matched AR. The corresponding finite-batch
aggregate TF rates are `198/883/1594/2670 tok/s`; request-completion skew and
tail drain therefore account for about half of the b64 loss. Host-clock and
CUDA-event rates agree within `0.1%`, so the residual full-batch scaling loss is
device-side. See `docs/amd_nvidia_tidar_tput_tests.md` for method, table, and
logs. Future serving comparisons should use fixed full-batch iterations or
continuous refill, with finite-batch aggregate completion reported separately.

2026-07-12 cross-platform no-drain update: a replicated-prompt, fixed-iteration
run keeps every request in lockstep and excludes prefill, startup, tail drain,
and the final partial TF iteration. At b1/b8/b16/b64, H100 AR is
`86/647/1298/4817 tok/s` and TF is `240/1813/3362/9247 tok/s`
(`2.79/2.80/2.59/1.92x`). MI300X AR is `56/455/815/3483 tok/s` and TF is
`170/1144/2556/6067 tok/s` (`3.01/2.52/3.14/1.74x`). Thus AMD TF is healthy
and faster than AR at every batch when drain is removed; the earlier
finite-batch slowdown is mostly drain plus sampled-trajectory variation. At
b64, AMD TF's `67.409 ms` device step is `1.30x` H100's `52.005 ms`, comparable
to AMD AR's `1.38x` step-latency gap. Full config, acceptance, logs, and the
Slurm reproducer are in `docs/amd_nvidia_tidar_tput_tests.md`.

2026-07-12 b64 bottleneck update: matched b32/b64 event profiles on H100 and
MI300X attribute about 70% of TF-step growth to the two `M=17B` backbone
graphs and about 20% to target sampling. H100 grows `40.327 -> 51.820 ms` and
MI300X grows `53.192 -> 67.092 ms`; prefix rejection remains below `0.011 ms`.
A graph-node H100 Nsight trace finds FusedMoE at `16.3%`, named CCA
convolution/layout kernels at about `16%` (above `20%` with adjacent
copies/concatenations), the two vocabulary GEMMs at `6.1%`, temperature plus
Gumbel at `6.4%`, and attention at about `1.9%`. The b64 TF/AR ratio falls
because AR remains at `M=B` and its step latency is nearly flat, while TF runs
two `M=17B` passes whose activation work grows with batch. On AMD, nearly all
of the extra b64 gap is target/draft model time; Gumbel is faster than H100.
Prioritize exact-shape low-row AITER MoE tuning, ROCm CCA fusion, and the
FP32-output vocabulary GEMMs. Full tables and logs are in the focused
throughput document.

The same no-drain setup at b32 reaches H100 AR/TF
`2479/6202 tok/s` (`2.50x`, accept `7.514`) and MI300X AR/TF
`1580/3579 tok/s` (`2.26x`, accept `5.885`). The b128 V2 TF memory-access fault
was fixed in `gpu/block_table.py`: gather and slot-mapping now pass each cache
group's typed tensor directly to Triton instead of dereferencing addresses from
a device-side `uint64` pointer array. H100 b128 reaches `8379 tok/s`,
`82.427 ms`, accept `5.396` (`0.98x` its `8512` AR); MI300X reaches
`6385 tok/s`, `108.640 ms`, accept `5.420` (`1.12x` its `5688` AR). The nearly
matched acceptance makes this a clean device comparison: AMD/H100 TF is
`0.76x`, and the AMD step is `1.32x` slower, close to the b64 `1.30x` gap. A
b64 H100 regression is within `0.2%` of the old path.

2026-07-12 H100 saturation update: a fixed-full-batch, identical-prompt sweep
holds acceptance at `6.4524` and maps TF batch to model GEMM rows with
`M=17B`. Steady TF reaches `7.70k/9.28k/10.97k/11.56k/12.14k/12.60k/12.99k`
tok/s at b64/b96/b160/b192/b224/b320/b448. The practical knee is b224-b320:
b224 is `93.5%` and b320 is `97.0%` of the b448 ceiling. This is end-to-end
throughput saturation, not the exact mathematical AI asymptote; a representative
2048-square BF16 GEMM is at `84%` of asymptotic AI at b320 and `88%` at b448,
while top-1 MoE expert GEMMs see approximately `M/16`. The shared block-table
pointer fix now validates b128; b256 has not been rerun with the fix. b544
reaches replay but hits a separate Triton illegal-memory-access. Full details
and logs are in `docs/amd_nvidia_tidar_tput_tests.md`.

2026-07-12 H100 optimization update: two exact working-tree changes improve
the synchronized steady-state ceiling without changing token hashes or
acceptance. The simple V2 sampler path reuses SMoE's existing FP32 logits
instead of allocating/copying another FP32 tensor; CCA caches versioned FP32
convolution parameters rather than converting them in every layer and pass.
Together they raise b64 from `7695` to `8025 tok/s` (`+4.3%`) and b320 from
`12600` to `13018 tok/s` (`+3.3%`). Nsight shows the remaining b320 cost is
CCA copies/layout/convolution (`>35%`), vocabulary GEMMs (`20.3%`), and exact
temperature/Gumbel sampling (`22.4%`); fused MoE is only `2.2%` and attention
about `0.2%`. FP32 Gumbel and CCA Triton fusion were faster but changed
acceptance and remain off by default. See the focused throughput document for
the full A/B table, rejected experiments, and logs.

## 2. TF throughput numbers

### Current V2 TiDAR TF, single GPU, async captured

Short-run headline context: `ibm-cnode-19`, GPU 3, `FULL_AND_PIECEWISE`, `ROCM_AITER_FA`, K=16, AIME thinking-on prompts, `num_speculative_tokens=16`, no EPLB. Long MT2000 rows are marked inline and use the later `ibm-cnode-107` `iter_0012600` runs.

**Best measured V2 TF throughput by batch size:**

| bsz | Backend / runner | Best tput | Mean accept len | Run shape | Notes |
|---:|---|---:|---:|---|---|
| 1 | V2 + async + `ROCM_AITER_FA` + `FULL_AND_PIECEWISE` + verify/drafter graphs | `108.635 tok/s` | `4.000` | MT128, offset 0, seed 0 | Current V2 |
| 3 | V2 + async + `ROCM_AITER_FA` + `FULL_AND_PIECEWISE` + verify/drafter graphs | `298.601 tok/s` | `4.000` | MT256, offset 12, seed 1 | Best of two b3 cases |
| 8 | V2 + async + `ROCM_AITER_FA` + `FULL_AND_PIECEWISE` + verify/drafter graphs | **`553.839 tok/s`** | `3.103` | MT512, offset 0, seed 0 | Current headline |
| 16 | V2 + async + `ROCM_AITER_FA` + `FULL_AND_PIECEWISE` + verify/drafter graphs | `990.631 tok/s` | `5.986` | MT2000, AIME25 thinking-off, `iter_0012600`, `n_sample=10`, seed 0 | Same-checkpoint AMD/NVIDIA matched sweep |
| 64 | V2 + async + `ROCM_AITER_FA` + `FULL_AND_PIECEWISE` + verify/drafter graphs | `2743.088 tok/s` | `6.120` | MT2000, AIME25 thinking-off, `iter_0012600`, seed 0 | Absolute AMD best from earlier nonmatched `n_sample=1` sweep. Same-checkpoint `n_sample=10` parity run: `1682.274 tok/s`, accept `5.886` |

Other measured V2 TF cases:

| Case | Throughput | Mean accept len | Notes |
|---|---:|---:|---|
| b3, MT128, offset 0, seed 0 | `224.316 tok/s` | `3.048` | Lower than b3/MT256 offset12 |
| b8, MT256, offset 8, seed 1 | `438.955 tok/s` | `2.485` | Harder prompt slice |
| b1, MT2000, AIME25 thinking-off | `100.146 tok/s` | `5.208` | Long v0.16-like config, `iter_0012600` |
| b8, MT2000, AIME25 thinking-off | `510.885 tok/s` | `5.471` | Same-checkpoint AMD/NVIDIA matched sweep, `n_sample=10`, `iter_0012600` |

Acceptance caveats from the 2026-07-06 follow-up:

- Current V2 acceptance probes use **target temperature 0.0** and **`tidar_diff_temperature=0.0`**. The V2 drafter takes argmax logits, and the V2 rejection path accepts while target argmax equals draft argmax.
- Do not compare these short V2 probes directly with the v0.16 6-7 table without matching workload. The v0.16 table used `iter_0012000`, AIME thinking-off, `MT=2000`, and long metric windows. Use `iter_0012600` plus one forced BOS for new collaborator-facing tests; keep `iter_0012000` for historical checkpoint controls.
- The old probe formula `total_tokens / (batch * propose_calls)` undercounts acceptance when requests finish early. `/shared/home/jinzhao/tidar_m5_logs/probe_v2_tidar_accept.py` now also accumulates actual V2 `num_sampled` across active requests and reports the old formula as `legacy_mean_accept_len`.
- `vllm/attention/ops/tf_attention.py` now honors `VLLM_TIDAR_TF_PAGED_NO_SPLITS=1` and the legacy `VLLM_TIDAR_FA_NO_SPLITS=1`. `vllm/v1/attention/backends/rocm_aiter_fa.py` now uses a gated TiDAR K+1 decode branch instead of the older unconditional `_dmql > 1` branch.
- Corrected b16/MT256 eager diagnostic on `ibm-head -> Slurm cnode-26`, GPU3, `iter_0012600`, AIME first16, `VLLM_TIDAR_USE_TF_PAGED_ATTENTION=1`, `PATCH_PROBE_REPEATS=2`: the log confirms `Using TiDAR TF paged attention in ROCm AITER-FA` for both draft (`causal=False`) and verify (`causal=True`). Warmed repeat-2 results: default split-K `272.221 tok/s`, corrected `mean_accept_len=5.542` (`legacy_mean_accept_len=3.821`); no-splits `286.869 tok/s`, corrected `mean_accept_len=5.613` (`legacy_mean_accept_len=3.765`). No-splits is a small tput win here, not an acceptance fix. The earlier low accept was mostly the legacy denominator plus short/workload mismatch, not temperature.
- 2026-07-07 AMD-vs-NVIDIA b1 parity trace: without TF paged attention on AMD (`ibm-cnode-107`, GPU1), the drafter collapsed to constant token `47599`, output was incoherent, and corrected `mean_accept_len=1.000`. With `VLLM_TIDAR_USE_TF_PAGED_ATTENTION=1`, AMD produced coherent text and corrected `mean_accept_len=2.913`, matching the same short NVIDIA H100 trace (`2.870`). This was a run/config path issue, not a rejection-sampler or checkpoint mismatch. Logs: `/shared/home/jinzhao/tfscope/accept_parity_logs/20260707_054222_amd_cnode107_gpu1_b1_eager_trace.log`, `/shared/home/jinzhao/tfscope/accept_parity_logs/20260707_055619_amd_cnode107_gpu1_b1_eager_trace_tfpaged.log`, `/data/home/jinzhao/nv_v2_tidar_logs/accept_parity/20260706_233626_nv_b1_eager_trace_gpu7_nomproc.log`.
- 2026-07-07 same-checkpoint AMD/NVIDIA long sweep used `iter_0012600` on both platforms, AIME25 thinking-off prompts, `MT=2000`, warmup 64, K=16, target/draft temp `0.0`, prompt-token IDs with a single BOS, ignore-EOS, `FULL_AND_PIECEWISE`, and `n_sample=10` (`num_input_sequences = bsz * 10`). Results are in the matched table below. This supersedes the earlier AMD-only `n_sample=1` sweep for cross-platform parity claims, though that earlier run remains the absolute best AMD b64 number observed so far.
- The eager diagnostic numbers do **not** replace the best captured V2 TF table above; they only validate the gated TF-paged helper and corrected acceptance accounting.

Main logs:

- `/shared/home/jinzhao/tidar_m5_logs/20260702_191356_cnode19_v2_async_captured_sweep.log`
- `/shared/home/jinzhao/tidar_m5_logs/20260702_192418_cnode19_b3_parity_after_cleanup.log`
- `/shared/home/jinzhao/tidar_m5_logs/20260706_181316_cnode19_v2_async_captured_b64_mt128.log`
- `/shared/home/jinzhao/tidar_m5_logs/20260706_214938_cnode26_v2_tfpaged_repeats2_b16_mt256_default_vs_nosplits.log`
- `/shared/home/jinzhao/tfscope/amd_iter12600_matched/20260707_075944_amd_cnode107_gpu1_iter12600_fp_nsample10_b1_warmimg.log`
- `/shared/home/jinzhao/tfscope/amd_iter12600_matched/20260707_075944_amd_cnode107_gpu3_iter12600_fp_nsample10_b8_warmimg.log`
- `/shared/home/jinzhao/tfscope/amd_iter12600_matched/20260707_075944_amd_cnode107_gpu4_iter12600_fp_nsample10_b16_warmimg.log`
- `/shared/home/jinzhao/tfscope/amd_iter12600_matched/20260707_075944_amd_cnode107_gpu5_iter12600_fp_nsample10_b64_warmimg.log`
- `/shared/home/jinzhao/tfscope/amd_iter12600_profile/20260707_081033_amd_cnode107_gpu4_iter12600_fp_profile_b16_mt512_warmimg.log`
- `/shared/home/jinzhao/tfscope/amd_ar_v2_iter12600_captured_20260707_091143/`

Current V2 has surpassed the old AMD TF path at b1, b16, and b64. The old b64/MT128 point was lower because it was short-window and graph-cap limited. The earlier 2026-07-07 AMD-only MT2000 rerun captured `1088 = 64 * (K + 1)` and reached `2.7k tok/s`, but use the same-checkpoint `n_sample=10` matched table below for AMD/NVIDIA parity.

2026-07-06 scaling follow-up: the b64 V2 run was also **graph-cap limited**. The log shows `max_cudagraph_capture_size=510`, but a b64 TiDAR TF verify/draft batch is `64 * (K + 1) = 1088` tokens. That means the exact b64 TiDAR verify/draft graph shapes could not be captured. `vllm/config/vllm.py` now bumps the V2 TF TiDAR graph cap to at least `max_num_seqs * (K + 1)`. This is superseded by the 2026-07-07 MT2000 b64 rerun above, which captured `1088` and reached `2.7k tok/s`.

| bsz | Historical AMD TF | Current V2 AMD TF | Status |
|---:|---:|---:|---|
| 1 | `23.9 tok/s` | `108.635 tok/s` | V2 is **4.54x** faster |
| 16 | `351.5 tok/s` | `990.631 tok/s` | V2 is **2.82x** faster |
| 64 | `862.6 tok/s` | `2743.088 tok/s` | V2 is **3.18x** faster as absolute best; same-checkpoint matched run is `1682.274 tok/s` (**1.95x** faster) |

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

2026-07-07 true AMD/NVIDIA same-checkpoint long sweep: both platforms used
`iter_0012600`, AIME25 thinking-off prompts, `MT=2000`, warmup 64, K=16,
target/draft temp `0.0`, prompt-token IDs with a single BOS, ignore-EOS,
`FULL_AND_PIECEWISE`, `n_sample=10`, `num_prompts=bsz`, `max_num_seqs=bsz`,
and `num_input_sequences=bsz*10`. NVIDIA used H100 `FLASH_ATTN` v3 on
`dgxh100-050`; AMD used MI300X `ROCM_AITER_FA` with TF-paged attention and
no-splits on `ibm-cnode-107` from warmed image `tidar-v024-built:iter12600-aiterwarm`.

| bsz | NVIDIA V2 FP tput / acc (`FLASH_ATTN` v3) | AMD V2 FP tput / acc (`ROCM_AITER_FA`) | AMD/NVIDIA tput |
|---:|---:|---:|---:|
| 1 | `148.951 tok/s` / `4.978` | `100.010 tok/s` / `5.208` | `0.67x` |
| 8 | `1029.468 tok/s` / `5.590` | `510.885 tok/s` / `5.471` | `0.50x` |
| 16 | `1656.976 tok/s` / `5.735` | `990.631 tok/s` / `5.986` | `0.60x` |
| 64 | `2127.865 tok/s` / `5.881` | `1682.274 tok/s` / `5.886` | `0.79x` |

2026-07-07 NVIDIA checkpoint-control run: exact same V2 TF script/config as
above, but on H100 only, paired by batch on the same GPU with `iter_0012000`
then `iter_0012600`. This isolates the checkpoint effect behind the older
v0.16 handoff's higher acceptance.

| bsz | `iter_0012000` tput / acc | `iter_0012600` tput / acc | Accept delta |
|---:|---:|---:|---:|
| 1 | `188.467 tok/s` / `6.339` | `157.188 tok/s` / `4.978` | `+1.361` |
| 8 | `1139.612 tok/s` / `6.623` | `984.326 tok/s` / `5.554` | `+1.068` |
| 16 | `1846.769 tok/s` / `6.475` | `1685.288 tok/s` / `5.628` | `+0.847` |
| 64 | `2280.917 tok/s` / `6.556` | `2098.162 tok/s` / `5.799` | `+0.757` |

Log dir: `/data/home/jinzhao/nv_v2_tidar_logs/iter12000_vs_12600_backtoback_20260707_030645/`.

2026-07-07 NVIDIA `iter_0012000` MT10000 AR/TF sweep on `vp-dgx-82`: V2 async,
`FLASH_ATTN` v3, `FULL_AND_PIECEWISE`, AIME26 thinking-on prompts,
prompt-token IDs with single BOS, EOS allowed, target/draft temp `0.0`, K=16,
`max_model_len=12000`, warmup 64, `n_sample=10`, and
`num_prompts=min(bsz, 15)`. All eight jobs were launched concurrently on GPUs
0-7, so absolute throughput is conservative versus solo runs; ratios are
apples-to-apples under the same host load. The earlier solo-ish b64 TF run on
the same node remains the best observed `iter_0012000` V2 TF b64 number:
`3728.701 tok/s`, acc `6.373`.

| bsz | AR tput | TF tput | TF corrected acc | TF/AR |
|---:|---:|---:|---:|---:|
| 1 | `82.538 tok/s` | `242.801 tok/s` | `9.101` | `2.94x` |
| 8 | `541.217 tok/s` | `965.614 tok/s` | `5.987` | `1.78x` |
| 16 | `908.467 tok/s` | `1622.849 tok/s` | `5.733` | `1.79x` |
| 64 | `2477.630 tok/s` | `3352.900 tok/s` | `5.901` | `1.35x` |

Log dir: `/data/home/jinzhao/nv_v2_tidar_logs/iter12000_m10k_sweep_20260707_174814/`.
Earlier best b64 TF log: `/data/home/jinzhao/nv_v2_tidar_logs/iter12000_m10k_vp82_20260707_155231/v2_tf64_m10k_iter12000_vp82_gpu6.log`.

2026-07-09 AMD `iter_0012000` MT10000 AR/TF sweep on `cnode-83`, GPUs 4/5:
same config as the NVIDIA MT10000 table above, with `ROCM_AITER_FA`, TiDAR TF
paged attention, and no-splits for TF. The b64 row was rerun after fixing the
full `1088` TiDAR graph cap.

| bsz | AR tput | TF tput | TF corrected acc | TF/AR | Notes |
|---:|---:|---:|---:|---:|---|
| 1 | `57.086 tok/s` | `100.812 tok/s` | `10.208` | `1.77x` | TF faster than AR at b1. |
| 8 | `413.490 tok/s` | `265.706 tok/s` | `5.770` | `0.64x` | TF slower than AR. |
| 16 | `686.762 tok/s` | `471.601 tok/s` | `5.727` | `0.69x` | TF slower than AR. |
| 64 | `2103.055 tok/s` | `1293.746 tok/s` | `6.056` | `0.62x` | Cap-fixed b64; final tail had two heavy-reject samples. |

Log dir:
`/shared/home/jinzhao/tfscope/amd_iter12000_m10k_match_nv_eos_nopip_20260709_044623/`.
Conclusion: on the matched `iter_0012000` MT10000 workload, AMD AR is within
`0.69-0.85x` of NVIDIA AR, but AMD TF is only `0.28-0.42x` of NVIDIA TF. Since
acceptance is comparable, the next AMD work should focus on reducing the cost
of the two ROCm AITER forwards and controlling long-tail/rejection-heavy drain,
not on rejection-sampler correctness.

Template/tokenizer caveat: `AutoTokenizer.chat_template` is not byte-identical
between these checkpoints. `iter_0012000`'s `chat_template.jinja` prepends
`{{ bos_token }}`, while `iter_0012600`'s does not. However, the TF checkpoint
control above did **not** call `apply_chat_template`: it used AIME25 prompts
already containing `<bos><|im_start|>...` and passed `prompt_token_ids` generated
with `add_special_tokens=False`. All 30 benchmark prompts tokenize identically
between `iter_0012000` and `iter_0012600` under that path, with first prompt
leading ids `[2, 105, 9731, 107, 106, 107, 105, 2364]`. So the measured accept
gap is not caused by chat-template application in this benchmark.

Forced-BOS retest: regenerated explicit `forcebos` datasets on both platforms
and reran `iter_0012600` V2 TF. The transform changed `0/30` prompts on both
NVIDIA and AMD; every run logged `leading4=[2,105,9731,107]` and
`leading_bos_count=1`. Acceptance stayed in the same `iter_0012600` band:

| bsz | NVIDIA H100 force-BOS tput / acc | AMD MI300X force-BOS tput / acc |
|---:|---:|---:|
| 1 | `149.475 tok/s` / `4.978` | `99.933 tok/s` / `5.208` |
| 8 | `947.597 tok/s` / `5.479` | `525.938 tok/s` / `5.622` |
| 16 | `1633.103 tok/s` / `5.642` | `975.718 tok/s` / `5.887` |
| 64 | `2168.040 tok/s` / `5.774` | `1740.261 tok/s` / `5.902` |

Logs: NVIDIA `/data/home/jinzhao/nv_v2_tidar_logs/iter12600_forcebos_20260707_120111/`; AMD `/shared/home/jinzhao/tfscope/amd_iter12600_forcebos_20260707_180130/`.

Takeaway: when the checkpoint/config are matched, AMD acceptance is not the
problem anymore; it is broadly comparable to NVIDIA, especially at b16/b64.
Throughput remains lower on AMD. The older AMD-only `2.7k tok/s` b64 result is
not the apples-to-apples parity number because that run used a different input
count/window (`n_sample=1`), although it remains a useful absolute best sighting.

2026-07-10 MT5000 apples-to-apples update: both platforms used
`iter_0012600`, forced single-BOS token IDs, ignore-EOS, V2 async,
`FULL_AND_PIECEWISE`, K=16, temperature `0.0/0.0`, and `n_sample=1` for both
AR and TF. Throughput is output tok/s; acceptance includes the normal `+1`:

| bsz | NVIDIA AR | NVIDIA TF / acc | TF/AR | AMD AR | AMD TF / acc | TF/AR |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | `79.723` | `207.554` / `7.562` | `2.60x` | `53.828` | `62.296` / `4.703` | `1.16x` |
| 8 | `539.015` | `972.872` / `6.136` | `1.80x` | `382.078` | `375.808` / `6.746` | `0.98x` |
| 16 | `968.246` | `1603.001` / `6.802` | `1.66x` | `772.428` | `661.769` / `8.325` | `0.86x` |
| 64 | `3093.504` | `3482.665` / `6.936` | `1.13x` | `2509.826` | `2102.265` / `8.250` | `0.84x` |

AMD TF reaches only `0.30/0.39/0.41/0.60x` NVIDIA TF at b1/b8/b16/b64,
while AMD AR reaches `0.68/0.71/0.80/0.81x`. At b8-b64 AMD acceptance is
higher than NVIDIA, yet TF is no faster than AMD AR. This cleanly isolates the
remaining problem to ROCm target/draft forward cost rather than rejection
quality. The b1 acceptance delta is noisy because that row contains one prompt
and the platform output trajectories diverge numerically.

Logs: NVIDIA `/data/home/jinzhao/nv_v2_tidar_logs/iter12600_forcebos_mt5k_n1_vp16_20260709_175900/`; AMD `/shared/home/jinzhao/tfscope/amd_iter12600_forcebos_mt5k_n1_c17_20260710_002037/`. The focused table and reproducer are in `docs/amd_nvidia_tidar_tput_tests.md`.

2026-07-10 matched-metadata temperature sweep: the AMD checkpoint packaging
was aligned to the `iter_0012600` NVIDIA metadata variant, which matches
`iter_0012000` on `residual_in_fp32=false`, `mamba_cache_dtype=float32`, and
the absence of the added SWA fields. All four weight-shard hashes and all 30
forced-BOS prompt-token sequences match across platforms. The run used MT4000,
target temperature `0.6`, argmax draft temperature `0`, `n_sample=1`, and the
same V2/capture/backend settings:

| bsz | NVIDIA AR | NVIDIA TF / acc | TF/AR | AMD AR | AMD TF / acc | TF/AR | AMD/NVIDIA AR | AMD/NVIDIA TF |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `82.258` | `132.427` / `4.619` | `1.61x` | `60.702` | `60.753` / `4.209` | `1.00x` | `0.74x` | `0.46x` |
| 8 | `542.912` | `680.086` / `5.317` | `1.25x` | `377.326` | `314.923` / `5.422` | `0.83x` | `0.70x` | `0.46x` |
| 16 | `975.975` | `1333.164` / `5.263` | `1.37x` | `773.367` | `587.371` / `5.444` | `0.76x` | `0.79x` | `0.44x` |
| 64 | `3141.139` | `4043.730` / `5.942` | `1.29x` | `2532.184` | `1788.693` / `5.002` | `0.71x` | `0.81x` | `0.44x` |

NVIDIA b1/b8 are three-seed medians; b16/b64
and all AMD rows remain seed 0. AMD AR is `0.74/0.70/0.79/0.81x` NVIDIA AR,
while AMD TF is `0.46/0.46/0.44/0.44x` NVIDIA TF at b1/b8/b16/b64.
Matching the metadata does not remove the AMD TF throughput
gap. Logs: NVIDIA
`/data/home/jinzhao/nv_v2_tidar_logs/iter12600_iter12000meta_mt4k_t06_d0_n1_vp16_20260709_190123/`; AMD
`/shared/home/jinzhao/tfscope/amd_iter12600_iter12000meta_mt4k_t06_d0_n1_c17_20260710_010137/`.

2026-07-07 captured AR baseline, same `iter_0012600` checkpoint and prompt
path, no speculative config, V2 async, `FULL_AND_PIECEWISE`, `n_sample=1`,
`MT=2000`, prompt-token IDs, ignore-EOS:

| bsz | NVIDIA H100 AR (`FLASH_ATTN` v3) | AMD MI300X AR (`ROCM_AITER_FA`) | AMD/NVIDIA |
|---:|---:|---:|---:|
| 1 | `82.649 tok/s` | `53.369 tok/s` | `0.65x` |
| 8 | `552.746 tok/s` | `419.094 tok/s` | `0.76x` |
| 16 | `1035.511 tok/s` | `774.441 tok/s` | `0.75x` |
| 64 | `3420.864 tok/s` | `2622.038 tok/s` | `0.77x` |

Log dirs: NVIDIA `/data/home/jinzhao/nv_v2_tidar_logs/ar_v2_iter12600_captured_20260707_034419/`; AMD `/shared/home/jinzhao/tfscope/amd_ar_v2_iter12600_captured_20260707_091143/`.

Matched b16 profile (`n_sample=1`, `MT=512`, same checkpoint/backend/capture):

| Platform | Tput / acc | Full-shape target | Full-shape draft | Full-shape reject sampler | Notes |
|---|---:|---:|---:|---:|---|
| NVIDIA H100, `FLASH_ATTN` v3 | `1394.609 tok/s` / `6.307` | `20.067 ms` | `17.579 ms` | `0.989 ms` | Log `/data/home/jinzhao/nv_v2_tidar_logs/iter12600_profile/20260707_014246_nv_gpu6_iter12600_fp_profile_b16_mt512.log` |
| AMD MI300X, `ROCM_AITER_FA` | `1197.474 tok/s` / `5.796` | `28.821 ms` | `26.314 ms` | `1.175 ms` | Log `/shared/home/jinzhao/tfscope/amd_iter12600_profile/20260707_081033_amd_cnode107_gpu4_iter12600_fp_profile_b16_mt512_warmimg.log` |

Profile read: AMD full-shape target and draft passes are about `1.4-1.5x`
slower than NVIDIA, while rejection sampling is close. The TF throughput gap is
therefore mostly backend/kernel forward latency, not rejection-sampler logic.

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
- 2026-07-08 b64 saturation/profile follow-up on `dgxh100-050` with
  `iter_0012000`, V2 async, `FLASH_ATTN` v3, `FULL_AND_PIECEWISE`, K=16,
  temp `0.0/0.0`, single-BOS prompt-token IDs, and a larger request pool
  (`num_prompts=64`, `n_sample=8`, 512 inputs) measured TF
  `3662.880 tok/s`, corrected accept `5.691`, and AR `2943.271 tok/s`, so
  TF/AR was `1.24x`. This improves over the concurrent b64 TF row
  (`3352.900 tok/s`) but does not beat the earlier solo-ish V2 TF sighting
  (`3728.701 tok/s`). The paired profile run (b64, MT2048, ignore-EOS,
  profiling enabled) showed full-shape device means of target/draft/reject
  `27.774 / 24.756 / 3.050 ms`, so the b64 TF ceiling is dominated by paying
  the two model forwards, not by rejection sampling.
- 2026-07-08 sampler/K follow-up on `vp-dgx-68`: skipping bonus-token logprob
  materialization when request logprobs are disabled is compile-safe but did
  not move the b64 profile result (`2526.647 tok/s`, accept `5.143`; full-shape
  target/draft/reject `27.509 / 24.499 / 3.074 ms`). A short K sweep did not
  justify moving away from K=16: K8 under-accepted, K24 was slower, and K32
  OOMed under the current capture/memory settings. Decision: keep K=16 as the
  default.
- 2026-07-08 b64 speedup-target follow-up on `vp-dgx-68`: with frequent refill
  (`num_prompts=64`, `n_sample=4`, MT2048, ignore-EOS), AR was
  `3432.078 tok/s` while TF fell to `2183.443 tok/s`; TF full-running windows
  still included prompt/refill work and averaged only `2060.429 tok/s`. With
  pure decode (`num_prompts=64`, `n_sample=1`, MT4096, ignore-EOS), AR was
  `3302.664 tok/s`, TF was `3906.714 tok/s`, and pure `Running=64`/prompt-0
  windows were `3314.573` for AR vs `5465.300` for TF (`1.65x`, small `n=2`).
  This reframed the b64 target: make serving/refill behave more like pure
  decode via spec-decode scheduling/prefill isolation.
- 2026-07-08 thresholded decode-first refill follow-up: added
  `VLLM_TIDAR_DECODE_FIRST_REFILL=1` and optional
  `VLLM_TIDAR_DECODE_FIRST_REFILL_MIN_RUNNING=<N>` for TiDAR. On the same
  refill-heavy b64 setup, default decode-first improved TF to
  `3052.678 tok/s` but underfilled waves; min-running `60` reached
  `3602.714 tok/s`; min-running `48` reached **`3847.481 tok/s`**, which is
  `1.12x` over the paired AR `3428.616 tok/s` and close to the pure-decode TF
  aggregate `3906.714 tok/s`. The vLLM interval mean accept for the min-running
  `48` run was about `5.03`; the probe-side acceptance monkeypatch reported
  null counters on the thresholded async path, so use vLLM metrics for those
  rows until the probe is repaired.
- 2026-07-08 AMD threshold follow-up on `cnode-83`: same short refill-heavy
  shape as the NVIDIA threshold test (`iter_0012000`, b64, `num_prompts=64`,
  `n_sample=4`, MT2048, ignore-EOS, K=16, prompt-token IDs) with
  `ROCM_AITER_FA`, `FULL_AND_PIECEWISE`, TiDAR TF paged attention/no-splits,
  and min-running `48` reached `2468.453 tok/s`, corrected accept `4.937`.
  The paired AR baseline on the same node/config family was `2877.797 tok/s`,
  so TF/AR was only `0.86x`. The log confirmed both AITER-FA TF pass modes
  (`causal=True` verify and `causal=False` draft) and the full `1088` b64 graph
  cap, so this is not the old graph-cap or missing-TF-paged-attention failure.
  Current read: the NVIDIA scheduler gate transfers functionally to AMD, but
  does not by itself close the high-batch speedup gap on MI300X.

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
- `/data/home/jinzhao/nv_v2_tidar_logs/iter12000_vs_12600_backtoback_20260707_030645/`
- `/data/home/jinzhao/nv_v2_tidar_logs/iter12000_m10k_sweep_20260707_174814/`
- `/data/home/jinzhao/nv_v2_tidar_logs/iter12000_m10k_vp82_20260707_155231/v2_tf64_m10k_iter12000_vp82_gpu6.log`
- `/data/home/jinzhao/nv_v2_tidar_logs/iter12000_b64_saturation_20260707_224020/`
- `/data/home/jinzhao/nv_v2_tidar_logs/iter12000_b64_sampleropt_profile_20260708_002150/`
- `/data/home/jinzhao/nv_v2_tidar_logs/iter12000_b64_k_sweep_20260708_002645/`
- `/data/home/jinzhao/nv_v2_tidar_logs/iter12000_b64_window_pair_20260708_005502/`
- `/data/home/jinzhao/nv_v2_tidar_logs/iter12000_b64_pure_decode_pair_20260708_010139/`
- `/data/home/jinzhao/nv_v2_tidar_logs/iter12000_b64_decodefirst_pair_20260708_010918/`
- `/data/home/jinzhao/nv_v2_tidar_logs/iter12000_b64_decodefirst_min60_20260708_011951/`
- `/data/home/jinzhao/nv_v2_tidar_logs/iter12000_b64_decodefirst_min48_20260708_012510/`
- `/data/home/jinzhao/nv_v2_tidar_logs/ar_v2_iter12600_captured_20260707_034419/`
- `/shared/home/jinzhao/tfscope/amd_v024_min48_20260708_214420/`

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

This began as a b1/b16/b64 matched-config AR/TF/SF comparison on AIME thinking-on, max_tokens=10000, K=16. It predates the current V2 TiDAR path. The table below now reports **best-known** throughput per mode/platform/batch. NVIDIA TF now uses V2 `FULL_AND_PIECEWISE` for b1/b16 and the V2 thresholded decode-first refill run for b64, which supersedes the older v0.16-family old-runner b64 value. AMD TF b1 uses the earlier current V2 short run; AMD TF b16 uses the same-checkpoint `iter_0012600` matched run; AMD TF b64 uses the absolute-best AMD-only V2 run, while the matched `iter_0012600` b64 parity value is `1682.274 tok/s`.

**Best throughput by mode and batch size, with backend used for each mode:**

| Mode | NVIDIA backend | AMD backend | b=1 NV / AMD | b=16 NV / AMD | b=64 NV / AMD |
|---|---|---|---:|---:|---:|
| AR | mixed captured: `FLASH_ATTN` v3 + cudagraph (`FULL` or `FULL_AND_PIECEWISE`) | mixed captured: `ROCM_AITER_FA` + cudagraph + AITER-MoE | 101 / **75.9** | 1142 / **833.8** | **3420.864** / **2622.038** |
| TF | V2 async `FLASH_ATTN` v3 + `FULL_AND_PIECEWISE`; b64 also uses TiDAR decode-first refill min-running `48` | V2 async captured `ROCM_AITER_FA` + `FULL_AND_PIECEWISE` | **262.855** / **108.635** | **1815.837** / **990.631** | **3847.481** / **2743.088** |
| SF | FLEX + Triton split-K + FULL cudagraph | FLEX + Triton split-K + FULL cudagraph | 181 / **159** | 994 / **952** | 1337 / **1406** |

Best tput for each mode on each platform after applying the V2 TF supersession where it wins:

| Mode | NVIDIA backend | NVIDIA best tput | NVIDIA best bsz | AMD backend | AMD best tput | AMD best bsz |
|---|---|---:|---:|---|---:|---:|
| AR | mixed captured: `FLASH_ATTN` v3 + cudagraph (`FULL` or `FULL_AND_PIECEWISE`) | **3420.864 tok/s** | 64 | mixed captured: `ROCM_AITER_FA` + cudagraph + AITER-MoE | **2622.038 tok/s** | 64 |
| TF | V2 async `FLASH_ATTN` v3 + `FULL_AND_PIECEWISE`; b64 with TiDAR decode-first refill min-running `48` | **3847.481 tok/s** | 64 | V2 async captured `ROCM_AITER_FA` + `FULL_AND_PIECEWISE` | **2743.088 tok/s** | 64 |
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
https://huggingface.co/Zyphra-staging/smoediffusion_128k-hf_iter_0012600
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
3. **Profile/diagnose AMD b64 TF under the thresholded scheduler.** The direct
   `cnode-83` run proved the scheduler gate works on AMD and captures the true
   `1088` b64 TiDAR shapes, but TF still landed below paired AR
   (`2468.453` vs `2877.797 tok/s`) despite corrected accept `4.937`. Next
   compare AMD full-shape target/draft/reject timings against the NVIDIA
   profile and test AMD-specific refill thresholds or lower-tail/drain shapes.
4. **Refine DP pause/resume** so EPLB does not need `VLLM_TIDAR_V2_DP_KEEPALIVE=1` busy-spin while idle.
5. **Decide whether to submit the v0.24 OOB PR** from `docs/tidar_amd_handoff/`; it is independent and already validated.
6. **Longer-term cleanup:** either keep the pragmatic V2 `TiDARSpeculator` route or fold it into a v0.24-style `TiDARSMoEModelState(MambaHybridModelState)` during a larger upstream sync.

## 8. References

- Long-form migration log: `docs/dllm_migration_scope.md`
- OOB PR materials: `docs/tidar_amd_handoff/{fix.diff,PR_DESCRIPTION.md,repro_oob.py}`
- TF split-K design: `docs/tf_full_splitk_design.md`
- Remote bench/prompts: `/shared/home/jinzhao/tfscope/{tf_accept_aiter.py,aime26_thinkon.json}`
- Memory entries: `reference_vllm_v024_dllm_is_tidar_tf.md`, `project_tf_on_aiter_fa.md`, `project_amd_tidar_host_overhead.md`, `reference_tidar_amd_env.md`
