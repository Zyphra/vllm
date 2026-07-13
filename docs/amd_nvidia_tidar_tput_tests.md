# TiDAR TF on AMD: Throughput Gap and Reproducer

_Last updated: 2026-07-13. Audience: AMD and vLLM performance engineers._

Repository: [Zyphra/vllm-smoe-amd](https://github.com/Zyphra/vllm-smoe-amd),
branch `jinzhao/tidar_v024`. Checkpoint:
[Zyphra-staging/smoediffusion_128k-hf_iter_0012600](https://huggingface.co/Zyphra-staging/smoediffusion_128k-hf_iter_0012600).

## Executive Summary

TiDAR two-forward (TF) is correct and operational on one MI300X with the vLLM
V2 GPU runner, async scheduling, target and draft cudagraph replay,
`ROCM_AITER_FA`, AITER unquantized MoE, CCA recurrent-state handling, and
prefix rejection.

The main result is the drain-controlled 10k-output lockstep table below. It
keeps the full batch active and times complete device iterations, avoiding the
request-tail drain that distorted older fixed-output tests while exercising
the full long-context decode path. Under this workload:

- Adaptive split-K restores AMD TF speedups of `2.15x`, `2.09x`, `1.81x`, and
  `1.07x` at b8/b16/b32/b64. The forced no-splits control made those rows look
  much worse. Batch 1 remains unresolved at `0.69x` AR.
- At b64, MI300X reaches `3128 tok/s` TF versus H100 `4008 tok/s`, or `0.78x`.
  The paired AR ratio is `2933/3668 = 0.80x`, so accepted-token platform scaling
  is nearly the same for AR and TF.
- The b64 TF device step is still `94.4 ms` on MI300X versus `60.2 ms` on H100
  (`1.57x` slower). The AR step is `21.8 ms` versus `17.4 ms` (`1.25x`
  slower); higher AMD acceptance (`4.615` versus `3.769`) compensates for most
  of that latency gap in output tok/s.
- At b1, no-splits preserves acceptance (`4.217`) but takes `112.3 ms` per TF
  step. Adaptive split-K cuts the step to `36.1 ms` but changes the trajectory
  to acceptance `1.330`; both deliver only about `37 tok/s`. This is the
  clearest remaining AMD TF acceptance/performance target.
- Single-GPU b128 is not a valid 10k lockstep comparison: TF exhausts cache
  capacity on both platforms before the common completion boundary. H100 AR
  also preempts, while MI300X AR remains resident.
- In the prior MT512 component profile, the remaining absolute AMD gap is
  concentrated in the two model backbones: at b64 AMD adds `6.73 ms` to target
  and `7.02 ms` to draft relative to H100, but only `0.73 ms` to all target
  sampling.
- A raw-byte b1/b64 trace localizes AMD's acceptance-shape variation to
  unquantized dense GEMMs. The first difference is the attention output
  projection, after byte-identical CCA and AITER attention outputs. AITER MoE
  is not the initial source; CCA convolution is a second source.
- An opt-in fixed-reduction CCA kernel removes that second source and improves
  matched MI300X b64 steady TF from `5139.7` to `5571.2 tok/s` (`+8.4%`) while
  reducing its step from `65.25` to `60.20 ms`. Making all dense projections
  fixed-reduction as well produces byte-identical b1/b64 layer outputs and the
  same complete acceptance trajectory, but that diagnostic dense kernel is too
  slow for production.

The older aggregate fixed-output tests made AMD TF look slower than AMD AR at
high batch because requests completed at different rates and the batch drained
through progressively smaller graph shapes. Those results are retained in the
appendix because they describe finite offline batches, but they should not be
used as the device-throughput ceiling. The 10k no-splits control had a separate
long-prefix attention bottleneck; adaptive split-K removes most of it.

## Drain-Controlled AMD/NVIDIA Result

The matched lockstep workload replicates one AIME25 prompt and sampling state
across every request (`num_prompts=1`, `n_sample=B`), forces exactly 10,000
output tokens, and ignores EOS. The probe reports only complete full-batch
decode iterations after warmup. AR emits one token per request per iteration;
TF reports accepted output tokens per complete target/sample/CCA/draft
interval. Prefill, startup, partial final TF iterations, and all smaller tail
shapes are excluded. Unlike the earlier MT512 table, these rates average the
device cost from short prefixes through a 10k-token context.

Both platforms use `iter_0012600`, exactly one forced BOS, target temperature
`0.6`, argmax draft, K=16, V2 async scheduling, exact full-batch graph capture,
and `FULL_AND_PIECEWISE`. H100 uses `FLASH_ATTN` v3 and the default CCA
convolution. MI300X uses `ROCM_AITER_FA`, AITER unquantized MoE, and the
batch-invariant CCA convolution. The table reports the best measured AMD TF
setting per batch: no-splits at b1 and adaptive TF paged split-K at b8-b64.

| bsz | NVIDIA AR | NVIDIA TF / acc | TF/AR | AMD AR | AMD TF / acc | TF/AR | AMD/NVIDIA AR | AMD/NVIDIA TF |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `84.750` | `129.680` / `4.280` | `1.53x` | `54.760` | `37.564` / `4.217` | `0.69x` | `0.65x` | `0.29x` |
| 8 | `619.152` | `1026.857` / `4.498` | `1.66x` | `397.457` | `855.992` / `4.646` | `2.15x` | `0.64x` | `0.83x` |
| 16 | `1190.385` | `1713.385` / `4.267` | `1.44x` | `837.559` | `1753.972` / `5.680` | `2.09x` | `0.70x` | `1.02x` |
| 32 | `2124.758` | `2646.111` / `3.754` | `1.25x` | `1490.878` | `2702.458` / `5.606` | `1.81x` | `0.70x` | `1.02x` |
| 64 | `3668.461` | `4007.641` / `3.769` | `1.09x` | `2932.523` | `3127.915` / `4.615` | `1.07x` | `0.80x` | `0.78x` |
| 128 | capacity-limited | capacity-limited | - | `4630.055` | capacity-limited | - | - | - |

All rates are device-event output tok/s. Acceptance includes TiDAR's normal
bonus token. Rows b1-b64 keep every request active until the common completion
boundary. At b128, H100 AR records only 6,830 full-batch calls out of 9,999;
H100 and MI300X TF retain b128 for only 618 and 597 calls, respectively. Those
early-context rates are intentionally omitted.

Acceptance varies with batch shape and long trajectory because BF16 dense-GEMM
and CCA reduction geometry can alter proposals near decision boundaries. The
controlled trace below shows that this is model-level numerical sensitivity,
not rejection-sampler corruption. Raw TF tok/s combines device time with the
acceptance shown in the table; use step latency for a kernel-only comparison.

This is not caused by consuming a different global RNG stream. Every replicated
request receives seed 0, and the V2 Gumbel sampler derives noise from the
request seed and absolute token position. A fixed batch/config is deterministic,
but changing B changes GEMM M, attention split count, and reduction order. Tiny
BF16 differences can eventually cross an argmax or Gumbel boundary; after that
the autoregressive trajectories and their mean acceptance differ.

Source logs:

- NVIDIA TF and b1-b32 AR: `/data/home/jinzhao/nv_v2_tidar_logs/mt10k_cca_fix_20260713_r2/`
- NVIDIA b64 AR retry: `/data/home/jinzhao/nv_v2_tidar_logs/mt10k_cca_fix_20260713_r2_retry/`
- AMD AR and no-splits control: `/shared/home/jinzhao/tfscope/amd_lockstep_steady_271083/`,
  `/shared/home/jinzhao/tfscope/amd_lockstep_steady_271084/`, and
  `/shared/home/jinzhao/tfscope/amd_lockstep_steady_271111/`
- AMD adaptive split-K TF: `/shared/home/jinzhao/tfscope/amd_lockstep_steady_271123/`

Slurm job 271083 is marked failed only because its final case was assigned the
invalid GPU ID 8; its completed b1/b8/b16 and AR-b32 logs are valid. The driver
now rejects out-of-range case layouts before launching.

## Batch-Shape Numerical Diagnostic

Fresh-process b1 and b64 runs used the same prompt, token IDs, positions,
checkpoint, K=16, and exact graph sizes `M=17` and `M=1088`. The probe hashes
the first request's BF16 draft hidden rows and FP32 logits as raw bytes. On
H100, all 16 hidden rows, all 16 logits rows, and all 16 top-1 proposals match
for each of the first eight traced proposal steps. Batch-shape invariance is
therefore attainable for this workload.

The initial AMD A/B results are:

| AMD variant | Hidden rows | Logit rows | Top-1 proposals | Result |
|---|---:|---:|---:|---|
| AITER MoE + AITER-FA + default CCA | `0/16` | `0/16` | `13/16` | Baseline |
| Triton MoE | `0/16` | `0/16` | `15/16` | Does not restore invariance; slower |
| Triton attention | `0/16` | `0/16` | `15/16` | Does not restore invariance |
| CCA unfold/einsum convolution | `0/16` | `0/16` | `13/16` | Does not restore invariance |
| ROCm skinny GEMM disabled | `0/16` | `0/16` | `13/16` | Does not restore invariance |
| Fixed-reduction `o_proj` | `0/16` | `0/16` | `10/16` | Layers 0-1 become exact |
| Fixed-reduction all dense linears | `0/16` | `0/16` | `10/16` | First mismatch moves to layer-2 CCA conv |
| Fixed-reduction dense + fixed-reduction CCA | `16/16` | `16/16` | `16/16` | All 80 layer outputs and the full acceptance trajectory match |

Layer-boundary hashes identify the first responsible operation. In layer 0,
the normalized input, CCA QKV, and AITER attention output before `o_proj` are
all `16/16` byte-identical. The `o_proj` output is only `7/16` identical. This
is an unquantized `ReplicatedLinear` with shape `M x 1024 -> M x 2048`.
Disabling the low-row `wvSplitKrc` path does not change the result, so the
problem is not limited to skinny GEMM; the fallback
`torch.nn.functional.linear`/ROCm BLAS path is also M-dependent.

Forcing only `o_proj` through v0.24's fixed-reduction Triton GEMM makes layer 0
and the following AITER MoE layer both `16/16` identical. This directly
exonerates AITER MoE as the initial numerical source. The next difference is
inside layer 2: its current input, Q/K linear output, and cached hidden state
are exact, but its cached Q/K window is already different before convolution.
The CCA value projection also differs in one row despite exact inputs. The
alternate convolution alone leaves this boundary unchanged because its input
state has already diverged.

Forcing all unquantized dense linears through the fixed-reduction kernel makes
layer 0, AITER MoE, the layer-2 Q/K and hidden caches, Q/K projections, and
value projections exact. The first remaining difference is then the default
CCA convolution (`14/16` exact rows), followed by layer-2 QKV (`15/16`). This
establishes the order: ROCm dense GEMMs are the first source and CCA convolution
is a second source; AITER MoE is not causal for the numerical divergence.

The new fixed-reduction CCA implementation completes the controlled pair. With
all dense projections and CCA fixed, every one of the first proposal's 80
layer outputs, all 16 draft hidden/logit/token rows, and the complete 27-step
acceptance trajectory are byte-identical between b1 and b64. The final token
hashes remain unchanged. This proves that dense GEMM and CCA reduction geometry
fully explain the observed model-level batch-shape divergence in this test.

The all-dense diagnostic is not a performance fix. Its fixed-reduction Triton
GEMM raises b1/b64 steps to `53.40/76.95 ms`. The CCA implementation is useful
independently: it is enabled with `VLLM_CCA_BATCH_INVARIANT_CONV=1`, is exact
across b1/b64 and cudagraph replay, and is faster than the existing library
convolution on MI300X (`0.052/0.052 ms` versus `0.094/0.125 ms` in the isolated
b1/b64 probe).

Matched full-model steady-state results are:

| MI300X variant | b1 acc / step / tok/s | b64 acc / step / tok/s | b1/b64 trajectory |
|---|---:|---:|---|
| rocBLAS control + default CCA | `5.24 / 36.31 ms / 144.3` | `5.24 / 65.25 ms / 5139.7` | Different |
| rocBLAS control + fixed CCA | `6.55 / 35.74 ms / 183.2` | `5.24 / 60.20 ms / 5571.2` | Dense-dependent |
| Fixed dense + fixed CCA | `5.038 / 53.40 ms / 94.4` | `5.038 / 76.95 ms / 4190.7` | Exact |

The production-relevant fixed-CCA A/B lowers b64 step latency by `7.7%` and
raises accepted throughput by `8.4%`; acceptance and the final b64 token hash
are unchanged. At b1 the trajectory changes because dense GEMM is still
M-dependent, so compare step latency (`-1.6%`) rather than attributing the
larger accepted-throughput change entirely to the CCA kernel.

The practical numerical priority is now a fast, fixed-reduction ROCm BF16
dense GEMM for the SMoE projection shapes. The diagnostic implementation is
opt-in through `VLLM_TIDAR_BATCH_INVARIANT_O_PROJ=1`; it proves causality but is
not a production setting. An isolated `M=17` versus `M=1088` probe finds only
3 of 34,816 BF16 outputs different, but default hipBLASLt, forced hipBLASLt,
rocBLAS, and rocBLAS with atomics disabled all produce the same hashes and
latency. Backend or atomics environment switches therefore do not solve it.
AITER MoE remains a throughput target, just not the cause of the b1/b64
numerical divergence. The fixed CCA path is ready for AMD review and broader
validation before becoming the default.

Diagnostic logs:

- H100 control: `/data/home/jinzhao/nv_v2_tidar_logs/vp49_shape_hash_control_20260713/`
- AMD component A/B: `/shared/home/jinzhao/tfscope/amd_shape_hash_ab_265694/`
- AMD Triton attention and layer trace: `/shared/home/jinzhao/tfscope/amd_shape_hash_ab_265808/`
- AMD stage trace: `/shared/home/jinzhao/tfscope/amd_shape_hash_ab_265883/`
- AMD skinny-GEMM control: `/shared/home/jinzhao/tfscope/amd_shape_hash_ab_265951/`
- AMD fixed-`o_proj` and state trace: `/shared/home/jinzhao/tfscope/amd_shape_hash_ab_267096/`
- AMD fixed-`o_proj` plus CCA unfold: `/shared/home/jinzhao/tfscope/amd_shape_hash_ab_266125/`
- AMD fixed-reduction all-dense control: `/shared/home/jinzhao/tfscope/amd_shape_hash_ab_268004/`
- AMD fixed dense + fixed CCA full-model A/B: `/shared/home/jinzhao/tfscope/amd_shape_hash_ab_270138/`
- AMD CCA and dense-GEMM microprobes: `/shared/home/jinzhao/tfscope/slurm-tidar-cca-probe-270153.log`

Reproduce the full-model controls and isolated kernels with:

```bash
ONLY=core sbatch benchmarks/tidar/slurm_shape_hash_ab.sh
ONLY=invariant_dense sbatch benchmarks/tidar/slurm_shape_hash_ab.sh
ONLY=cca_fixed sbatch benchmarks/tidar/slurm_shape_hash_ab.sh
ONLY=invariant_dense_cca_fixed sbatch benchmarks/tidar/slurm_shape_hash_ab.sh
sbatch benchmarks/tidar/slurm_cca_kernel_probe.sh
```

## Actual Remaining Gap

The 10k result adds a long-prefix effect that the MT512 profiles did not expose.
Forcing a single TF attention pass raises the AMD b64 step to `139.8 ms` and
reduces TF to `1880 tok/s`. Adaptive split-K lowers the step to `94.4 ms` and
raises TF to `3128 tok/s`. The remaining b64 TF step ratio is `1.57x` AMD/H100,
versus `1.25x` for AR, although higher AMD acceptance makes accepted-token
platform scaling nearly equal (`0.78x` TF versus `0.80x` AR).

Batch 1 is the sharper unresolved case. Adaptive split-K reaches a competitive
`36.1 ms` step but changes the draft trajectory to acceptance `1.330`;
no-splits restores acceptance `4.217` but takes `112.3 ms`. The custom AMD TF
attention reads the full paged prefix for 17 query positions in each target and
draft pass, so both split selection and verifier-feeding numerics need focused
work.

The direct MI300X attention-mode A/B is:

| bsz | No-splits TF / acc / step | Adaptive TF / acc / step | Adaptive throughput gain |
|---:|---:|---:|---:|
| 1 | `37.564 / 4.217 / 112.3 ms` | `36.882 / 1.330 / 36.1 ms` | `0.98x` |
| 8 | `283.421 / 4.103 / 115.8 ms` | `855.992 / 4.646 / 43.4 ms` | `3.02x` |
| 16 | `619.448 / 4.516 / 116.7 ms` | `1753.972 / 5.680 / 51.8 ms` | `2.83x` |
| 32 | `1675.888 / 6.119 / 116.8 ms` | `2702.458 / 5.606 / 66.4 ms` | `1.61x` |
| 64 | `1879.970 / 4.106 / 139.8 ms` | `3127.915 / 4.615 / 94.4 ms` | `1.66x` |

The earlier matched MT512 b32 and b64 event profiles isolate the residual
short-context device gap. `Target` and `Draft` are backbone graph replays.
`Target sample` includes the target vocabulary projection, temperature/Gumbel
sampling, and prefix rejection. LM-head and Gumbel values are nested inside
those totals.

| Platform | bsz | TF step | Target | Draft | Target sample | Target LM head | Draft LM head | Gumbel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H100 | 32 | `40.327 ms` | `18.177 ms` | `15.536 ms` | `2.803 ms` | `0.927 ms` | `0.876 ms` | `1.340 ms` |
| H100 | 64 | `51.820 ms` | `22.426 ms` | `19.374 ms` | `5.289 ms` | `1.667 ms` | `1.612 ms` | `2.626 ms` |
| MI300X | 32 | `53.192 ms` | `24.166 ms` | `21.600 ms` | `3.178 ms` | `1.498 ms` | `1.192 ms` | `1.266 ms` |
| MI300X | 64 | `67.092 ms` | `29.153 ms` | `26.396 ms` | `6.019 ms` | `3.032 ms` | `2.191 ms` | `2.253 ms` |

At b64, AMD versus H100 adds:

| Component | AMD - H100 | AMD/H100 |
|---|---:|---:|
| Target backbone | `+6.727 ms` | `1.30x` |
| Draft backbone | `+7.022 ms` | `1.36x` |
| Entire target sample | `+0.730 ms` | `1.14x` |
| Prefix rejection | about `+0.004 ms` | negligible |

The two backbones account for nearly all of the absolute platform gap. AMD
Gumbel is actually `0.37 ms` faster; rejection is not a useful optimization
target.

The b32-to-b64 latency growth has the same structure on both GPUs. Roughly 70%
comes from target plus draft backbone growth and about 20% from full-vocabulary
sampling. AR uses approximately `M=B`, while each TF backbone uses `M=17B` and
TF runs it twice. Large-M arithmetic intensity approaching a limit does not
make latency independent of M: FLOPs and activation traffic still grow with M.
TF throughput continues to rise at b64, but AR scales faster because its
small-M decode remains favorable for weight reuse and amortization.

An H100 b64 Nsight trace gives the current operation mix: fused MoE is `16.3%`
of GPU kernel time; named CCA convolution/layout is about `16%` and exceeds
`20%` with adjacent copies and concatenations; vocabulary GEMMs are `6.1%`;
temperature plus FP64 Gumbel are `6.4%`; FlashAttention is about `1.9%`.
Top-1 E=16 MoE at b64 routes only about `1088/16 = 68` rows per expert, so the
expert GEMMs remain in an inefficient small-M regime even though the aggregate
TF row count is 1088.

Profile logs:

- H100 b32: `/data/home/jinzhao/nv_v2_tidar_logs/vp49_lockstep_b32_profile_20260712_225215/tf_b32.log`
- H100 b64: `/data/home/jinzhao/nv_v2_tidar_logs/vp49_lockstep_b64_profile_20260712_225044/tf_b64.log`
- H100 graph nodes: `/data/home/jinzhao/nv_v2_tidar_logs/vp49_b64_graphnodes_20260712/tf_b64_nodes.nsys-rep`
- MI300X b32: `/shared/home/jinzhao/tfscope/amd_lockstep_steady_265615/tf_b32.log`
- MI300X b64: `/shared/home/jinzhao/tfscope/amd_lockstep_steady_265614/tf_b64.log`

## Specific AMD Help Requested

The highest-value work is inside the model graphs at the exact TiDAR shapes:

1. Profile and tune long-prefix `vllm/attention/ops/tf_attention.py`. First fix
   the b1 split-K numerical trajectory without losing its `36.1 ms` step; then
   reduce the adaptive b64 `94.4 ms` step. Sweep split count using prefix length
   and launch occupancy, and compare target and draft attention separately.
   Preserve causal verify and bidirectional draft numerics.
2. Provide or tune a fixed-reduction ROCm BF16 dense GEMM for the SMoE
   `ReplicatedLinear` shapes, starting with attention `o_proj`
   (`K=1024`, `N=2048`, `M=17/136/272/544/1088`). It should retain the H100's
   b1/b64 numerical trajectory without giving up the current BLAS throughput.
3. Review, tune, and help land the new fixed-reduction Triton CCA convolution.
   It closes the CCA divergence, is cudagraph-stable, and improves matched b64
   TF by `8.4%` on MI300X. Broader prompt, prefill, dtype, and shape validation
   is needed before enabling it by default.
4. Tune AITER unquantized top-1 MoE for E=16, H=2048 and aggregate
   `M=544/1088/2176`, paying attention to low per-expert row counts. Current
   logs select the generic two-stage default rather than a tuned row. This is
   a throughput target, not the initial numerical-divergence source.
5. Trace and reduce the ROCm CCA layout, copy, and convolution chain in target
   and draft. Any replacement must preserve candidate-state stashing, stable
   request slots, and separate draft scratch state.
6. Tune the FP32-output vocabulary GEMM. At b64 the AMD target LM head is
   `1.82x` H100 and the draft LM head is `1.36x` H100. A fused draft
   LM-head-plus-argmax path is especially relevant because draft temperature is
   zero.
7. Use a ROCm kernel trace to verify the long-context shares above. Prefix
   rejection is already negligible.
8. Preserve acceptance and output behavior. Performance changes should be
   compared using full-batch step time as well as accepted tok/s so numeric
   trajectory changes are visible.

Earlier AMD screening found no gain from Triton MoE, the alternate SMoE AITER
op, removing router padding, CCA Triton fusion, or CCA unfold/einsum. Leave
those variants off unless their implementations change.

## MT512 b128 Block-Table Fix

The original short-context b128 fault was in the V2 block-table preparation
path, not memory capacity or graph size. On H100,
`CUDA_LAUNCH_BLOCKING=1` localized the first
misaligned access to `_gather_block_tables_kernel` at `(2 cache groups, 128
requests)`. Replacing only that gather exposed `_compute_slot_mappings_kernel`
as the next failure. Both kernels loaded raw block-table addresses from a
device-side `uint64` pointer array.

`vllm/v1/worker/gpu/block_table.py` now launches once per cache group and
passes the typed source, destination, and slot-mapping tensors directly to
Triton. This removes pointer-to-pointer loads while remaining graph-capturable.
The old combined launch saved one kernel launch, but the replacement changes
b64 H100 TF from `9247` to `9230 tok/s` (`-0.2%`), within run variance.

The fixed `M=2176` graph captures and runs on both platforms:

| Platform | AR | TF / acc | TF/AR | TF step | Acceptance-normalized TF at acc 7.514 |
|---|---:|---:|---:|---:|---:|
| H100 | `8512.417` | `8379.099` / `5.396` | `0.98x` | `82.427 ms` | `11669 tok/s` |
| MI300X | `5687.594` | `6385.331` / `5.420` | `1.12x` | `108.640 ms` | `8853 tok/s` |

The normalized column is diagnostic, not a measured output rate. It removes
the b128 trajectory change to show capacity at the b1-b64 reference acceptance.
The measured AMD/H100 TF step ratio is `1.32x`, close to `1.30x` at b64; b128
does not introduce a new AMD-specific scaling failure.

Diagnostic logs:

- `/data/home/jinzhao/nv_v2_tidar_logs/vp49_b128_launchblocking_20260712.log`
- `/data/home/jinzhao/nv_v2_tidar_logs/vp49_b128_typed_blocktables_20260712/tf_b128.log`
- `/data/home/jinzhao/nv_v2_tidar_logs/vp49_b128_typed_blocktables_20260712/tf_b64_regression.log`
- `/shared/home/jinzhao/tfscope/amd_lockstep_steady_265635/tf_b128.log`

## Branch and Implementation Context

`jinzhao/tidar_v024` is based on the Zyphra v0.16 fork at `12cb2e780`. Despite
the branch name, it is not stock upstream vLLM v0.24. It selectively ports the
v0.24/DiffusionGemma architecture needed for the fork's existing SMoE, CCA,
AITER, and TiDAR stack.

The starting fork already had the V2 GPU runner, `AsyncOutput`, EAGLE-style
speculative decode, scheduler integration, and cudagraph management. This
branch adds hybrid Mamba/CCA cache-group support, stable request-keyed CCA
state, post-rejection state commit, a V2-native self-draft `TiDARSpeculator`,
exact TiDAR graph shapes, and per-pass causal selection. The complete upstream
`gpu/model_states/` framework was not copied; equivalent minimum hooks live in
`gpu/attn_utils.py`, `gpu/model_runner.py`, CCA, and `gpu/spec_decode/tidar.py`.

For K=16, each steady TF iteration performs:

1. A causal target verify over K draft IDs plus one target slot.
2. Target sampling and prefix rejection, including the normal bonus token.
3. Commit of the accepted target CCA candidate state to a stable request slot.
4. A bidirectional self-draft over `[last_token, mask, ..., mask]` using the
   committed state and separate scratch state.
5. Async output propagation and scheduler handoff of the next K draft IDs.

Both forwards have uniform size `B * (K + 1) = 17B`. The verify pass uses the
V2 graph manager; `TiDARSpeculator` captures its own persistent-buffer draft
graphs. Two-forward TiDAR does not need per-sequence-causal unified attention:
target and draft are separate forwards, so the AITER-FA causal flag is selected
per pass.

On AMD, a correct run logs both:

```text
Using TiDAR TF paged attention in ROCm AITER-FA (S=17, causal=True).
Using TiDAR TF paged attention in ROCm AITER-FA (S=17, causal=False).
```

Draft temperature is zero, so proposals are argmax. Target temperature `0.6`
is supported through the V2 sampler. The fallback mask token is ID 4 when the
checkpoint has no configured mask ID, matching the old runner.

## Code Map

| Area | File |
|---|---|
| V2 runner and verify graph | `vllm/v1/worker/gpu/model_runner.py` |
| Async output | `vllm/v1/worker/gpu/async_utils.py` |
| V2 self-draft | `vllm/v1/worker/gpu/spec_decode/tidar.py` |
| Per-pass attention setup | `vllm/v1/worker/gpu/attn_utils.py` |
| ROCm AITER-FA backend | `vllm/v1/attention/backends/rocm_aiter_fa.py` |
| TF paged attention | `vllm/attention/ops/tf_attention.py` |
| CCA | `vllm/model_executor/layers/mamba/cca.py` |
| SMoE/MoE/LM head | `vllm/model_executor/models/smoe.py` |
| Block tables and b128 fix | `vllm/v1/worker/gpu/block_table.py` |
| AR probe | `benchmarks/tidar/probe_v2_ar.py` |
| TF probe | `benchmarks/tidar/probe_v2_tidar_nv.py` |
| AMD lockstep Slurm driver | `benchmarks/tidar/slurm_lockstep_steady.sh` |
| NVIDIA lockstep driver | `benchmarks/tidar/run_lockstep_nvidia.sh` |
| AMD shape-hash A/B driver | `benchmarks/tidar/slurm_shape_hash_ab.sh` |

## Lockstep Reproducer

Clone the tested branch and download the public checkpoint:

```bash
git clone --branch jinzhao/tidar_v024 \
    https://github.com/Zyphra/vllm-smoe-amd.git
cd vllm-smoe-amd

python3 -m pip install -q huggingface_hub
export CKPT=/shared/home/$USER/checkpoints/smoediffusion_128k-hf_iter_0012600
huggingface-cli download Zyphra-staging/smoediffusion_128k-hf_iter_0012600 \
    --local-dir "$CKPT"
```

The tested AMD image is `zyphra/rocm-primus:aiter_pa_swa` with Torch 2.10,
Triton 3.7, and ROCm 7.2. The first load in a fresh image can spend about 12
minutes compiling AITER kernels. On shared IBM nodes, do not use GPU 0.

The checked-in Slurm script runs AR or TF lockstep tests across GPUs. Submit
the modes separately to cover b1-b64 without using GPU 0:

```bash
CKPT="$CKPT" BATCHES="1 8 16 32 64" RUN_AR=1 RUN_TF=0 \
    MAX_TOKENS=10000 CCA_BATCH_INVARIANT=1 IGNORE_EOS=1 \
    sbatch benchmarks/tidar/slurm_lockstep_steady.sh

CKPT="$CKPT" BATCHES="1 8 16 32 64" RUN_AR=0 RUN_TF=1 \
    MAX_TOKENS=10000 CCA_BATCH_INVARIANT=1 TF_PAGED_NO_SPLITS=0 \
    IGNORE_EOS=1 \
    sbatch benchmarks/tidar/slurm_lockstep_steady.sh
```

Set `IGNORE_EOS=0` to measure natural completion and acceptance; the wrappers
then omit `--ignore-eos`. That is a different workload from the 10k stress
table: `MAX_TOKENS=10000` becomes only a cap, requests may finish much earlier,
and a varied-prompt batch will drain unless the benchmark continuously refills
it. Report natural-EOS acceptance separately from fixed-length lockstep
throughput.

For a direct single-GPU AMD TF run, build the branch in the container and use:

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_ATTENTION_BACKEND=ROCM_AITER_FA
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MHA=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_ROCM_MOE_PADDING=1
export VLLM_TIDAR_TWO_FORWARD=1
export VLLM_TIDAR_USE_TF_PAGED_ATTENTION=1
export VLLM_TIDAR_TF_PAGED_NO_SPLITS=0
export VLLM_TIDAR_FA_NO_SPLITS=0
export VLLM_CCA_BATCH_INVARIANT_CONV=1
export PATCH_PROBE_STEADY=1
export PATCH_PROBE_STEADY_BATCH=64
export PATCH_PROBE_EXACT_CUDAGRAPH_BATCH=1

python -u benchmarks/tidar/probe_v2_tidar_nv.py \
  --ckpt "$CKPT" --dataset benchmarks/tidar/aime25_zpo_texts.json \
  --batch 64 --num-prompts 1 --max-num-seqs 64 --n-sample 64 \
  --max-tokens 10000 --warmup-tokens 64 --repeats 1 --seed 0 \
  --target-temp 0.6 --draft-temp 0 --num-spec-tokens 16 \
  --max-model-len 12000 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.65 --backend ROCM_AITER_FA \
  --cudagraph-mode FULL_AND_PIECEWISE --prompt-token-ids --force-bos \
  --ignore-eos
```

For H100, use the same arguments with `--backend FLASH_ATTN`,
`VLLM_ATTENTION_BACKEND=FLASH_ATTN`, and `VLLM_FLASH_ATTN_VERSION=3`; remove
the ROCm/AITER variables. `benchmarks/tidar/run_lockstep_nvidia.sh` wraps the
same probe and accepts `CASES="ar:1:1 tf:1:2 ..."` for direct multi-GPU runs.

For the best measured AMD b1 configuration, set
`VLLM_TIDAR_TF_PAGED_NO_SPLITS=1` and `VLLM_TIDAR_FA_NO_SPLITS=1`. Keep adaptive
split-K (`0`) for b8-b64.

A valid AMD run must report one forced BOS, AITER Flash Attention, AITER
unquantized MoE, both paged-attention causal signatures, and exact capture size
`17B`. Compare `PATCH_PROBE_STEADY` device step time before comparing accepted
tok/s.

## Appendix: Earlier Results

### MT512 Drain-Controlled Reference

This was the original short-context lockstep table. It remains useful for
isolating model-graph throughput, but it does not exercise long-prefix TF
attention and must not be substituted for the 10k table above.

| bsz | NVIDIA AR | NVIDIA TF / acc | TF/AR | AMD AR | AMD TF / acc | TF/AR | AMD/NVIDIA AR | AMD/NVIDIA TF |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `85.995` | `240.175` / `7.514` | `2.79x` | `56.466` | `170.186` / `6.564` | `3.01x` | `0.66x` | `0.71x` |
| 8 | `646.746` | `1813.426` / `7.514` | `2.80x` | `454.600` | `1143.828` / `6.633` | `2.52x` | `0.70x` | `0.63x` |
| 16 | `1297.721` | `3362.292` / `7.514` | `2.59x` | `814.879` | `2555.903` / `7.408` | `3.14x` | `0.63x` | `0.76x` |
| 32 | `2479.075` | `6201.975` / `7.514` | `2.50x` | `1580.296` | `3579.318` / `5.885` | `2.26x` | `0.64x` | `0.58x` |
| 64 | `4816.921` | `9247.443` / `7.514` | `1.92x` | `3482.532` | `6067.079` / `6.390` | `1.74x` | `0.72x` | `0.66x` |
| 128 | `8512.417` | `8379.099` / `5.396` | `0.98x` | `5687.594` | `6385.331` / `5.420` | `1.12x` | `0.67x` | `0.76x` |

### Finite-Batch Drain Results

These historical tests divide total generated tokens by total time for a fixed
set of requests with no refill. They are valid finite offline-batch numbers,
but TF requests finish at different iterations because accepted lengths vary.
The active batch therefore shrinks, and the aggregate rate mixes full-batch
execution with inefficient tail shapes. They motivated the lockstep benchmark
above and should not be read as the hardware ceiling.

#### MT4000, Target Temperature 0.6

Checkpoint `iter_0012600`, one forced BOS, K=16, argmax draft, ignore EOS,
`FULL_AND_PIECEWISE`; H100 uses `FLASH_ATTN` v3 and MI300X uses
`ROCM_AITER_FA` plus AITER MoE.

| bsz | NVIDIA AR | NVIDIA TF / acc | TF/AR | AMD AR | AMD TF / acc | TF/AR |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | `82.258` | `132.427` / `4.619` | `1.61x` | `60.702` | `60.753` / `4.209` | `1.00x` |
| 8 | `542.912` | `680.086` / `5.317` | `1.25x` | `377.326` | `314.923` / `5.422` | `0.83x` |
| 16 | `975.975` | `1333.164` / `5.263` | `1.37x` | `773.367` | `587.371` / `5.444` | `0.76x` |
| 64 | `3141.139` | `4043.730` / `5.942` | `1.29x` | `2532.184` | `1788.693` / `5.002` | `0.71x` |

Logs:

- NVIDIA: `/data/home/jinzhao/nv_v2_tidar_logs/iter12600_iter12000meta_mt4k_t06_d0_n1_vp16_20260709_190123/`
- AMD: `/shared/home/jinzhao/tfscope/amd_iter12600_iter12000meta_mt4k_t06_d0_n1_c17_20260710_010137/`

#### MT5000, Greedy

| bsz | NVIDIA AR | NVIDIA TF / acc | TF/AR | AMD AR | AMD TF / acc | TF/AR |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | `79.723` | `207.554` / `7.562` | `2.60x` | `53.828` | `62.296` / `4.703` | `1.16x` |
| 8 | `539.015` | `972.872` / `6.136` | `1.80x` | `382.078` | `375.808` / `6.746` | `0.98x` |
| 16 | `968.246` | `1603.001` / `6.802` | `1.66x` | `772.428` | `661.769` / `8.325` | `0.86x` |
| 64 | `3093.504` | `3482.665` / `6.936` | `1.13x` | `2509.826` | `2102.265` / `8.250` | `0.84x` |

AMD acceptance is higher than NVIDIA at b8-b64 here, yet aggregate AMD TF is
not faster than AR. This is direct evidence that acceptance alone does not
explain the finite-batch result.

Logs:

- NVIDIA: `/data/home/jinzhao/nv_v2_tidar_logs/iter12600_forcebos_mt5k_n1_vp16_20260709_175900/`
- AMD: `/shared/home/jinzhao/tfscope/amd_iter12600_forcebos_mt5k_n1_c17_20260710_002037/`

#### H100 Metric-Control Experiment

An earlier mixed-prompt H100 run timed complete full-batch TF iterations while
also reporting aggregate completion throughput:

| bsz | AR | Aggregate TF | Full-batch TF | Full-batch TF/AR | Full-batch acc |
|---:|---:|---:|---:|---:|---:|
| 1 | `82.4` | `198.4` | `208.5` | `2.53x` | `6.905` |
| 8 | `558.2` | `882.7` | `1091.4` | `1.96x` | `5.383` |
| 16 | `1123.6` | `1593.7` | `2183.9` | `1.94x` | `5.906` |
| 64 | `3316.6` | `2669.5` | `5270.3` | `1.59x` | `5.023` |

At b64, drain cuts the observed TF rate by about `49%`, while the complete
full-batch path remains `1.59x` faster than AR. CUDA-event and host-clock rates
agree within `0.1%`, confirming that the timed intervals are device-bound.

Logs: `/data/home/jinzhao/nv_v2_tidar_logs/vp85_steady_direct_20260712/`.

For migration history and distributed EPLB status, see
`docs/TIDAR_AMD_HANDOFF.md`.
