# TiDAR SF multi-GPU on MI300X: data-parallel & expert-parallel status

Status of running TiDAR **single-forward (SF)** spec-decode across multiple
MI300X GPUs via vLLM's in-engine parallelism (`--data-parallel-size`,
`--enable-expert-parallel`, `--tensor-parallel-size`). Tested 2026-06-12 on
cnode-28 (8× MI300X), `vllm-smoe-amd` `jinzhao/tidar_v016`, smoediffusion
`iter_0012600`, SF `[0,4,7,11]`, FLEX_ATTENTION, captured
(FULL_DECODE_ONLY) unless noted.

## TL;DR

Every **in-engine multi-GPU** mode is currently broken for SF on this AMD
stack; single-GPU works. Scale by running **independent single-GPU
replicas** behind a load balancer.

| Config | Result |
|---|---|
| **Single GPU** (DP=1), captured | ✅ **Works** — 803 tok/s b=16 (148 b=1), accept ~5.6 |
| `--data-parallel-size 8` (with **or without** `--enable-expert-parallel`), captured | ❌ Crash at cudagraph capture — RCCL watchdog on the captured MoE all-to-all |
| `--data-parallel-size 8 --enable-expert-parallel`, eager | ❌ Inits + serves a single request, then deadlocks under concurrent load |
| `--tensor-parallel-size 8 --enable-expert-parallel` ("EP-alone"), captured | ❌ Crash even earlier — TiDAR scratch-block init-ordering bug under the TP executor; watchdog bug behind it; CCA doesn't split under TP anyway |

The shared, deep blocker is a **torch-ROCm `ProcessGroupNCCL` bug**: it
can't handle GPU collectives captured into a cudagraph. That hits *any*
multi-GPU captured config (DP all-to-all, EP all-to-all, TP all-reduce).

## What works: single GPU

`DP=1`, captured, FLEX, SF `[0,4,7,11]` → **803 tok/s** at b=16 (acc 5.63),
**148 tok/s** at b=1 (2.1× AR). No inter-GPU collective, no bug. This is the
proven path.

## Data-parallel (`--data-parallel-size 8`)

Notes that bit us:
- Offline `LLM(data_parallel_size=8)` is **rejected** by vLLM ("not
  supported for single-process usage and may hang"). Use `vllm serve` (or
  the `examples/offline_inference/data_parallel.py` multi-process pattern).
- `vllm serve` **does** honor `--attention-backend FLEX_ATTENTION`
  (`rocm.py:317 Using FlexAttention backend`); SF acceptance is healthy
  (~5.6) when FLEX is actually active. (Without FLEX, SF silently collapses
  to accept ~1.0.)
- **EP is not required to hit the all-to-all.** Even *without*
  `--enable-expert-parallel`, DP>1 on this MoE model sets up the
  `AgRsAll2AllManager` (`cuda_communicator.py:124`) and routes a token
  all-to-all across DP ranks.

### Captured: crash at cudagraph capture

During capture of the SF width (n=1360 = b=16 × 85), two things happen:

1. A dense/MoE GEMM hits `HIPBLAS_STATUS_INTERNAL_ERROR` in `hipblasLtMatmul`
   (m=256, n=1360, k=256). **Fixable** by forcing rocBLAS:
   `TORCH_BLAS_PREFER_HIPBLASLT=0` (verified — the hipBLASLt error
   disappears). But this is a *secondary* symptom; the crash persists:
2. The **RCCL collective watchdog** (`ProcessGroupNCCL.cpp:2093`) throws
   `HIP error: operation not permitted on an event last recorded in a
   capturing stream` (`hipErrorCapturedEvent`) on the captured all-to-all,
   and the NCCL watchdog terminates all ranks. **This is the real blocker.**

Confirmed for both `--enable-expert-parallel` on and off.

### Eager: deadlocks under load

With `--enforce-eager` (no capture → no watchdog-capture crash): the engine
initializes, EP shards the MoE to 2 experts/rank (16/8), and a single
low-load request returns correct output. But under concurrent load (b=16),
the DP per-step `all_reduce` (gloo) times out after 600s →
`EngineDeadError` on all ranks — the all-to-all desyncs across ranks (under
spec-decode's variable per-request behavior, ranks fall out of collective
lockstep).

## Expert-parallel without DP — "EP-alone" (`--tensor-parallel-size 8 --enable-expert-parallel`)

In vLLM, EP needs a multi-GPU group; without DP that group is TP. TP=8 + EP
= one model sharded across 8 GPUs (TP attention/dense, EP experts). It
crashes **earlier** than DP, with a different first error:

1. `RuntimeError: num_gpu_blocks not set; call _ensure_scratch_blocks AFTER
   cache init.` — a **TiDAR scratch-block init-ordering bug under the TP
   multiproc executor** (`tidar.py:192`, called at `:332`/`:451`, reads
   `cache_config.num_gpu_blocks` before the KV cache is initialized). Works
   under the single-process executor (single-GPU/DP); runs too early under
   TP's multiproc executor. **In our repo — fixable.**
2. Behind it: the **same captured-collective watchdog bug** (TP captures
   all-reduce collectives; `custom_all_reduce` was registering graph
   addresses right before the crash).
3. Caveat that makes TP pointless for this model: `WARNING: TP>1 detected,
   CCA does not support TP ... every rank will run as if TP=1` — this
   model's attention/SSM (CCA) does **not** actually shard under TP.

## Root cause (the common blocker)

torch's `ProcessGroupNCCL` watchdog records/queries a HIP event per
collective to detect hangs. On NVIDIA, torch skips this tracking for
collectives issued during CUDA-graph capture
(`currentStreamCaptureStatus`). On this **ROCm torch build, the HIP
graph-capture status isn't detected**, so the watchdog tracks the captured
collective and queries its event — illegal during capture →
`hipErrorCapturedEvent`. Env toggles do **not** disable the watchdog's
event query (tried `TORCH_NCCL_ASYNC_ERROR_HANDLING=0`,
`TORCH_NCCL_ENABLE_MONITORING=0`). This blocks any captured GPU collective,
which is what every multi-GPU mode needs.

## Recommendation

For this model (which fits on one GPU, so EP — meant for models too big for
one GPU — buys nothing), **don't use vLLM's in-engine DP/EP/TP**. Run **8
independent single-GPU vLLM replicas** (each DP=1, captured, FLEX, the
proven 803-tok/s SF path) behind a round-robin load balancer →
**~6.4k tok/s aggregate**, no inter-GPU collective, none of the above bugs.

## To actually fix in-engine multi-GPU (substantial, partly upstream)

1. **torch-ROCm `ProcessGroupNCCL`**: skip watchdog work for collectives
   issued during HIP graph capture (detect HIP capture status). C++ change,
   needs a torch rebuild. Fixes the captured-collective crash for DP, EP,
   and TP. *This is the dominant blocker.*
   - Alternative: PIECEWISE cudagraph **with the collective added to
     `splitting_ops`** so it runs eager between captured pieces. NOTE:
     PIECEWISE *alone* was tested and does NOT help — the all-to-all is not
     a split boundary, so it still lands inside a captured piece and crashes
     identically (`hipErrorCapturedEvent`). It only helps if the collective
     op is registered as a splitting op (vLLM-level work; uncertain whether
     the all-to-all is a splittable registered op; untested for SF).
2. **TiDAR scratch-block ordering** under the multiproc executor (our repo):
   defer `_ensure_scratch_blocks` until after cache init. Needed for the TP
   path (only gets it past init; #1 still required for capture).
3. **Eager all-to-all desync** under spec-decode: keep the MoE all-to-all in
   lockstep across ranks despite variable per-request spec-decode behavior.
   Needed for eager multi-GPU.

## How each was tested

```bash
# All via `vllm serve` in the jinzhao/vllm-tidar-amd image, env:
#   VLLM_SKIP_SDPA_PREINIT=1, VLLM_TIDAR_PROPOSAL_ACC_LEVELS=0,4,7,11
#   --attention-backend FLEX_ATTENTION
#   --speculative-config '{"method":"tidar","num_speculative_tokens":16,"tidar_diff_temperature":0.0}'
#   --max-model-len 10000 --max-num-seqs 16 --gpu-memory-utilization 0.85
# DP:        --data-parallel-size 8 [--enable-expert-parallel]
# EP-alone:  --tensor-parallel-size 8 --enable-expert-parallel
# captured:  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
# eager:     --enforce-eager
```
