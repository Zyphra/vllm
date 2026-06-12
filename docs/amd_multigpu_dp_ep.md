# TiDAR SF multi-GPU on MI300X: data-parallel & expert-parallel status

Status of running TiDAR **single-forward (SF)** spec-decode across multiple
MI300X GPUs via vLLM's in-engine parallelism (`--data-parallel-size`,
`--enable-expert-parallel`, `--tensor-parallel-size`). Tested 2026-06-12 on
cnode-28 (8× MI300X), `vllm-smoe-amd` `jinzhao/tidar_v016`, smoediffusion
`iter_0012600`, SF `[0,4,7,11]`, FLEX_ATTENTION, captured
(FULL_DECODE_ONLY) unless noted.

## TL;DR

Single-GPU works. Captured multi-GPU **now gets past cudagraph capture**
with two flags (below), but **serving is then blocked by a stack of
TiDAR-SF × vLLM-DP integration bugs** — SF's token inflation breaks several
data-parallel invariants. For multi-GPU **today**, run **independent
single-GPU replicas** behind a load balancer, or **DP=8 no-EP eager** for a
single integrated endpoint.

| Config | Result |
|---|---|
| **Single GPU** (DP=1), captured | ✅ **Works** — 803 tok/s b=16 (148 b=1), accept ~5.6 |
| `--data-parallel-size 8` (no EP), **eager** | ✅ **Works under load** — ~1187 tok/s aggregate (eager, thin batch). Integrated endpoint. |
| `--data-parallel-size 8 [--enable-expert-parallel]`, captured | ⚠️ **Captures OK with the two flags below**, but first request crashes on a DP+SF integration bug (Flex offsets / DP-pad disagreement). Does not serve yet. |
| `--data-parallel-size 8 --enable-expert-parallel`, eager | ❌ Serves a single request, deadlocks under load (EP all-to-all desync) |
| `--tensor-parallel-size 8 --enable-expert-parallel` ("EP-alone"), captured | ❌ Crashes at init — TiDAR scratch-block ordering bug under the TP executor; CCA doesn't split under TP anyway |

## What works: single GPU

`DP=1`, captured, FLEX, SF `[0,4,7,11]` → **803 tok/s** at b=16 (acc 5.63),
**148 tok/s** at b=1 (2.1× AR). No inter-GPU collective, no bug. The proven
path; scale it with independent replicas.

## Data-parallel (`--data-parallel-size 8`)

Notes that bit us:
- Offline `LLM(data_parallel_size=8)` is **rejected** by vLLM ("not
  supported for single-process usage and may hang"). Use `vllm serve` (or
  the `examples/offline_inference/data_parallel.py` multi-process pattern).
- `vllm serve` **does** honor `--attention-backend FLEX_ATTENTION`
  (`rocm.py:317 Using FlexAttention backend`); SF acceptance is healthy
  (~5.6) when FLEX is active. (Without FLEX, SF silently collapses to accept
  ~1.0 — make sure the flag isn't dropped.)

### Captured cudagraph crash — root cause + the FIX (flags, no rebuild)

The captured crash is a `ProcessGroupNCCL.cpp:2093` watchdog throwing
`HIP error: operation not permitted on an event last recorded in a
capturing stream` (`hipErrorCapturedEvent`). The non-obvious part: **the
MoE all-to-all is NOT the culprit** — vLLM's CUDA communicator already
routes `all_gather(v)` / `reduce_scatter(v)` / `all_reduce` through
**pynccl** (no watchdog, captures fine; RCCL loads, pynccl enabled). The
lone leak is **one direct c10d collective**:

- **`vllm/v1/worker/dp_utils.py:55` — `dist.all_reduce(tensor, group=get_dp_group().device_group)`**,
  the per-step DP coordination of token-counts / ubatch / padding. It runs
  on the DP **GPU** group via torch c10d, bypassing pynccl, so it's the one
  collective the watchdog tracks → captured-event crash.

**Fix (verified to clear the crash — reaches "ready", all 8 ranks):**
- **`--disable-nccl-for-dp-synchronization`** — routes that all-reduce to
  the **CPU/gloo** group (not a GPU collective → not captured → no
  watchdog). This is the real fix; **no code change, no torch rebuild.**
- **`TORCH_BLAS_PREFER_HIPBLASLT=0`** — clears a *separate* symptom: a
  `HIPBLAS_STATUS_INTERNAL_ERROR` on the SF-width GEMM (m=256, n=1360,
  k=256) whose unfused fallback also records a capture-illegal event.

### Past capture: TiDAR-SF × DP integration bug stack (blocks serving)

With capture cleared, the first real request crashes. SF inflates each
request's tokens to `verify_len + P·(K+1)` (variable per rank), which
violates several vLLM data-parallel assumptions — surfacing as different
errors depending on DP size / batch distribution (idle ranks run dummy
batches with SF-inflated shapes):

1. **`flex_attention.py:44 _offsets_to_doc_ids_tensor`** → `RuntimeError:
   repeats can not be negative`. A non-monotonic `query_start_loc` (negative
   `offsets.diff()`) under DP (seen at DP=8).
2. **`dp_utils.py:146 _synchronize_dp_ranks`** → `assert
   should_attempt_dp_padding == should_dp_pad`. DP ranks **disagree on
   whether to DP-pad** under SF, violating vLLM's "all ranks agree on
   padding" invariant (seen at DP=2, idle rank).

These are integration bugs in vLLM's DP coordination / padding /
Flex-metadata code — TiDAR SF was never validated against the internal DP
path. Fixable, but a multi-bug effort (each fix so far revealed the next),
not a quick patch.

### Eager: works no-EP; deadlocks +EP under load

- **DP=8 no-EP, eager:** ✅ serves the full load, no deadlock, ~1187 tok/s
  aggregate (eager, thin per-rank batch), FLEX+SF active. A working
  integrated endpoint today.
- **DP=8 +EP, eager:** initializes, serves a single request, but under
  concurrent load the DP all-reduce (gloo) times out 600s →
  `EngineDeadError` — the EP all-to-all desyncs across ranks under
  spec-decode's variable per-request behavior.

## Expert-parallel without DP — "EP-alone" (`--tensor-parallel-size 8 --enable-expert-parallel`)

In vLLM, EP needs a multi-GPU group; without DP that group is TP. TP=8 + EP
crashes at **init**, before capture:

1. `RuntimeError: num_gpu_blocks not set; call _ensure_scratch_blocks AFTER
   cache init.` — a **TiDAR scratch-block init-ordering bug under the TP
   multiproc executor** (`tidar.py:192`, called at `:332`/`:451`; reads
   `cache_config.num_gpu_blocks` before the KV cache is initialized). Works
   under the single-process executor (single-GPU/DP). **In our repo —
   fixable.** Past it is untested.
2. Caveat that makes TP pointless for this model: `WARNING: TP>1 detected,
   CCA does not support TP ... every rank will run as if TP=1` — this
   model's attention/SSM (CCA) does **not** actually shard under TP, so
   you'd pay all-reduce cost without real model splitting.

## Recommendation

This SMoE fits on one GPU, so EP (which shards experts for models too big
for one GPU) buys nothing. For multi-GPU SF **today**:
- **Independent single-GPU captured replicas** behind a round-robin load
  balancer → **~6.4k tok/s aggregate**, no inter-GPU collective, none of the
  above bugs. Fastest, lowest risk.
- **DP=8 no-EP, eager** → one integrated endpoint, ~1187 tok/s (works).

Pursuing a single **captured in-engine DP/EP** server means committing to
the DP+SF integration work below.

## To get captured in-engine DP/EP serving (remaining work)

1. ✅ **Capture crash** — solved with `--disable-nccl-for-dp-synchronization`
   + `TORCH_BLAS_PREFER_HIPBLASLT=0` (above). No torch rebuild needed; the
   all2all was already pynccl. *(Supersedes the earlier belief that this
   required a torch-ROCm `ProcessGroupNCCL` patch.)*
2. **DP+SF integration** (our repo, multi-bug): make SF's inflated token
   layout consistent with vLLM's DP coordination — monotonic
   `query_start_loc` for idle/dummy + padded DP batches
   (`_offsets_to_doc_ids_tensor`), and a uniform `should_attempt_dp_padding`
   across ranks (`_synchronize_dp_ranks`). Expect more spots behind these.
3. **Eager all-to-all desync** under spec-decode (for the eager +EP path):
   keep the MoE all-to-all in lockstep despite variable per-request behavior.
4. **TiDAR scratch-block ordering** under the multiproc executor (for the TP
   path): defer `_ensure_scratch_blocks` until after cache init.

## How each was tested

```bash
# vllm serve in the jinzhao/vllm-tidar-amd image, env:
#   VLLM_SKIP_SDPA_PREINIT=1, VLLM_TIDAR_PROPOSAL_ACC_LEVELS=0,4,7,11
#   --attention-backend FLEX_ATTENTION
#   --speculative-config '{"method":"tidar","num_speculative_tokens":16,"tidar_diff_temperature":0.0}'
#   --max-model-len 10000 --max-num-seqs 16 --gpu-memory-utilization 0.85
# DP:           --data-parallel-size 8 [--enable-expert-parallel]
# EP-alone:     --tensor-parallel-size 8 --enable-expert-parallel
# captured:     --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
# eager:        --enforce-eager
# capture fix:  --disable-nccl-for-dp-synchronization  + env TORCH_BLAS_PREFER_HIPBLASLT=0
```
