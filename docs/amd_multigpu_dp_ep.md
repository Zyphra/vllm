# TiDAR multi-GPU on MI300X: data-parallel & expert-parallel status

Status of running TiDAR spec-decode across multiple MI300X GPUs via vLLM's
in-engine parallelism (`--data-parallel-size`, `--enable-expert-parallel`,
`--tensor-parallel-size`). `vllm-smoe-amd` `jinzhao/tidar_v016`,
smoediffusion `iter_0012600`, FLEX_ATTENTION.

> **2026-06-14 — DP+EP serving now WORKS.** The earlier "blocked, use
> replicas" verdict is superseded. The **two-forward (TF)** path serves
> DP+EP under **sustained concurrency** in two modes: **eager** (via the
> coordinate-fold deadlock fix, committed) and **captured `PIECEWISE`**
> (full capture perf, dodges the one remaining ROCm bug). `FULL_DECODE_ONLY`
> still hangs — but that's an upstream torch-ROCm captured-collective bug,
> not our code, and PIECEWISE sidesteps it. **This matters now because the
> next model is 80B SMoE: it does NOT fit one GPU, so EP is essential and
> single-GPU replicas are no longer an option.** SF×DP integration (below)
> is still unfinished; the working path today is **TF**.

## TL;DR (2026-06-14)

| Config (TF, DP=8+EP, sustained concurrency) | Result |
|---|---|
| **eager** (`--enforce-eager`) + fold | ✅ **Works** — 4 concurrent batches (8/8/12/8) clean; 9.6 tok/s b=1 |
| **captured `PIECEWISE`** + fold | ✅ **Works** — same sustained concurrency; **14.0 tok/s b=1 (1.46× eager)** |
| captured `FULL_DECODE_ONLY` + fold | ❌ **Hangs on the 2nd concurrent batch** — upstream ROCm captured-collective bug (see below) |
| single GPU (DP=1), captured | ✅ Works (reference) |

- **The eager-EP rollout deadlock is FIXED** (the coordinate **fold**,
  committed `d2f406a3f` on `jinzhao/tidar_v016`). This was the critical-path
  blocker for 80B RL rollout. Validated eager DP=2 (5/5), DP=8 (8 seq + 6
  conc), captured DP=2 (5 seq + 4 conc), captured DP=8 (8 seq), and **eager +
  PIECEWISE DP=8 under sustained concurrency** (4 batches).
- **`PIECEWISE` is the recommended rollout mode**: it retains essentially all
  of `FULL`'s capture speedup (14.0 ≈ 14.2 tok/s b=1) yet stays correct under
  concurrency, with no upstream dependency.
- **`FULL_DECODE_ONLY` is upstream-blocked on AMD**: a plain captured DP+EP
  server (TiDAR fully disabled) also hangs on the 2nd concurrent batch →
  TiDAR-independent torch-ROCm captured-collective bug. PIECEWISE avoids it
  by running the EP all-to-all eager between captured compute pieces.

## The eager-EP rollout deadlock — the coordinate FOLD (FIXED, committed)

**Symptom:** DP=8+EP, the nested TiDAR draft forward's `set_forward_context`
issued a *conditional* 2nd DP-coordinate all-reduce; idle ranks synchronized
every step but the active rank skipped it on its request-finish/boundary
step → mismatched collectives on the gloo group → `gloo::EnforceNotMet`
crash or hang. (This is "blocker #2 / the eager-EP gloo deadlock" from the
06-12 notes below.)

**Fix — the fold:** instead of a separate draft-synchronize collective, fold
the draft real-token count into ROW 5 of the **existing** outer-coordinate
all-reduce (`_run_ar` tensor `[5,dp]→[6,dp]`), which is exactly-once-per-step
by construction (the DP barrier). Each rank derives `should_run` + its
per-rank effective draft vector **locally** from the all-reduced row → lockstep
by construction, off-by-one structurally impossible. Plus a commitment
catch-all (active rank honors `should_run`) and the eagle-dummy recursion
guard (`not skip_drafter_dummy`). Core: `dp_utils.py` (`_run_ar` row 5,
`coordinate_batch_across_dp`), `tidar.py` (`_coordinate_draft_forward` reads
the fold), `gpu_model_runner.py` (`_store_tidar_draft_fold`, commitment).
**Committed `d2f406a3f`, pushed to `origin jinzhao/tidar_v016`.**

## Captured DP+EP under concurrency — FULL hangs, PIECEWISE works (2026-06-14)

With the fold in, **eager** DP=8+EP serves sustained concurrency cleanly.
**Captured `FULL_DECODE_ONLY`** serves sequentially but **hangs on the 2nd
concurrent batch**. Investigation (full 8-rank py-spy, bisection):

1. **Not a concurrency-count limit:** eager DP=8 handles 8 concurrent (and 4
   sustained batches) fine. Captured-specific.
2. **Not the TiDAR draft, and not our code:** a **plain captured DP=8+EP
   server with TiDAR/spec fully disabled** ALSO hangs on the 2nd concurrent
   batch (first batch OK, then deadlocks). → **upstream torch-ROCm
   captured-collective bug**: the EP all-to-all (AgRs = allgather +
   reduce-scatter; the only EP backend on AMD — pplx/deepep need NVSHMEM)
   captured inside a FULL graph doesn't survive idle→active re-entry across
   concurrent batches.
3. **The TiDAR symptom on top:** with the draft deadlock fixed, execution
   reaches the captured main forward's EP collective → with FULL it hangs;
   forcing the draft eager or padding the draft to a uniform composition
   turns the hang into an HSA scatter OOB (`HSA_STATUS_ERROR_EXCEPTION
   0x1016`) — same root, the captured EP collective under non-uniform
   concurrent traffic. Two opt-in draft flags staged for the draft EP path
   (`VLLM_TIDAR_DP_EAGER_DRAFT`, `VLLM_TIDAR_DP_UNIFORM_DRAFT`, both
   default-off) but they don't fix the **main**-forward captured collective.

**The fix that works: `cudagraph_mode=PIECEWISE`.** PIECEWISE captures the
compute-heavy pieces but splits at boundaries, so the EP all-to-all runs
**eager between pieces** — dodging the captured-collective bug entirely while
keeping capture perf. Validated: PIECEWISE DP=8+EP+TF survives 4 sustained
concurrent batches (8/8/12/8), healthy.

### Perf (b=1 steady-state, 256 tok, DP=8+EP+TF, cnode-11)

| mode | tok/s | vs eager | sustained concurrency |
|---|---|---|---|
| `--enforce-eager` | 9.6 | 1.0× | ✅ |
| **`PIECEWISE`** | **14.0** | **1.46×** | ✅ |
| `FULL_DECODE_ONLY` (seq only) | 14.2 | 1.48× | ❌ hangs |

PIECEWISE ≈ FULL on perf and is the only captured mode correct under
concurrency → **use PIECEWISE for the rollout.** (b=1 = 1 active rank of 8;
the 1.46× is the per-request capture step-rate benefit, not aggregate
throughput.)

### Run it

```bash
# image jinzhao/vllm-tidar-amd:latest, env:
#   VLLM_SKIP_SDPA_PREINIT=1  VLLM_ATTENTION_BACKEND=FLEX_ATTENTION
#   VLLM_TIDAR_TWO_FORWARD=1  VLLM_TIDAR_PROPOSAL_ACC_LEVELS=0,4,7,11
#   TORCH_BLAS_PREFER_HIPBLASLT=0  VLLM_ENGINE_READY_TIMEOUT_S=1800
vllm serve <ckpt> \
  --data-parallel-size 8 --enable-expert-parallel \
  --disable-nccl-for-dp-synchronization \
  --attention-backend FLEX_ATTENTION \
  --speculative-config '{"method":"tidar","num_speculative_tokens":16,"tidar_diff_temperature":0.0}' \
  --dtype bfloat16 --max-model-len 10000 --max-num-seqs 16 \
  --gpu-memory-utilization 0.65 \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}'   # <- the rollout mode
# eager fallback: replace the compilation-config with --enforce-eager (gpu-mem 0.6)
```

Infra gotchas (this cluster): `FULL` capture is memory-heavy under DP=8 — use
`gpu-mem ≤0.70` (0.65 for PIECEWISE) or it rc=137 OOMs during the 33-size
capture. Nodes get silently occupied without a scheduler reservation
(`check_gpus_ids` "free" ≠ actually free — verify with `rocm-smi`). Image tar
to load on a fresh node: `/shared/home/jinzhao/workspace/tidar/vllm-tidar-amd.tar`
(72G, ~2.5–4 min `docker load`). Use `--enforce-eager` (NOT
`cudagraph_mode=NONE`, which keeps TiDAR's large capture-size warmup and trips
`assert num_tokens <= max_num_batched_tokens`).

## Remaining work

1. **Multi-node DP+EP** (the real 100-node blocker for 80B): inter-node RCCL
   for the expert all-to-all across machines, in PIECEWISE. Not yet started.
2. **SF×DP integration** (below) — still unfinished; TF is the working path.
3. **Upstream:** the FULL captured-collective bug is torch-ROCm; PIECEWISE is
   the workaround, so this is low priority unless FULL perf is ever needed.

---

## (Reference, 2026-06-12) SF multi-GPU status

The below predates the fold + PIECEWISE findings and concerns the
**single-forward (SF)** path, which is still unfinished on DP. Kept for the
SF×DP bug stack and the capture-crash root cause (both still valid).

### What works: single GPU

`DP=1`, captured, FLEX, SF `[0,4,7,11]` → **803 tok/s** at b=16 (acc 5.63),
**148 tok/s** at b=1 (2.1× AR). No inter-GPU collective, no bug.

### Captured cudagraph crash — root cause + the FIX (flags, no rebuild)

The captured crash was a `ProcessGroupNCCL` watchdog `HIP error: operation
not permitted on an event last recorded in a capturing stream`
(`hipErrorCapturedEvent`). The MoE all-to-all is **not** the culprit (vLLM
routes it through pynccl). The lone leak was **one direct c10d collective**:
`dp_utils.py` `dist.all_reduce(..., group=get_dp_group().device_group)`, the
per-step DP coordination, on the DP **GPU** group via torch c10d, bypassing
pynccl → the one collective the watchdog tracks.

**Fix (no code change, no rebuild):**
- **`--disable-nccl-for-dp-synchronization`** — routes that all-reduce to the
  **CPU/gloo** group (not captured → no watchdog). The real fix.
- **`TORCH_BLAS_PREFER_HIPBLASLT=0`** — clears a separate
  `HIPBLAS_STATUS_INTERNAL_ERROR` on the SF-width GEMM whose unfused fallback
  also records a capture-illegal event.

### SF×DP integration bug stack (still blocks SF serving)

SF inflates each request's tokens to `verify_len + P·(K+1)` (variable per
rank), violating vLLM DP assumptions:
1. **`flex_attention.py _offsets_to_doc_ids_tensor`** → `repeats can not be
   negative` — non-monotonic `query_start_loc` under DP.
2. **`dp_utils.py _synchronize_dp_ranks`** → `assert should_attempt_dp_padding
   == should_dp_pad` — ranks disagree on DP-padding under SF.
TF has no such inflation (monotonic offsets) → TF is the working path; SF×DP
remains a multi-bug effort.

### EP-alone (`--tensor-parallel-size 8 --enable-expert-parallel`)

TP=8+EP crashes at init: TiDAR scratch-block init-ordering bug under the TP
multiproc executor (`tidar.py` reads `cache_config.num_gpu_blocks` before KV
cache init). Also `CCA does not support TP` → no real model split. Not the
path for this work.
