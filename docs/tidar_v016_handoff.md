# TiDAR vLLM v0.16 Port — Handoff

**Branch:** `jinzhao/tidar_v016` @ `c1e8c9f9c`
**Repo:** `git@github.com:Zyphra/Zvllm.git`
**Node tested:** vp-dgx-89 (147.68.0.89) — node 2 was contended, all post-2026-06-03 measurements use idle node 89
**Env:** `/data/home/jinzhao/workspace/tidar/Zvllm-v016/.venv-v016`
**Date:** 2026-06-03

## What works

| Mode | Status | Notes |
|---|---|---|
| AR eager | ✅ | Reference, ~30 tok/s on iter_0012000 |
| AR FULL captured | ✅ | Standard vLLM path |
| SF eager | ✅ | Coherent |
| SF PIECEWISE captured | ✅ | Coherent |
| **SF FULL_DECODE_ONLY captured** | ✅ | **Primary path** — 176/361/619 tok/s b=1/8/16 |
| TF eager | ✅ | ~20 tok/s, coherent |
| TF PIECEWISE captured | ✅ (opt-in) | **13.9 tok/s b=1, n=3, max_tokens=200**, coherent math reasoning end-to-end. Requires `VLLM_TIDAR_ROUTER_PAD=1` — same env var that fixed TF FULL captured. Without it, process segfaults in `cudaGraphLaunch → CUDAGraph::replay()`. Same root cause as TF FULL (buggy CUTLASS align1 kernel from the unaligned router output stride). |
| **TF FULL_DECODE_ONLY captured** | ✅ (opt-in) | **228 tok/s n=10, 264 tok/s n=3** on smoediffusion iter_0012000, AIME, K=16, idle vp-dgx-89 (3-run sigma~0.5%). v0.16 now at 94.8% of v0.15 (was 70% before c1e8c9f9c's disable_padded_drafter_batch fix). Requires `VLLM_TIDAR_ROUTER_PAD=1` + `VLLM_TIDAR_FA_NO_SPLITS=1` + `VLLM_ATTENTION_BACKEND=FLASH_ATTN`. v0.15 reference n=10 = 240, n=3 = 278. See "v0.16 vs v0.15 perf gap" below for the recovery story. |
| `vllm serve` DP=8 | Not retested | v0.15 handoff has working command; should port over once TF captured is fixed (or stay SF-only) |

## Quickstart

```bash
ssh vp-dgx-51   # 147.68.0.51 (fallback: node 44)
cd /data/home/jinzhao/workspace/tidar/Zvllm-v016
git fetch origin && git checkout jinzhao/tidar_v016 && git pull --ff-only
source .venv-v016/bin/activate

export VLLM_ATTENTION_BACKEND=FLEX_ATTENTION
export VLLM_TIDAR_PROPOSAL_ACC_LEVELS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
# (SF needs FLEX backend; the env var fallback is silent if missing — see
#  reference_zvllm_inference_env memory.)

# Perf knob — set to 1 to wrap the SMoE cat()+FusedMoE call in an opaque
# custom op, hiding the cat() from inductor and forcing probs to
# materialize once. +45% on TF FULL b=1 dense, +54% on SF sparse P;
# ~neutral (-3%) on SF dense P b=3. See "Custom-op-wrapped MoE call"
# below. Default OFF.
# export VLLM_TIDAR_SMOE_MOE_OP=1
```

### Run SF FULL captured (the supported path)

```python
from vllm import LLM, SamplingParams
ckpt = "/data/checkpoints/smoediffusion_128k_64node-hf/iter_0012000"
llm = LLM(
    model=ckpt, dtype="bfloat16", gpu_memory_utilization=0.85,
    max_model_len=4096, max_num_seqs=16, enforce_eager=False, seed=0,
    swap_space=16.0, attention_backend="FLEX_ATTENTION",
    speculative_config={
        "method": "tidar",
        "num_speculative_tokens": 16,
        "tidar_diff_temperature": 0.0,
    },
    compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
)
out = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=600))
```

`cudagraph_copy_inputs=True` is auto-forced under TiDAR (don't override).

### Run TF FULL captured (opt-in via env var)

```bash
export VLLM_TIDAR_TWO_FORWARD=1
export VLLM_TIDAR_ROUTER_PAD=1   # required (router padding to dodge buggy cutlass align1 kernel)
export VLLM_TIDAR_FA_NO_SPLITS=1 # required for max accept (forces FA num_splits=1)
export VLLM_ATTENTION_BACKEND=FLASH_ATTN   # FA, not FLEX — FLEX is intrinsically bad for TF
```

```python
llm = LLM(
    model=ckpt, ..., enforce_eager=False,
    speculative_config={...},
    compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
    kernel_config={"enable_flashinfer_autotune": False},   # +8% tok/s (146 -> 158); see "FlashInfer autotune" below
)
```

**158 tok/s @ accept 5.71 at b=1** on iter_0012000 (n=3, AIME). 4.4x over v0.16 FLEX baseline (~36 tok/s).
The padded-router workaround is dormant without `VLLM_TIDAR_ROUTER_PAD=1`, so SF
(which doesn't need the fix) stays on the baseline path by default. `NO_SPLITS=1` is
TF-only — gated on `VLLM_TIDAR_TWO_FORWARD=1`, doesn't affect SF.

### Run TF eager (fallback)

```python
import os
os.environ["VLLM_TIDAR_TWO_FORWARD"] = "1"

llm = LLM(
    model=ckpt, ..., enforce_eager=True,
    speculative_config={...},
)
```

### Benchmark + profile

`scripts/bench_tidar.py` (committed) drives all SF/TF accept + perf numbers in this doc. Env knobs:
- `BENCH_K/B/N/MT` — speculative K, batch, num prompts, max output tokens
- `BENCH_MODE` — `ar` / `tf` / `sf`
- `BENCH_EAGER` — 1 to disable compile + cudagraph
- `BENCH_CG` — `NONE` / `PIECEWISE` / `FULL_DECODE_ONLY`
- `BENCH_GPU_MEM`, `BENCH_MML`, `BENCH_MNBT` — memory util, max_model_len, max_num_batched_tokens
- `BENCH_FI_AUTOTUNE_OFF=1` — disable FlashInfer autotune (+8% tok/s for TF, no accept change)
- `BENCH_O=0` — force optimization_level=O0 (diagnostic)
- `BENCH_COMPILE_MODE`, `BENCH_COMBO_OFF` — compile-mode diagnostic knobs

`scripts/profile_sf.py` (committed) — one-prompt torch.profiler capture of SF FULL captured for per-component breakdown.

## What's broken

### TF FULL captured — FIXED in `43c0fa08e` (opt-in via `VLLM_TIDAR_ROUTER_PAD=1`)

**Root cause:** `SMoERouter`'s final linear projects `D=mlp_expansion -> num_experts (=17 with MOD)`. That output stride (17) is not a multiple of 8, so under captured-cudagraph replay cublas selects `cutlass_75_tensorop_s1688gemm_bf16_64x64_tn_align1` — a kernel that **writes 1+ bytes past its output allocation**. In eager mode the regular allocator's padding hides the OOB; the cudagraph memory pool's tight fit exposes it. Verified with `compute-sanitizer --tool=memcheck`: every TF FULL crash hits this exact kernel at `+0x5430`, OOB by 1 byte from a 1.18 GB-adjacent allocation.

**Fix:** project against a padded `D -> E_padded` weight (E=17 → E_padded=24, the next multiple of 8). cublas picks a different (correct) sm_90 kernel for the aligned output stride. Slice back to E columns for downstream code. Padded weight buffer is filled lazily on first forward from `router_mlp[4].weight`, no checkpoint format change needed.

**Why opt-in:** SF FULL captured doesn't hit the buggy kernel at its runtime shapes, and the padded matmul has small overhead at b=16 in noisy benchmarks. SF stays on the baseline path by default.

**Validation (smoediffusion iter_0012000, b=1, max_tokens=1000 after warmup):**
- `VLLM_TIDAR_TWO_FORWARD=1` `VLLM_TIDAR_ROUTER_PAD=1` `cudagraph_mode=FULL_DECODE_ONLY` → **23.5 tok/s, coherent math reasoning end-to-end**. Was 0 tok/s (engine crashed at step 3 = ~5 tokens).
- Without `VLLM_TIDAR_ROUTER_PAD=1`, TF FULL still crashes (verified — the fix is dormant when the flag is off).

**Drafter-capture: SHIPPED via warmup-time capture (`0001ddc4d`).**

The TF drafter forward is now captured at warmup, alongside the standard verifier captures. Step rate at b=1: **27 tok/s** (vs ~4-7 tok/s eager drafter), a 6-7x lift purely from capturing the drafter graph. The verifier graph is captured by the standard warmup loop; the drafter graph is captured by a new `TiDARProposer.warmup_capture_drafter_graphs()` hook called from `gpu_model_runner.capture_model` after the standard loop, inside the same `graph_capture(device=)` context.

How it composes:
- **Dispatcher** registers a parallel `BatchDescriptor(is_drafter_pass=True)` FULL key per shape under TiDAR (same loop v0.15 has gated on `method == "tidar"`). It also FILTERS those keys out of `get_capture_descs()` — the standard `_dummy_run` loop builds verifier metadata, which would bake the wrong write-side pointer if it captured the drafter key.
- **Runner** rebinds `self.drafter.model = self.model` after the `CUDAGraphWrapper` wrap, so drafter forwards dispatch into the wrapper. After the standard capture loop, it calls `drafter.warmup_capture_drafter_graphs()`.
- **Proposer**'s warmup hook builds a dummy `CommonAttentionMetadata` pointing at runner persistent buffers (`runner.input_ids`, `runner.positions`, `runner.query_start_loc`, `runner.seq_lens`, FA group's `slot_mapping`), with `seq_lens` filled to `max_model_len` so FA's kernel-baked `max_seqlen_k` is large enough for any runtime sequence. Sets drafter overrides (`state_indices_tensor_write_override = block_table[:, 1]`, `cca_drafter_pass=True`), builds attention metadata via the standard builders, then calls `self.model(...)` under `set_forward_context(... batch_descriptor=is_drafter_pass=True ...)`. The wrapper finds no entry for that descriptor and captures the drafter graph on the non-default capture stream.

**Caveat — separate v0.16 TF accept regression:** at b=1 on smoediffusion iter_0012000, TF mode has 0% accept rate regardless of capture mode or backend (also reproducible with TF EAGER, no patches). The 27 tok/s figure is captured-step-rate at 0% accept; once the TF accept regression is identified and fixed the captured drafter will compound the spec speedup.

Reproducing the captured TF speedup:
- TF EAGER (no captured graphs): `BENCH_EAGER=1 BENCH_CG=NONE BENCH_MODE=tf` → ~4 tok/s (FA backend) / ~14 tok/s (FlexAttention, slightly different perf characteristics)
- TF FULL captured, eager drafter (no patches): ~7-14 tok/s
- **TF FULL captured + warmup drafter (`0001ddc4d`): 27 tok/s b=1, FA backend**

Earlier dead-end attempts (kept for record because the search illuminates what to NOT try):
- The lazy-capture-in-propose route (`07fd22287`, `49e4b1841` doc commits) captured the drafter graph successfully but appeared broken — output was garbage. That misdiagnosis came from the underlying TF accept regression masquerading as a capture-correctness bug.
- The first warmup attempt used `seq_lens=K+1=17` for the dummy metadata, baking too-small `max_seqlen_k` into the FA kernel. Fixed by filling to `max_model_len`.
- The first warmup attempt also passed CCA's block_table as `CommonAttentionMetadata.block_table_tensor`. Fixed by passing FA's; CCA's draft slot is plumbed separately via the override.

**2026-06-01 second attempt log (also reverted):** retried the v0.15-style minimal port to v0.16 — pair the runner rebind `self.drafter.model = self.model` (added after the CUDAGraphWrapper wrap) with the dispatcher registering an `is_drafter_pass=True` FULL key per shape (same loop v0.15 has under `method == "tidar"`), then bracket the lazy capture in propose() with `graph_capture(device=...)`. Also filtered `is_drafter_pass=True` entries out of `get_capture_descs()` so the warmup loop doesn't pre-capture with verifier metadata baked.

Result: 0% acceptance and 1.29 tok/s b=1 dense P (vs 14.25 baseline). The captured drafter graph replays without crashing but produces garbage tokens, so the verifier rejects every draft. So whatever metadata the captured drafter is reading at replay isn't the buffer the proposer is writing at runtime — even though both CCAAttentionMetadataBuilder's `_has_initial_states_p_buf` / `_query_start_loc_p_buf` and the runner-side `input_ids` / `positions` / `query_start_loc` / `seq_lens` / `slot_mapping` ARE in persistent buffers in v0.16's tidar.py (the d768183 pinning was ported). Something else still rebuilds a fresh tensor each call — most likely either (a) the per-layer-attn metadata dict itself (rebuilt per call, with new FA/CCA metadata object IDs), or (b) one of the CCA stash buffers (`stash_xb`, `qkv_packed3_prefill`) that the proposer constructs each step in `_build_draft_inputs`. The captured graph baked the FIRST call's tensor IDs and reads stale data on replay.

Pointers for the next attempt: nsys diff between v0.15 drafter-capture and v0.16 attempt; or `compute-sanitizer --tool=initcheck` + memcheck on the drafter graph's first replay to find the unpinned read source.

**2026-06-01 third attempt log (Approach B, also reverted):** Implemented the bigger refactor the handoff originally recommended — a `TiDARProposer.warmup_capture_drafter_graphs()` hook that builds dummy drafter metadata + inputs inside `gpu_model_runner.capture_model`'s `graph_capture(device=)` context, AFTER the standard verifier captures complete. This avoids the per-call `graph_capture(device=)` bracket in `propose()` entirely; the drafter graph is captured at warmup with the correct metadata in scope, then runtime `propose()` just dispatches and replays.

Implementation:
- `vllm/v1/spec_decode/tidar.py`: new `warmup_capture_drafter_graphs()` + `_warmup_capture_one_drafter_shape()` methods. Constructs dummy seq_lens=K+1, query_start_loc=[0,K+1,..], slot_mapping=[0..n-1], etc. — pointing at runner persistent buffers — then sets drafter overrides (`state_indices_tensor_write_override = block_table[:, 1]`, `cca_drafter_pass=True`) and calls `self.model(...)` with `BatchDescriptor(is_drafter_pass=True)`. Wrapped in `torch.inference_mode()` to match `_dummy_run`'s decorator (FlexAttention metadata builder inplace-mutates persistent inference tensors that fail outside that mode).
- `vllm/v1/cudagraph_dispatcher.py`: register `is_drafter_pass=True` key per shape; filter from `get_capture_descs()` so the standard loop doesn't try to capture it with verifier metadata.
- `vllm/v1/worker/gpu_model_runner.py`: runner rebind + call the new hook after the regular capture loop inside `graph_capture(device=)`.

Result: drafter graph captures successfully at warmup (logs show "TiDAR Tier 3: warmup-capturing drafter graph at num_tokens=17 (batch_size=1, K+1=17)"). Generation runs at ~13 tok/s (vs 14.25 eager baseline), **still 0% acceptance**. Same garbage-token symptom as the lazy-capture attempt — confirms the bug is NOT about *when* the drafter graph gets captured.

This rules out my earlier hypotheses (non-default stream, graph_capture context-manager state, late-binding warmup vs lazy). The captured drafter graph itself is structurally wrong even with all the proper metadata in scope at capture. Most likely cause: **FlexAttention bakes seq-length-dependent Python ints into the captured kernels** (e.g., `seq_lengths=(num_actual_tokens, total_cache_tokens)` on the block_mask), so the captured graph attends only over the dummy seq_lens=K+1 KV slice instead of the real runtime sequence. v0.15 used FlashAttention (FA backend) for the drafter, which doesn't have this property; v0.16's `VLLM_ATTENTION_BACKEND=FLEX_ATTENTION` is the only viable backend for SF, but FlexAttention's per-step seq-len bakeness may be incompatible with TF drafter capture.

To progress further: try the warmup capture path with `VLLM_ATTENTION_BACKEND=FLASH_ATTN` (the v0.15 backend) to test the FlexAttention-bakeness hypothesis. If FA works, the right shipping path is to capture the drafter graph against the FA backend even when the verifier uses FlexAttention — or accept the eager-drafter ceiling for TF mode (SF stays the supported path).

### TF PIECEWISE captured — FIXED by `VLLM_TIDAR_ROUTER_PAD=1` (same flag as TF FULL)

Confirmed root cause is identical to TF FULL captured: the buggy `cutlass_75_tensorop_s1688gemm_bf16_64x64_tn_align1` kernel writes 1+ bytes past its output during cudagraph replay, on the SMoERouter's `D=2048 → E=17` final projection. Under PIECEWISE the OOB lands in a place the runtime can't recover from, so it surfaces as a process **segfault** (`cudaGraphLaunch → at::cuda::CUDAGraph::replay()`) rather than the `cudaErrorIllegalAddress` exception you get under FULL.

The router-pad workaround introduced in `43c0fa08e` (project against a padded `D → E_padded=24` weight to dodge the buggy kernel) applies identically to PIECEWISE — no code change needed beyond setting the env var.

**Validation (smoediffusion iter_0012000, K=16, b=1, max_tokens=200, n=3, TF mode):**
- `VLLM_TIDAR_TWO_FORWARD=1` `VLLM_TIDAR_ROUTER_PAD=1` `cudagraph_mode=PIECEWISE` → **13.88 tok/s aggregate, all 3 prompts produce coherent math reasoning** (`finish=length`, mean acceptance length 5.08/3.77/3.51 across SpecDecoding windows).
- Without `VLLM_TIDAR_ROUTER_PAD=1`: process segfaults at first replay of the captured subgraph (post warmup, in the first user `llm.generate()` call). C-stack: `cudaGraphLaunch → at::cuda::CUDAGraph::replay`.

PIECEWISE captures 6 subgraphs (vs FULL's 6 too in this config) and reuses them across drafter + verifier passes, so the fix lands once and benefits both forwards.

### TF captured drafter accept: no real drift (earlier 22% gap was a measurement artifact)

Earlier handoff drafts claimed a ~22% accept drop from TF EAGER (cited 7.29) to TF FULL
captured (5.71). The 7.29 figure was a single outlier SpecDecoding window from an n=1
bench. Fresh n=3 benches on the same checkpoint show eager and captured are on par:

| TF FA config | n | true-mean accept | tok/s |
|---|---:|---:|---:|
| EAGER (original "7.29" log) | 1 | 6.70 (windows: 6.10, 7.29) | 31.6 |
| **EAGER (fresh n=3)** | 3 | **5.14** | 22.1 |
| FULL captured + NO_SPLITS=1 | 3 | 5.71 | 146.5 |
| FULL captured + NO_SPLITS=1 + FI_autotune_off | 3 | 5.69 | 158.0 |
| FULL captured + NO_SPLITS=1 + drafter cg disabled | 3 | 5.70 | 42.5 |
| SF FLEX EAGER | 3 | 7.01 | 41.4 |
| SF FLEX FULL_DECODE_ONLY | 3 | 7.56 | 168.2 |

Captured TF FA accept (5.71) actually **slightly exceeds** fresh n=3 EAGER (5.14) on these
prompts. There is no structural captured-mode drift to fix.

Things tested while chasing the phantom drift (all ruled out, but recorded for record):
- inductor codegen (mode=NONE)
- dynamo tracing (DYNAMO_TRACE_ONCE)
- `combo_kernels=True` inductor fusion
- DeepGEMM JIT warmup (`VLLM_USE_DEEP_GEMM=0`)
- optimization_level pass_config (O0 all-False fuses)
- `VLLM_TIDAR_DISABLE_DRAFTER_CAPTURE=1` (forces drafter eager under captured verify; accept unchanged at 5.70, tok/s drops to 42)
- `flashinfer_autotune` — no accept impact, BUT see below for the +8% tok/s win

Note: EAGER and captured generate slightly different greedy sequences (n_out=662/579/683
vs 605/957/650 on the first 3 AIME prompts). That divergence is real bf16 numerics drift
— captured-mode kernels round differently than eager — but it doesn't degrade accept
rate; both modes converge to the correct answer (`\boxed{70}` etc.) just via different
token sequences.

### FlashInfer autotune perf gotcha (TF, +8% tok/s when disabled)

`kernel_config.enable_flashinfer_autotune=True` is the default for `optimization_level >= O1`
(i.e., any non-`enforce_eager` run). It runs `_dummy_run(max_num_batched_tokens, is_profile=True)`
to benchmark FlashInfer op implementations and cache the best per-shape. The
autotune is at MNBT=2048-4096; the runtime drafter+verify forwards are 17-34 tokens.
The selected ops are mis-sized — no accept impact, but ~8% throughput on the table.

Set `kernel_config={"enable_flashinfer_autotune": False}` in the LLM config to recover
the 8%. No accept change, just kernel selection at small M. Could be wired as a default
for TiDAR mode in a follow-up; for now it's a one-line knob in the user config.

## v0.16 vs v0.15 perf gap (TF FA b=1, K=16)

Apples-to-apples bench (n=3 AIME prompts, max_tokens=1500, AIME25 thinking-off,
T_AR=0):

### Full v0.15 vs v0.16 matrix (idle node 89, n=10 mt=2000)

| mode  | batch | v0.15 | v0.16 head | ratio | notes |
|---|---:|---:|---:|---:|---|
| AR (no spec)    | 1 |   65 |  101 | **156%** | v0.16 forward is faster |
| **TF (FA)**     | 1 |  240 |  228 |   95%  | residual 5% structural |
| TF (FA)         | 4 |  566 |  608 | **107%** | v0.16 faster |
| TF (FA)         | 8 | 1137 |  856 |   75%  | **v0.15 lead at b=8 — scheduler/bookkeep scaling** |
| **SF (FLEX)**   | 1 |  ~75 |  223 | **297%** | v0.15 SF crashes at short bench; my n=10 measurement of 75 may not be apples-to-apples |
| SF (FLEX)       | 8 |  319 |  634 | **199%** | v0.16 SF much faster |

(b=4 TF and AR runs are 2-run mean; b=1 is 5-run mean; b=8 is 2-run.)

### Where the remaining gaps are

- **TF b=1**: residual ~5%. v0.16 has small fixed per-step overhead (engine subprocess IPC, scheduler bookkeeping) that v0.15 doesn't.
- **TF b=8**: ~25% gap. v0.16's overhead scales linearly with batch in some places (CPU-list path for sampled_token_ids, per-req scheduler iteration). The c1e8c9f9c fix is still net-positive at b=8 (vs the use_gpu_toks=True alternative which gave 550 tok/s — so dpdb=True wins by +56% at b=8) but doesn't close to v0.15.
- **Everywhere else**: v0.16 is at par or significantly faster.

To close the TF b=8 gap would need batch-scaling-aware optimizations — out of session scope. The dominant b=1 case is closed.

#### Original cumulative wins table (kept for history)

| config | v0.15 | v0.16 head | ratio |
|---|---:|---:|---:|
| n=10 mt=2000 b=1 (idle node 89) | **240** | **228** | 95% |
| n=3 mt=1500 b=1 | **278** | **264** | 95% |
| n=10 mt=2000 b=4 | **566** | **608** | **107% (v0.16 FASTER)** |
| AR mode (no spec decode), n=10 b=1 | 64.8 | 101 | **156% (v0.16 FASTER)** |
| v0.16 head minus c1e8c9f9c (use_gpu_toks=True for TiDAR) | n/a | 168 | -35% vs head |
| v0.16 pre-perf-hunt baseline | n/a | ~150 | before 5e4b95df0 |

### Shipped wins (this hunt)

| commit | win | confirmed gain (n=10, idle node 89) |
|---|---|---:|
| `5e4b95df0` | lm_head: bf16xbf16->fp32 via `out_dtype` (SM90 Tensor Cores) instead of fp32xfp32 (SM80 GEMM); 87.6ms->27ms across 200 tokens (3.2x lm_head GEMMs) | +8-10% (~150 -> 163) |
| `84950b974` | MOE_OP default REVERTED to OFF. The earlier flip to ON (771a701d6) was based on a high-variance n=3 measurement that included a 204 tok/s outlier; on idle node 89 n=10 the same flip is -2.7% (158.8 vs 163.3). | n/a (reverts a misjudgment) |
| `528f2b851` | Skip `_update_states_after_model_execute` for TiDAR (no GDN backend, no mamba_cache_mode=align consumer). Eliminates a dead-code .cpu().numpy() sync + cat/argmax/loop. The sync gets absorbed into the later `parse_output` sync (same total wait for GPU), so the win is the eliminated GPU compute + Python work, not the sync itself. | +1.9% (163 -> 165) |
| `8e79bdc39` | Skip `compute_causal_conv1d_metadata` when the metadata builder is CCAAttentionMetadataBuilder. The function builds Mamba1 causal_conv1d kernel metadata that CCA never reads (CCA has its own conv path). Was 6.7ms per call × 2 calls/step = ~13.5ms wall time, mostly overlapping with surrounding GPU work — net on-critical-path saving is ~0.3ms. | +1.0% (165 -> 168) |
| `c1e8c9f9c` | **THE BIG ONE.** Force `disable_padded_drafter_batch=True` for TiDAR in SpeculativeConfig validation. Restores v0.15's `_use_padded_drafter_batch()=False` behavior, which v0.16 dropped when refactoring the gate to a pure config field. Effect: TiDAR's draft fires AFTER bookkeep (CPU-list path), so bookkeep's `.cpu()` sync waits only for verify+sample (not drafter). Drafter then runs in parallel with the next step's preprocess+forward, saving ~5-10ms/step. | **+35.7%** (168 -> 228) |

### Methodology + caveats

- Per-step phase timing (instrumented via a `_TidarTime` ContextManager added to
  `gpu_model_runner.py`) showed `bookkeep` phase = ~14ms/step (76% of step time)
  on TF FA b=1 captured. Inside `bookkeep`, `RejectionSampler.parse_output` does
  a `.cpu().numpy()` sync that waits ~12-14ms for queued GPU work (verify forward
  + sample). Similar 14ms wait inside `_update_states_after_model_execute` which
  fires just before bookkeep.
- nsys identified the two top GPU kernels pre-fix as `sm80_xmma_gemm_f32f32_f32f32_f32`
  (fp32xfp32 matmul) eating ~65% of total GPU kernel time on TF FA b=1 — the fp32
  lm_head path. After 5e4b95df0, the same logical workload runs as `nvjet_tss_512x16_64x3`
  / `_512x24_64x3` (Hopper cublasLt SM90 kernels), 3.2x faster on this workload.
- Run-to-run variance is ~15%, so any sub-10% perf change can't be confirmed in a single
  run; need 3-5 run averaging or longer benches.

### Ruled out (each tried, no real gain)

- `VLLM_TIDAR_NO_COPY_INPUTS=1` (skipping the per-step cudagraph input copy):
  no improvement, and unsafe historically (TF crash without it).
- Async pinned non-blocking copy in `_update_states_after_model_execute`: 156-185
  tok/s range; mean equal to baseline. The next-step reader needs sync, so
  going async without a CUDA event makes the next read stale and the model
  generates different outputs (sometimes longer, sometimes shorter).
- Skipping `_update_states_after_model_execute` entirely when no GDN attention
  backend / mamba-align: mean = baseline mean (168.7 vs 168.7 over 3 runs each).
  The .cpu() sync was "absorbed" into the next .cpu() (parse_output) since GPU
  work has to drain somewhere.
- `VLLM_TIDAR_SMOE_MOE_OP=1` (pre-lm_head-fix): -5% regression at b=1 — the
  cat() avoidance was negated by larger custom-op block overhead while the fp32
  lm_head was the hot path. After lm_head SM90 fix, the moe_op now stacks
  cleanly.

### Where the remaining 36% gap likely sits

Sum of small structural differences vs v0.15:
1. `unified_kv_cache_update` is a separate splitting_op kernel call in v0.16 (40
   layers × 2 forwards × read+write index kernels = ~160 per step). v0.15 had
   this inline in FA's forward. nsys shows 26ms/200 tokens on these.
2. v0.16's `RejectionSampler.parse_output` + `_update_states_after_model_execute`
   both do `.cpu().numpy()` syncs on `sampled_token_ids`. v0.15 has only the
   parse_output one (no equivalent of `_update_states_after_model_execute`'s
   pre-bookkeep sync). Per-step impact unclear given the variance.
3. v0.16 captures 6 cudagraph sizes [1,2,4,8,17,306] vs v0.15's [17]. Dispatcher
   lookup per step is larger.
4. `cudagraph_copy_inputs=True` is forced for TiDAR (v0.15 had False). Per-step
   memcpy of inputs into captured buffers.

### Structural attempts (this hunt, none shipped)

Tried but didn't pan out within session scope:

1. **Re-fuse `unified_kv_cache_update` into FA's forward** (inline `reshape_and_cache_flash`
   inside `FlashAttentionImpl.forward`, set `forward_includes_kv_cache_update=True`).
   Result: FA's main `_vllm_fa3_C.fwd` kernel raises `out must have shape (total_q,
   num_heads, head_size_v)` during warmup. The flag flip changes a code path in
   `attention.py` (the no-output-tensor branch becomes legal) which in turn affects
   how the output shape is set up. Needs deeper investigation of the attention.py
   dispatch + output-tensor allocation interactions.

2. **Async scheduling for TiDAR** (add `tidar` to the eligible methods in
   `vllm/config/vllm.py:651`). Structural blockers in the existing async path:
   - `_bookkeeping_sync` async branch (line 3295) asserts
     `sampled_token_ids.shape[-1] == 1`; for TF spec-decode this is K+1=17. Need
     to either widen the assertion or add a TiDAR-specific async path that handles
     `[batch, K+1]` shapes.
   - `_commit_tidar_cca_state` currently requires `valid_sampled_token_ids` on CPU
     to compute per-request `num_accepted`. Either compute on GPU directly (replace
     `len(num_accepted_per_batch_idx)` with `idx_gpu.shape[0]`) or sync per-step
     (defeats the async benefit).
   - Consumer of `prev_sampled_token_ids` in `_update_states` assumes shape
     `[num_reqs, 1]` for the bonus column (line 1415: `prev_sampled_token_ids[:n, 0]`).
     For TF spec-decode shape `[num_reqs, K+1]`, the bonus is at column 0 which
     happens to work, but the consumer's downstream draft-token scatter (line
     1431) assumes a separate `_draft_token_ids` tensor — not how TiDAR represents
     state.

3. **Minimal `cudagraph_capture_sizes` for TF b=1** (drop `small_grid [1,2,4,8]`
   and `sf_sizes [306]`, keep only `spec_sizes [17]`). Result: mean across 3 runs
   was 171.7 vs baseline 168.7 — within noise. Dispatcher lookup overhead at 6
   sizes vs 1 is apparently small per step (microseconds, not milliseconds).

### What would actually close the gap

These need engineering work beyond this session:
- **Re-fuse kv_cache_update into FA forward** — requires understanding the
  attention.py output-tensor allocation contract when `forward_includes_kv_cache_update`
  flips. ~26ms/200 tokens of GPU kernel time recoverable.
- **Full async scheduling for TiDAR** — requires (a) widening async path assertions
  to allow spec-decode shapes, (b) GPU-only `_commit_tidar_cca_state`, (c) reworking
  `prev_sampled_token_ids` consumption for TF mode. ~25-28ms/step potentially
  recoverable.
- **Re-investigate the 1.29 GB of D2D memcpy** nsys shows (760 calls/200 tokens).
  Not clear what's being copied or where; needs targeted profiling.

## What's slow

SF FULL captured at b=1/8/16:

| batch | mine | v0.15 handoff | ratio |
|---|---:|---:|---:|
| 1 | 176 | 289 | 61% |
| 8 | 361 | 926 | 39% |
| 16 | 619 | 1146 | 54% |
| 32 | 747 | n/a | n/a |

nsys breakdown at b=1 (SF FULL captured):
- **66.3% — `triton_red_fused__softmax__to_copy_add_bitwise_not_gather_mean_mul_ne_pow_rsqrt_view_5`** (48k calls × 118μs).
  Inductor fused **SMoERouter → MOD masking → RMSNorm** into one Triton kernel. This is the dominant cost.
- 26.7% — `FillFunctor<int>` (104k calls × 22μs, zero-fills). Likely FlexAttention SF mask bookkeeping.
- 2.4% — `_sf_attention_fwd_kernel_paged` (the actual SF attention is small)
- ~5% other

Same `SMoERouter` code as v0.15 (zero diff in the class). v0.15's `SMoExperts(probs, indices)` was a 3-arg API with separate tensors; v0.16's `FusedMoE(packed_logits)` is 2-arg with `cat()` packing, and inductor fuses the cat + MOD + router into the monster kernel.

#### What I tried (all ineffective or net-negative)

| attempt | b=1 | b=8 | b=16 |
|---|---:|---:|---:|
| baseline (`9da7f7d07`, P=0,5) | 120 | 392 | 681 |
| `cudagraph_copy_inputs=False` | same | same | same |
| `cca_prefill_fused` (bf16 Triton) in SF vectorized | 5× SLOWER + degenerate output |
| `@torch.compiler.disable` on `SMoERouter.forward` | dynamo hard-fails, rejected |
| `ignore_mod_in_smoe_block=True` | 141 | — | — |
| `custom_ops=["all"]` | 152 | — | — |
| `custom_ops=["+rms_norm"]` | **185** (+5%) | **295** (-18%) | **516** (-17%) |
| `custom_ops=["+rms_norm","+silu_and_mul","+rotary_embedding"]` | 128 | — | — |
| `combo_kernels=False` | same | — | — |
| **`torch.ops.vllm.smoe_pack_logits` custom op wrapping the cat()** | 121 (=) | 396 (=) | — |
| **direct `fused_experts(probs, indices)` bypass, skip FusedMoE.forward** | **57 (−52%)** | **169 (−57%)** | **306 (−55%)** |
| **`probs.contiguous()` to force materialization** | 48 (−60%) | — | — |

#### Root cause identified (kernel diff vs v0.15)

Diffed the cached Triton kernels v0.15 vs v0.16. v0.15's equivalent fused kernel is named `triton_red_fused__to_copy_add_bitwise_not_mean_mul_ne_pow_rsqrt_5` (residual + MOD + RMSNorm). v0.16's is `triton_red_fused__softmax__to_copy_add_bitwise_not_gather_mean_mul_ne_pow_rsqrt_view_5` — **v0.16 ALSO has `softmax`, `gather`, `view`** fused in.

Reading the inductor-generated Triton: the v0.16 kernel literally **recomputes softmax inline** (`exp(logit-max)/sum`) for the MOD passthrough's `hidden_states * probs`. v0.15 just looked up the already-materialized `probs[topk_id]`. Inductor decided that re-computing softmax+gather was cheaper than keeping `probs` in memory.

The cat() that v0.16's `FusedMoE.forward(packed_logits)` requires is what triggered this. `probs` flows into TWO downstream consumers: (a) the cat for FusedMoE, (b) `hidden_states_mod = hidden_states * probs`. With the cat in v0.16, inductor's heuristic re-fires softmax+gather rather than reusing the stored `probs`.

#### What's likely to help

1. **Force inductor to materialize probs** — but `.contiguous()` and `.clone()` both regress (cudagraph buffer reallocation overhead larger than the recompute savings). Need a custom-op black box that consumes `probs` and returns it unchanged.
2. **Restore v0.15's 3-arg expert API** by replacing `FusedMoE.forward(hidden_states, packed_logits)` with a direct call to `fused_experts(hidden_states, w13_weight, w2_weight, probs, indices, ...)` — bypasses FusedMoE.forward entirely. **Tried and it regresses 50%+** because the `fused_experts` direct path lacks the cudagraph-friendly dispatch machinery FusedMoE wraps it in. Would need a custom wrapper that's cudagraph-friendly AND takes probs/indices separately.
3. **Hand-tuned MoE router Triton kernel** that does `down_proj → rmsnorm → router_mlp → softmax → topk → gather → MOD_mask` in one well-tuned kernel. Beats inductor's auto-fusion. Highest potential impact but most work.
4. **Investigate `FillFunctor<int>` 103k calls** — that's ~4700 int-zero fills per step. If reducible (FlexAttention SF mask bookkeeping is the suspect), saves ~27% of GPU time.

#### Custom-op-wrapped MoE call — SHIPPED, opt-in via `VLLM_TIDAR_SMOE_MOE_OP=1`

Idea (1) plus a wrinkle: instead of a barrier op that consumes-and-returns probs (tried earlier as `vllm::smoe_pack_logits` with `mutates_args=[]` — didn't help), wrap **the entire MoE call including the cat()** in an opaque custom op. The op takes `(hidden_states, probs, indices, layer_name)` and runs `torch.cat(...)` + `FusedMoE.forward_impl(...)` internally; inductor sees just an opaque dispatch and must materialize `probs` as the op's input. Downstream `hidden_states * probs` then reads the materialized buffer instead of re-firing softmax+gather inline.

Shipped as `torch.ops.vllm.tidar_smoe_moe_with_probs` (commit `b03144b13`), opt-in via `VLLM_TIDAR_SMOE_MOE_OP=1`. Default OFF.

**Measured (smoediffusion iter_0012000, node 2 GPU 4, max_tokens=200/400):**

| config                     | baseline | +moe_op | gain  | notes |
|----------------------------|---------:|--------:|------:|-------|
| SF FULL b=3 P=0,5          |   103.99 |  159.74 |  **+54%** | sparse-proposal SF |
| SF FULL b=3 P=0..16 dense  |    63.92 |   61.63 |   -3% | large M absorbs the recompute |
| **TF FULL b=1 P=0..16 dense** |    14.25 |   **20.73** |  **+45%** | per-step is always K+1=17 tokens |

The fix is **M-dependent**: helps when the MoE call sees small M (TF's always-K+1 layout, or SF's sparse proposal config), neutral at large M (SF dense's 306 tokens/req). At large M the inline softmax recompute is amortized across many MoE expert-kernel runs; at small M the recompute dominates and the materialize-once path saves ~half.

Outputs verified coherent (math-reasoning prompts, mean acceptance length 5+).

**Failed earlier attempt: probs-materialization barrier** (no-op custom op with `mutates_args=["t"]`). The idea was to fence inductor at the probs site so downstream reads couldn't fuse back to softmax. **Hurt perf** (SF b=3 P=0,5: 104 → 59 tok/s, -43%) — likely inductor emitted extra stores around the mutates-args fence. Discarded.

Approaches still untried that could close the dense-P SF gap: hand-tuned MoE router Triton kernel; `FillFunctor<int>` reduction; combining moe_op with FlexAttention SF mask_mod refactor.

## Smaller follow-ups

- ~~`_sf_mmlu_sweep.py` crashes at FULL captured~~ — **FIXED** in `bd74b3151`: removed the `llm.generate(["hi"], max_tokens=4)` pre-timing warmup. torch.compile + cudagraph capture already happen during `LLM(...)` construction, so the warmup wasn't needed for correctness. Root cause turned out to be a more general bug (see below).
- ~~Eager-mode output drift after ~600 tokens~~ — **VERIFIED FIXED at HEAD**: TF eager runs 700-token math reasoning coherently end-to-end; SF eager runs 900-token reasoning coherently end-to-end (both `temperature=0.0`, `seed=0`). The fix was commit `3026d9151` (force CCA vectorized path under all TiDAR, so the K+1 stash candidates exist for `commit_spec_decode_state` even outside FULL captured).
- ~~`cudaErrorCapturedEvent` from Triton autotune~~ — **NOT REPRODUCIBLE at HEAD** with `VLLM_TIDAR_SF_TRITON=1` (the default, paged-cache Triton kernel). Capture completes cleanly because `cudagraph_num_of_warmups=1` runs the SF Triton kernel inside `graph_capture(...)` but outside the actual `torch.cuda.graph()` block — autotune events go to a non-capturing stream and the chosen config is cached for the subsequent capture. **Caveat:** the opt-out `VLLM_TIDAR_SF_TRITON=0` (multi-call FA fallback) now fails engine init at HEAD — should be removed or fixed if it's still load-bearing anywhere.

### Newly discovered: multi-`llm.generate()` engine corruption

While verifying the sweep-script fix I found that **two consecutive `llm.generate()` calls on the same LLM instance under SF FULL captured will crash the second call** with `cudaErrorIllegalAddress` at `_update_states_after_model_execute → .cpu()`. The first call works; the second dies. Reproduces with `max_num_seqs=16`, batch grown from 1 → 4 across calls. The previous sweep-script `["hi"] max_tokens=4` warmup hit this same bug. Single `llm.generate()` calls with multiple prompts in one batch work fine (that's how `bench_tidar.py` reports the 176/361/619 tok/s numbers).

Likely candidates: block-table state from the finished request isn't fully released before the next call schedules; or some scheduler-side state (`num_computed_tokens`, `_num_computed_tokens_cache`) sticks between calls in a way the captured graph can't tolerate. Not investigated further — `bench_tidar.py` and `_sf_mmlu_sweep.py` both work around it by issuing a single batched call.

## Commit chain (this port)

| commit | what |
|---|---|
| `cd68dec2c` | Phase 1+2 base (TF + SF eager; PIECEWISE captured runs but output was degenerate, nobody checked at the time) |
| `fa4aced31` | Phase 3 CCA vectorized prefill (captures cleanly, wrong shape) |
| `8c41cc74a` | perf gate + fp32 conv math |
| `0ff753452` | 4 capture-wiring fixes (Block 8, 8a, 10, 12) — right shape, replay crashes |
| `407e4db31` | CCA write-side fix (TF eager coherent for ~300 tok) |
| `85e6bb074` | `cudagraph_copy_inputs=True` default → SF FULL replay works |
| `a5f3dcc7a` | cudagraph-safe `cca_prefill_fused` API + `_conv_qk_apply` in SF vectorized |
| `3026d9151` | force CCA vectorized path under ALL TiDAR (not just FULL) — fixes TF eager drift |
| `878231bc8` | port Block 5 mix-logit / drafts-only / ar-only to v0.16 |
| `9da7f7d07` | `is_drafter_pass` scaffolding + force `cudagraph_copy_inputs=True` |
| `43c0fa08e` | TF FULL/PIECEWISE captured FIX: router-pad workaround (`VLLM_TIDAR_ROUTER_PAD=1`) dodges buggy cutlass align1 OOB |
| `b03144b13` | perf: `VLLM_TIDAR_SMOE_MOE_OP=1` opaque MoE custom op (+45% TF dense, +54% SF sparse P) |
| `0001ddc4d` | TF drafter warmup capture (eager-drafter ceiling 7-14 tok/s -> 27 tok/s captured) |
| `35a01a325` | FA backend 0% accept fix (slot_mapping dict missing in `set_forward_context`) |
| `a658cb610` | perf: cache fp32 lm_head weight transpose (2GB/step recompute removed) |
| `1d23c6385` | perf: hoist `_commit_tidar_cca_state` (idx_gpu/arange_gpu) |
| `9945e9e5b` | perf: cache `_get_cca_block_slots` result |
| `80ab16112` | perf: cache FA group id lookup in `_build_draft_inputs` |
| `284091c04` | perf: cache `/tmp/tidar_mix_w` file-read |
| `955cc6361` | perf: skip per-step `state_indices_tensor.tolist()` sync |
| `96cc939c7` | perf+accept: `VLLM_TIDAR_FA_NO_SPLITS=1` forces FA `num_splits=1` (recovers +10% accept on TF captured) |
| `5a51e6ea4` | fix: gate `VLLM_TIDAR_FA_NO_SPLITS` on TF mode only (SF combo was broken) |
| `55e08f7cb` | warn: log warning when SF mode runs with FA backend (FA + SF is broken; force FLEX) |
| `4414adfd8` | chore: commit bench_tidar.py + profile_sf.py |
| `eb694969c` | docs: handoff update with v0.15 baseline + drafter-drift correction |
| `f87075885` | docs: correct phantom 22% drafter drift (was n=1 outlier window) |
| `5e4b95df0` | perf: lm_head SM90 Tensor Cores via out_dtype (3.2x lm_head GEMMs, +9-15% TF FA b=1) |
| `771a701d6` | perf: default `VLLM_TIDAR_SMOE_MOE_OP=1` (later reverted: see 84950b974) |
| `84950b974` | perf: revert MOE_OP default to OFF (regresses 2.7% at n=10; the n=3 "win" was variance artifact) |
| `5255a5648` | docs: update with idle-node-89 measurements (node 2 was contended) |
| `528f2b851` | perf: skip _update_states_after_model_execute for TiDAR (no GDN/mamba-align consumer) +1.9% n=10 |
| `f4f94cb9d` | docs: handoff update for skip_uam |
| `8e79bdc39` | perf: skip compute_causal_conv1d_metadata for CCA (dead-code Mamba1 conv kernel metadata) +1.0% n=10 |
| `caef41083` | docs: handoff update for skip_conv1d_metadata |
| `c1e8c9f9c` | perf: **force disable_padded_drafter_batch=True for TiDAR** — restores v0.15 fast path. +35.7% n=10. THE BIG ONE. |

## v0.15 vs v0.16 — things to know if you keep working on this

- v0.16's runner does **NOT** rebind `drafter.model` to the CUDAGraphWrapper-wrapped model. v0.15 did (`gpu_model_runner.py:3504`). In v0.16 `TiDARProposer.load_model` binds `self.model = target_model` (unwrapped) BEFORE the runner wraps. So `drafter.forward` bypasses the wrapper and runs eager. This is mostly fine (drafter doesn't get the perf benefit of cudagraphs, but the verifier path is the dominant cost) — except it makes the v0.15 trick of "lazily capture a drafter-pass graph with write→draft-scratch in scope" impossible without first rebinding (and rebinding + lazy capture produces zero-output drafter graphs, see above).
- v0.16's `BatchDescriptor` is a NamedTuple. I added `is_drafter_pass: bool = False` at the end (defaults preserve equality with existing call sites). Threaded through `_create_padded_batch_descriptor` and `dispatch()`. Scaffolding only — no `is_drafter_pass=True` keys are registered, so dispatcher always returns NONE for drafter dispatches.
- v0.16's `_dummy_run` is the warmup capture driver and uses `_determine_batch_execution_and_padding` which calls `cudagraph_dispatcher.dispatch`. Warmup capture sets `uniform_decode=True, num_tokens=K+1` for the TF verifier shape. Drafter override metadata is NOT in scope at warmup → captured graph for `is_drafter_pass=True` (if registered) would have write→AR, not write→draft.
- v0.16's `BatchDescriptor` includes `num_reqs`, `has_lora`, `num_active_loras`. The runtime dispatch trace shows TWO calls per step from `gpu_model_runner.py:3419` early on — one `uniform=True` (verifier, FULL match) and one `uniform=False` (the prefill, which gets padded to 17 with `uniform=False` and finds no FULL key). The `uniform=False` dispatch is harmless.
- v0.16's CCA layer has the new `_gw_weight_T` cached weight view, `refresh_runtime_weight_views()`, and a bunch of env-gated fused ops (`_CCA_FUSED_ENABLED`, `_CCA_TRITON_FUSION_ENABLED`, `_CCA_DIM_PRESERVE_CONV_ENABLED`). All default OFF. None used in the active code path.
- `mamba_attn.py` builder's persistent buffer copy (`self.state_indices_tensor`) fires for pure-decode batches under has_full_cudagraphs. For TF verifier (classified as PREFILL since query_len=17 > 1), the persistent buffer copy is SKIPPED and `state_indices_tensor` returns the fresh slice from `mamba_get_block_table_tensor(...)[:, 0]`. Pointer is stable (slice of persistent `block_table_tensor`), so this is fine for capture/replay — but worth knowing if you're tracing pointer stability.

## Related memory entries (Claude Code project memory)

- `project_tidar_v016_port_in_progress` — running status (this handoff is a snapshot of it)
- `project_sf_captured_cudagraph_fixes` — env_override patches preserved across the port
- `project_sf_kp1_layout_required` — K+1 layout respected
- `project_sf_requires_flex_backend` — v0.16 env var is silent, route through `attention_backend` kwarg
- `project_sf_kmask_no_bonus_dead` — K+1 + TFNA0 vs paper-faithful K-mask comparison (K+1 wins, dataset-dependent)
- `feedback_smoediffusion_naming` — new eval scripts use `smoediffusion_` prefix
- `feedback_always_log_stats` — TF/SF benches must pass `--log-stats`
