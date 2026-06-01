# TiDAR vLLM v0.16 Port — Handoff

**Branch:** `jinzhao/tidar_v016` @ `9da7f7d07`
**Repo:** `git@github.com:Zyphra/Zvllm.git`
**Node tested:** vp-dgx-51 (147.68.0.51)
**Env:** `/data/home/jinzhao/workspace/tidar/Zvllm-v016/.venv-v016`
**Date:** 2026-05-31

## What works

| Mode | Status | Notes |
|---|---|---|
| AR eager | ✅ | Reference, ~30 tok/s on iter_0012000 |
| AR FULL captured | ✅ | Standard vLLM path |
| SF eager | ✅ | Coherent |
| SF PIECEWISE captured | ✅ | Coherent |
| **SF FULL_DECODE_ONLY captured** | ✅ | **Primary path** — 176/361/619 tok/s b=1/8/16 |
| TF eager | ✅ | ~20 tok/s, coherent |
| TF PIECEWISE captured | ❌ | Segfault in `cudaGraphLaunch` |
| **TF FULL_DECODE_ONLY captured** | ✅ (opt-in) | **22 tok/s b=1**, coherent. Requires `VLLM_TIDAR_ROUTER_PAD=1` env var. Without it, still crashes at step 3. |
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
export VLLM_TIDAR_ROUTER_PAD=1   # required for TF FULL captured
```

```python
llm = LLM(
    model=ckpt, ..., enforce_eager=False,
    speculative_config={...},
    compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
)
```

22 tok/s at b=1 on iter_0012000. The padded-router workaround is dormant
without `VLLM_TIDAR_ROUTER_PAD=1`, so SF (which doesn't need the fix)
stays on the baseline path by default.

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

`scripts/bench_tidar.py` and `scripts/profile_sf.py` (uncommitted, in tree) drive the SF perf numbers above and the nsys profile that pinpoints the SMoERouter hotspot.

## What's broken

### TF FULL captured — FIXED in `43c0fa08e` (opt-in via `VLLM_TIDAR_ROUTER_PAD=1`)

**Root cause:** `SMoERouter`'s final linear projects `D=mlp_expansion -> num_experts (=17 with MOD)`. That output stride (17) is not a multiple of 8, so under captured-cudagraph replay cublas selects `cutlass_75_tensorop_s1688gemm_bf16_64x64_tn_align1` — a kernel that **writes 1+ bytes past its output allocation**. In eager mode the regular allocator's padding hides the OOB; the cudagraph memory pool's tight fit exposes it. Verified with `compute-sanitizer --tool=memcheck`: every TF FULL crash hits this exact kernel at `+0x5430`, OOB by 1 byte from a 1.18 GB-adjacent allocation.

**Fix:** project against a padded `D -> E_padded` weight (E=17 → E_padded=24, the next multiple of 8). cublas picks a different (correct) sm_90 kernel for the aligned output stride. Slice back to E columns for downstream code. Padded weight buffer is filled lazily on first forward from `router_mlp[4].weight`, no checkpoint format change needed.

**Why opt-in:** SF FULL captured doesn't hit the buggy kernel at its runtime shapes, and the padded matmul has small overhead at b=16 in noisy benchmarks. SF stays on the baseline path by default.

**Validation (smoediffusion iter_0012000, b=1, max_tokens=500):**
- `VLLM_TIDAR_TWO_FORWARD=1` `VLLM_TIDAR_ROUTER_PAD=1` `cudagraph_mode=FULL_DECODE_ONLY` → **22.4 tok/s, coherent math reasoning end-to-end**. Was 0 tok/s (engine crashed at step 3 = ~5 tokens).
- Without `VLLM_TIDAR_ROUTER_PAD=1`, TF FULL still crashes (verified — the fix is dormant when the flag is off).

### TF PIECEWISE captured — still broken

PIECEWISE TF fails earlier in `cudaGraphLaunch` with a process segfault. Likely the same router-output OOB triggered from a piecewise-captured subgraph; the `VLLM_TIDAR_ROUTER_PAD=1` workaround has not been tested on this path yet. The SF FULL captured path is the supported alternative.

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
