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
| TF FULL_DECODE_ONLY captured | ❌ | Crashes deterministically at 3rd verifier replay (cudaErrorIllegalAddress) |
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

### Run TF (eager only — FULL captured is broken)

```python
import os
os.environ["VLLM_TIDAR_TWO_FORWARD"] = "1"

llm = LLM(
    model=ckpt, ..., enforce_eager=True,   # <-- required for TF today
    speculative_config={...},
    # do NOT pass compilation_config with FULL/PIECEWISE — both crash
)
```

### Benchmark + profile

`scripts/bench_tidar.py` and `scripts/profile_sf.py` (uncommitted, in tree) drive the SF perf numbers above and the nsys profile that pinpoints the SMoERouter hotspot.

## What's broken

### TF FULL/PIECEWISE captured — deferred

Verifier graph A replays cleanly for steps 1 and 2, then crashes inside `entry.cudagraph.replay()` at step 3 with `cudaErrorIllegalAddress`. Reproduces:
- across K=4 (shape 5) and K=16 (shape 17) — **step-count specific, not shape-specific**
- max_tokens 1-4: OK (1-2 replays only). max_tokens ≥ 5: crash at 3rd replay.
- with/without `cudagraph_copy_inputs`, `combo_kernels`, `enable_chunked_prefill`, `cudagraph_capture_sizes=[17]` only
- with/without the v0.15-style `drafter.model = self.model` rebind (rebind + lazy capture *succeeds at capture* but produces all-zero spec tokens and the next verifier crashes anyway)

PIECEWISE TF fails earlier in `cudaGraphLaunch` with a process segfault — likely the same root cause, different surface.

#### Already ruled out

| Hypothesis | Result |
|---|---|
| Dispatcher routing | Verifier always returns FULL match for `(uniform=True, drafter_pass=False)`. Confirmed via runtime print. |
| Drafter corrupting state | Drafter is eager + `cca_drafter_pass=True` → `skip_writes=True`, doesn't touch AR slot. |
| Recent CCA gate change (`_use_spec_vectorized = bool(_spec_stash_conv is not None)`) | Reverting to commit `a5f3dcc7a`'s FULL-only gate still crashes. Bug predates the fix. |
| Persistent buffer pointer drift | `state_indices_tensor`, `_spec_stash_*`, `block_table_tensor`, `query_start_loc.gpu`, `seq_lens.gpu`, `slot_mapping.gpu` are all persistent slices with stable `data_ptr()`. |
| v0.15-style drafter rebind + lazy capture | Lazy capture succeeds (with `graph_capture(device=...)` bracket to satisfy non-default-stream). Captured drafter emits all-zero tokens → next verifier crashes. Removing rebind reverts to original step-3 crash. |
| max_num_seqs > 1, chunked_prefill off, capture_sizes=[17] only, block_size > 16 | All still crash at step 3 (block_size > 16 is rejected by FlexAttention). |

#### Next things to try

1. **`compute-sanitizer --tool memcheck`** on a TF FULL run. The OOB access is inside a captured kernel — the only practical way to localize it is device-side memory tracking. PyTorch's `TORCH_USE_CUDA_DSA=1` requires a custom build (env var alone has no effect).
2. **v0.15 ↔ v0.16 CCA bisect**. v0.15's CCA forward worked under TF FULL. v0.16 added `cca_prefill_fused`, `cca_decode_fused`, `cca_prefill_fused_hip`, `fused_pad_gather_scatter`, `fused_qk_mean`, and the `_gw_weight_T` runtime weight view cache. Even though the new fused paths are env-gated (default off), `__init__` runs them. Diffing the active code path between v0.15 and v0.16 would narrow this down.
3. **Drafter rebind producing zeros** is a separate bug worth investigating in its own right (would unlock a route to capturing the drafter graph, even if it doesn't fix the verifier crash).

The crash being **deterministic at step 3 regardless of shape, prompt length, K, batch size, or block crossing** is a strong fingerprint but I haven't been able to attribute it to anything observable from Python.

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
| baseline (`9da7f7d07`) | 176 | 361 | 619 |
| `cudagraph_copy_inputs=False` | same | same | same |
| `cca_prefill_fused` (bf16 Triton) in SF vectorized | 5× SLOWER + degenerate output |
| `@torch.compiler.disable` on `SMoERouter.forward` | dynamo hard-fails, rejected |
| `ignore_mod_in_smoe_block=True` | 141 | — | — |
| `custom_ops=["all"]` | 152 | — | — |
| `custom_ops=["+rms_norm"]` | **185** (+5%) | **295** (-18%) | **516** (-17%) |
| `custom_ops=["+rms_norm","+silu_and_mul","+rotary_embedding"]` | 128 | — | — |
| `combo_kernels=False` | same | — | — |

#### What's likely to help

1. **Rewrite SMoEBlock + SMoERouter** to avoid the cat()+MOD inductor fusion. Either restore the 3-arg `SMoExperts` API or unroll the cat differently. Highest potential impact.
2. **Hand-tuned MoE router Triton kernel** that does `down_proj → rmsnorm → router_mlp → softmax → topk → gather → MOD_mask` in one well-tuned kernel. Beats inductor's auto-fusion.
3. **Investigate `FillFunctor<int>` 103k calls** — that's ~4700 int-zero fills per step. If reducible (FlexAttention SF mask bookkeeping is the suspect), saves ~27% of GPU time.
4. **Try direct `fused_experts` call** in place of `FusedMoE.forward()` — bypass the generic vLLM class entirely.

## Smaller follow-ups

- `_sf_mmlu_sweep.py` crashes at FULL captured with the same illegal-memory pattern that direct invocation no longer hits. Likely the `llm.generate(["hi"], max_tokens=4)` warmup leaves the engine in a state that subsequent SF steps can't recover from. Direct invocation skipping that warmup works.
- Eager-mode output drift after ~600 tokens (loops back to restating the problem — likely RoPE position vs CCA state mismatch slowly accumulating). Lower priority since FULL captured doesn't drift in the same window.
- `cudaErrorCapturedEvent` from Triton autotune `Event.elapsed_time` during SF capture — workaround was `VLLM_TIDAR_SF_TRITON=0` falling through to FlexAttention (see `project_sf_captured_cudagraph_fixes`). Proper fix would be pre-capture autotune.

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
