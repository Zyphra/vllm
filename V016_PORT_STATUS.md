# jinzhao/tidar_v016 — WIP port status

Branched from `origin/smoe-aiter-moe` on 2026-05-30 to port `jinzhao/tidar`'s
TiDAR (TF + SF) onto vLLM v0.16.0. Phase 1 scope: SF-only minimum viable,
eager mode first.

## What's done

| file | status | notes |
|---|---|---|
| `vllm/v1/spec_decode/tidar.py` | new + 1 import fix | `vllm.attention.layer` → `vllm.model_executor.layers.attention.attention` |
| `vllm/v1/spec_decode/tidar_single_forward.py` | new (copied verbatim) | |
| `vllm/v1/spec_decode/cca.py` | new (copied verbatim) | spec_decode CCA support |
| `vllm/v1/spec_decode/metadata.py` | merged | added `draft_probs` + `draft_logits` Optional fields onto aiter-moe's `cu_num_sampled_tokens` addition |
| `vllm/attention/ops/sf_attention.py` | new (copied verbatim) | paged Triton SF kernel |
| `vllm/env_override.py` | merged | appended SF cudagraph workarounds after aiter-moe's torch 2.9 inductor patches |
| `vllm/config/speculative.py` | minimal port | added `"tidar"` to `SpeculativeMethod`, `tidar_diff_temperature` field, `__post_init__` tidar branches (drafter setup + token-tree), `use_tidar()` method. Skipped: `compute_hash` refactor to `uses_aux_hidden_state_outputs()`, `deepseek_v32` handling, MTP consolidation (those are general v0.16 vLLM-side changes, not TiDAR-specific) |
| `vllm/v1/attention/backends/flex_attention.py` | **brute-force copy from jinzhao/tidar** | aiter-moe's structural changes (method renames, new `supports_attn_type`, `supports_mm_prefix`, `get_prefix_lm_mask_mod`, etc.) are NOT preserved. Will likely need adapting at runtime |
| `vllm/v1/worker/gpu_model_runner.py` | **brute-force copy from jinzhao/tidar** + 1 import fix | aiter-moe's ~3000-line refactor lost; `vllm.attention.layers.chunked_local_attention` → `vllm.model_executor.layers.attention.chunked_local_attention` |
| `scripts/_sf_mmlu_sweep.py` | new (with hmmt dataset) | |
| `handoff.md`, `SF_proposal_layouts.md`, `accept_comparison_OPD.md` | new (docs) | |
| `docs/imgs/acc_dist_p17_thinking.png` | new | |
| CCA model arch files (`vllm/model_executor/layers/mamba/cca.py`, `vllm/v1/attention/backends/cca_attn.py`) | **kept aiter-moe's versions** | aiter-moe has their own SMoE arch CCA implementation; ours would conflict |

## What's NOT done

| step | reason |
|---|---|
| `vllm/config/vllm.py` cuda_graph_sizes override for TiDAR K+1 capture shapes | Phase 2 — only fires under captured mode, deferred to start with eager |
| `vllm/v1/attention/backends/flex_attention.py` — adapt to aiter-moe's `supports_attn_type` / `supports_mm_prefix` / `get_prefix_lm_mask_mod` / etc. | Will surface as build/runtime errors; fix iteratively after first build |
| `vllm/v1/worker/gpu_model_runner.py` — adapt to aiter-moe's ~3000-line refactor (model loader, KV cache manager, scheduler hooks, etc.) | Same — fix iteratively after first build |
| `pip install -e . VLLM_USE_PRECOMPILED=0` recompile | Not run. Required before any import test will succeed (existing venv has v0.15 `_custom_ops` already registered → `register_fake` collision when v0.16's `_custom_ops` tries to register the same op) |
| Smoke test TF + SF on iter_0012000 | Blocked on build |

## How to continue

```bash
cd /tmp/_v016_port  # on vp-dgx-4 OR clone fresh from jinzhao/tidar_v016
# Set up a clean venv (don't use the shared /data/home/jinzhao/workspace/tidar/.venv,
# which the other worktrees depend on)
python -m venv .venv-v016
source .venv-v016/bin/activate
pip install --upgrade pip wheel ninja
VLLM_USE_PRECOMPILED=0 pip install -e . --no-build-isolation
# ^ expect 30-60 min on H100 node for the full C/Triton/CUDA build

# Try imports
python -c "from vllm import LLM"

# Fix errors iteratively, recompile only if C-side changes
# (Python-only changes don't need re-pip-install with editable mode)
```

## Anticipated breakages (rough order of likelihood)

1. **Import path changes** — any `from vllm.X import Y` where X moved. Already hit 2; many more likely. Run `python -c "from vllm.v1.spec_decode.tidar import TiDARProposer"` for the first wave.
2. **`EagleProposer` base class API** — TiDARProposer extends EagleProposer. If EagleProposer's method signatures (`propose`, `prepare_inputs`) changed between v0.15 and v0.16, our overrides will break.
3. **`FlexAttentionMetadata` field names** — we add SF-specific fields (`tidar_single_forward_proposal_acc_levels`); aiter-moe added their own fields (`logical_block_ids` etc.). Need to add ours to the new dataclass.
4. **`SpecDecodeMetadata.make_dummy` signature** — aiter-moe added `cu_num_sampled_tokens`; our `draft_probs` / `draft_logits` are kwarg-defaulted so should be backward-compat, but verify.
5. **`gpu_model_runner.py` model-loading code** — we have the v0.15 model loader call; aiter-moe restructured this. May need surgical replacement of our entire model-loading block with aiter-moe's.

## Memory references

- [[project_sf_captured_cudagraph_fixes]] — the 5 env_override patches needed; ported.
- [[project_sf_kp1_layout_required]] — K+1 layout requirement; respected.
- [[project_sf_requires_flex_backend]] — FLEX_ATTENTION backend requirement; respected.
- [[project_sf_multi_call_acceptance_regression]] — multi-call FA path is eager-only bug; preserved.
