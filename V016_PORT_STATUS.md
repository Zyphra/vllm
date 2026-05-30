# jinzhao/tidar_v016 — Phase 1 SF-only eager port: ✅ DONE (2026-05-30)

Branched from `origin/smoe-aiter-moe` on 2026-05-30 to port `jinzhao/tidar`'s
TiDAR (TF + SF + samplers) onto vLLM v0.16.0. Phase 1 scope: SF-only minimum
viable, eager mode first.

## Status

✅ **Phase 1 complete and validated.** Both TF and SF modes boot cleanly on
v0.16.0+aiter-moe and generate tokens with non-trivial speculative
acceptance.

Final smoke tests on `iter_0012000` (vp-dgx-51, .venv-v016, eager mode):

| run | mode | tok/s | result |
|---|---|---|---|
| `v016_TF_smoke`, mt=128, n=1 | TF (VLLM_TIDAR_TWO_FORWARD=1) | 38.26 | 128/128 tokens, no errors |
| `v016_SF_smoke`, mt=128, n=1 | SF (default) | 43.80 | 128/128 tokens, no errors |
| `v016_SF_2k`, mt=2048, n=2 | SF (default) | 41.39 | 4096/4096 tokens; SpecDecoding metrics confirm acceptance up to 17.0 mean / 100% per-pos |

## Build infrastructure (already done; reuse, don't rebuild)

- Worktree: `/data/home/jinzhao/workspace/tidar/Zvllm-v016` on **vp-dgx-51** (`147.68.0.51`)
- Venv: `.venv-v016/` (must `source` it, or prepend `.venv-v016/bin` to `PATH` for cmake 4.3.2)
- vllm package: `vllm-0.16.1.dev37+gb106800a0.cu128`, torch 2.9.1+cu126, Triton 3.5.1, full C/CUDA/Triton compiled
- Built with: `VLLM_USE_PRECOMPILED=0 pip install -e . --no-build-isolation`
- Build deps: `cmake>=3.26.1, setuptools>=77,<81, setuptools-scm>=8.0, numpy, grpcio-tools==1.78.0, datasets` (sweep script needs the last one)

## Commits on jinzhao/tidar_v016

```
347bc46b2 feat(v016 port): 7 TiDAR injection blocks + runtime fixes — pipeline runs end-to-end
4cfe60bb3 wip(v016 port): reset gpu_model_runner.py to aiter-moe baseline; update V016_PORT_STATUS.md with 7-block injection plan
c527ac174 wip(v016 port): more import fixes; pivoting strategy for gpu_model_runner.py
dd7c7caaf wip(v016 port): batch import fixes for flex_attention.py + first wave of gpu_model_runner.py
b106800a0 wip: brute-force copy flex_attention.py + gpu_model_runner.py from jinzhao/tidar onto smoe-aiter-moe base
b54275125 wip: port tidar-only files + small shared (metadata, env_override, speculative) onto smoe-aiter-moe
72effe5ad Update qwen3 reasoning parser   ← aiter-moe HEAD baseline
```

## What's done

### Files added wholesale (no conflict with aiter-moe)
- `vllm/v1/spec_decode/tidar.py` (+runtime patches; see below)
- `vllm/v1/spec_decode/tidar_single_forward.py` (verbatim from jinzhao/tidar)
- `vllm/v1/spec_decode/cca.py` (spec-decode CCA, verbatim)
- `vllm/attention/ops/sf_attention.py` (paged Triton SF kernel, verbatim)
- `scripts/_sf_mmlu_sweep.py` (+ hmmt dataset support)
- `handoff.md`, `SF_proposal_layouts.md`, `accept_comparison_OPD.md`, `docs/imgs/acc_dist_p17_thinking.png`

### Small shared files merged
- `vllm/v1/spec_decode/metadata.py` — added `draft_probs` + `draft_logits` Optional fields onto aiter-moe's `cu_num_sampled_tokens`
- `vllm/env_override.py` — appended SF cudagraph workarounds after aiter-moe's torch 2.9 inductor patches
- `vllm/config/speculative.py` — added `"tidar"` to method literal, `tidar_diff_temperature` field, `__post_init__` tidar branches, `use_tidar()` method
- `vllm/v1/attention/backends/flex_attention.py` — copied wholesale + minor import-path adjustments

### 7 TiDAR hooks injected into aiter-moe's `gpu_model_runner.py`
- **Block 1**: `_draft_probs_by_req_id` + `_draft_logits_by_req_id` dicts in `__init__`
- **Block 2**: `_gather_draft_probs()` method
- **Block 3**: `_gather_draft_logits()` method
- **Block 4**: `spec_decode_metadata.draft_probs/draft_logits` population in `_calc_spec_decode_metadata`
- **Block 5 (minimal)**: `spec_decode_metadata.draft_probs` threaded through `self.rejection_sampler` call instead of hard-coded `None`. **NOTE**: full v1 mix-logit / no_bonus / drafts_only / ar_only logic from `jinzhao/tidar` was NOT ported — v0.16's `rejection_sampler` signature changed (it now handles target/bonus splitting internally). Adding mix-logit v1 on v0.16 requires understanding the new signature; defer to follow-up.
- **Block 6**: TiDAR SF early-exit (`extract_drafts_from_hidden`) at top of eagle dispatch branch
- **Block 7**: drafter stash hooks (`last_draft_probs/logits` → `_draft_probs/logits_by_req_id`) after `self.drafter.propose`

Plus: TiDARProposer dispatch case in drafter `__init__` (was falling into `EagleProposer` because `use_eagle()` returns True for `"tidar"`).

### Runtime API drift fixes
- `vllm/model_executor/layers/mamba/abstract.py` — relaxed `get_kv_cache_spec` whitelist for `smoe` + `tidar` method (Mamba+spec was rejecting everything except `qwen3_next`)
- `tidar.py` `dummy_run` — accepts v0.16 kwargs (`use_cudagraphs`, `is_graph_capturing`, `slot_mappings`)
- `tidar.py` `propose` signature — aligned with v0.16 EagleProposer: `last_token_indices` ↔ `token_indices_to_sample`, `mm_embeds` ↔ `mm_embed_inputs`, accept `num_rejected_tokens_gpu` + `slot_mappings`
- `tidar.py` `_build_draft_inputs` — `self.block_size` → `self.attn_metadata_builder.kv_cache_spec.block_size` (eagle.py:624 pattern)
- `tidar.py` `CommonAttentionMetadata` constructor — `seq_lens_cpu` is now a `@property`, use `_seq_lens_cpu` private field; same for `num_computed_tokens_cpu`
- `tidar.py` `use_cuda_graph` / `cudagraph_batch_sizes` — guarded with `getattr(...)` defaults (removed in v0.16)
- `tidar.py` `BatchDescriptor` — `uniform_decode` → `uniform`; `is_drafter_pass` field removed

## What's NOT done (Phase 2 and beyond)

### Phase 2: captured (cudagraph) mode
- Need to re-add the SF capture-shape override (the `cuda_graph_sizes` explicit list logic from our `jinzhao/tidar` `vllm/config/__init__.py`) into v0.16's `vllm/config/vllm.py`. Only matters under `enforce_eager=False`.
- Validate captured mode reproduces handoff numbers (handoff: TF 321 / SF 302 tok/s at K=16 b=1 thinking-off).

### Full sampler suite
- Block 5 currently only threads `draft_probs` through. The full mix-logit v1 / v2 / no_bonus / drafts_only / ar_only sampler block from jinzhao/tidar still needs porting onto v0.16's restructured `rejection_sampler` call site.

### Auxiliary
- `[STASH_DBG step=...]` and `[GATHER_DBG step=...]` debug prints in Block 7 / gather methods are still present from jinzhao/tidar. Remove in a follow-up.

## Continuation recipe

```bash
ssh vp-dgx-51   # 147.68.0.51
cd /data/home/jinzhao/workspace/tidar/Zvllm-v016
git pull origin jinzhao/tidar_v016 --ff-only

# Verify Phase 1 still works (eager smoke test)
export PATH=/data/home/jinzhao/workspace/tidar/Zvllm-v016/.venv-v016/bin:$PATH
export VLLM_ATTENTION_BACKEND=FLEX_ATTENTION
export VLLM_TIDAR_PROPOSAL_ACC_LEVELS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD \
  .venv-v016/bin/python scripts/_sf_mmlu_sweep.py \
    --ckpt /data/checkpoints/smoediffusion_128k_64node-hf/iter_0012000 \
    --thinking off --max-tokens 2048 --max-model-len 4096 \
    --batch 1 --K 16 --n 2 --mode tidar --eager \
    --t-ar 0.0 --dataset aime25 --tag smoke
# Expected: ~40 tok/s, SpecDecoding metrics with non-trivial acceptance
```

## Memory references

- [[project_sf_captured_cudagraph_fixes]] — ported
- [[project_sf_kp1_layout_required]] — respected
- [[project_sf_requires_flex_backend]] — respected
- [[project_sf_multi_call_acceptance_regression]] — preserved (eager-only)
