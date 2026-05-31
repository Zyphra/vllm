# jinzhao/tidar_v016 — Phase 1+2 SF-only port: ✅ DONE (2026-05-30)

Branched from `origin/smoe-aiter-moe` on 2026-05-30 to port `jinzhao/tidar`'s
TiDAR (TF + SF + samplers) onto vLLM v0.16.0.

## Status

✅ **Phase 1 + Phase 2 functional.** Both eager and captured (cudagraph)
modes boot cleanly on v0.16.0+aiter-moe and run TiDAR with non-trivial
speculative acceptance.

### Smoke results on `iter_0012000` (vp-dgx-51, .venv-v016)

| run | mode | tok/s | result |
|---|---|---|---|
| `v016_TF_smoke` mt=128 n=1 | TF eager | 38.26 | 128/128 tokens ✓ |
| `v016_SF_smoke` mt=128 n=1 | SF eager | 43.80 | 128/128 tokens ✓ |
| `v016_SF_2k` mt=2048 n=2 | SF eager | 41.39 | SpecDecoding mean 1.32 → 15.02 → 17.00 |
| `v016_SF_captured2` mt=512 n=1 | SF captured (PIECEWISE) | **72.11** | clean, **1.75× over eager** |
| `v016_SF_cap_2k` mt=2048 n=2 | SF captured | 50.45 | windows: 14.43 → 17.00 → 6.04 → 1.30 → 1.23 |

## Build infrastructure (already done; reuse, don't rebuild)

- Worktree: `/data/home/jinzhao/workspace/tidar/Zvllm-v016` on **vp-dgx-51** (`147.68.0.51`)
- Venv: `.venv-v016/` (must `source` it, or prepend `.venv-v016/bin` to `PATH` for cmake 4.3.2)
- vllm package: `vllm-0.16.1.dev37+gb106800a0.cu128`, torch 2.9.1+cu126, Triton 3.5.1
- Built with: `VLLM_USE_PRECOMPILED=0 pip install -e . --no-build-isolation`
- Build deps: `cmake>=3.26.1, setuptools>=77,<81, setuptools-scm>=8.0, numpy, grpcio-tools==1.78.0, datasets`

## Commits on jinzhao/tidar_v016

```
e3ae8552e feat(v016 port): Phase 2 captured mode — Block 8 cuda_graph_sizes override + dispatcher signature fix
2587d0ea2 docs(v016 port): Phase 1 SF-only eager DONE
347bc46b2 feat(v016 port): 7 TiDAR injection blocks + runtime fixes — pipeline runs end-to-end
... (WIP commits)
72effe5ad Update qwen3 reasoning parser   ← aiter-moe HEAD baseline
```

## What's done

### Phase 1 (eager)
- TiDAR-only files copied verbatim (tidar.py, tidar_single_forward.py, sf_attention.py, cca.py, scripts, docs)
- 7 TiDAR hooks injected into aiter-moe's gpu_model_runner.py:
  - Block 1: `_draft_probs_by_req_id` + `_draft_logits_by_req_id` dicts
  - Block 2: `_gather_draft_probs()` method
  - Block 3: `_gather_draft_logits()` method
  - Block 4: SpecDecodeMetadata.draft_probs/draft_logits population
  - Block 5 (minimal): thread draft_probs through to rejection_sampler
  - Block 6: TiDAR SF early-exit (extract_drafts_from_hidden)
  - Block 7: drafter stash hooks after propose
- TiDARProposer dispatch case (was falling into EagleProposer)
- Runtime API drift fixes: Mamba whitelist, dummy_run, propose signature,
  block_size lookup, CommonAttentionMetadata constructor, use_cuda_graph guard,
  BatchDescriptor rename

### Phase 2 (captured)
- **Block 8** in `vllm/config/vllm.py`: auto-populate `cudagraph_capture_sizes`
  with multiples of K+1 (TF verify) and sf_per_req (SF combined-forward)
  when TiDAR is detected. Avoids default stride-8 missing K+1=17.
- `cudagraph_dispatcher.dispatch()` call site updated for v0.16 signature
  (individual kwargs `num_tokens=`, `uniform_decode=`, not a BatchDescriptor)

## Known limitations / follow-ups

| | what | impact |
|---|---|---|
| **PIECEWISE only** | `CCAAttentionBackend` in v0.16 doesn't support `FULL_DECODE_ONLY` cudagraph mode with spec-decode. Falls back to PIECEWISE. | Limits captured perf vs the v0.15.x handoff's 302 tok/s (FULL_DECODE_ONLY). Current best: 72 tok/s. |
| **Flat per-pos acceptance** | All K=16 per-position rates within a SpecDecoding window are identical (e.g., all 0.839, all 0.315). On v0.15.x we saw the typical pos0 > pos1 > ... decay. | Suggests SF early-exit (Block 6) may not be firing — we may actually be running in TF mode under the hood despite SF being the default. The `last_inflated_total` precondition might not be met. Needs investigation. |
| **Block 5 minimal** | Full v1 mix-logit / v2 / no_bonus / drafts_only / ar_only sampler block from jinzhao/tidar NOT ported. v0.16's `rejection_sampler` signature changed (handles target/bonus splitting internally). | mix-logit features unusable on this branch until rejection_sampler call site is rebuilt. |
| **Debug prints** | `[STASH_DBG step=...]` and `[GATHER_DBG step=...]` debug prints from jinzhao/tidar still fire on first 3 steps. | Cosmetic; remove later. |

## Continuation recipe

```bash
ssh vp-dgx-51   # 147.68.0.51
cd /data/home/jinzhao/workspace/tidar/Zvllm-v016
git pull origin jinzhao/tidar_v016 --ff-only

export PATH=/data/home/jinzhao/workspace/tidar/Zvllm-v016/.venv-v016/bin:$PATH
export VLLM_ATTENTION_BACKEND=FLEX_ATTENTION
export VLLM_TIDAR_PROPOSAL_ACC_LEVELS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16

# eager smoke
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD \
  .venv-v016/bin/python scripts/_sf_mmlu_sweep.py \
    --ckpt /data/checkpoints/smoediffusion_128k_64node-hf/iter_0012000 \
    --thinking off --max-tokens 2048 --max-model-len 4096 \
    --batch 1 --K 16 --n 2 --mode tidar --eager \
    --t-ar 0.0 --dataset aime25 --tag smoke
# Expected: ~41 tok/s, SpecDecoding metrics

# captured smoke (drop --eager)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD \
  .venv-v016/bin/python scripts/_sf_mmlu_sweep.py \
    --ckpt /data/checkpoints/smoediffusion_128k_64node-hf/iter_0012000 \
    --thinking off --max-tokens 2048 --max-model-len 4096 \
    --batch 1 --K 16 --n 2 --mode tidar \
    --t-ar 0.0 --dataset aime25 --tag captured_smoke
# Expected: ~50-72 tok/s, "TiDAR detected: setting cudagraph_capture_sizes" in log
```

## Memory references

- [[project_sf_captured_cudagraph_fixes]] — ported
- [[project_sf_kp1_layout_required]] — respected
- [[project_sf_requires_flex_backend]] — respected
- [[project_sf_multi_call_acceptance_regression]] — preserved (eager-only)
