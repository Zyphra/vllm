# jinzhao/tidar_v016 — WIP port status (updated 2026-05-30)

Branched from `origin/smoe-aiter-moe` on 2026-05-30 to port `jinzhao/tidar`'s
TiDAR (TF + SF) onto vLLM v0.16.0. Phase 1 scope: SF-only minimum viable,
eager mode first.

## Current state

| commit | what | status |
|---|---|---|
| `b54275125` | tidar-only files + small shared (metadata, env_override, speculative) | ✓ imports |
| `b106800a0` | brute-force copy flex_attention.py + gpu_model_runner.py from jinzhao/tidar; first 2 import-path fixes | ✗ many APIs drifted |
| `dd7c7caaf` | flex_attention.py batch fixes; first wave gpu_model_runner.py fixes | flex_attention.py imports ✓ |
| `c527ac174` | more gpu_model_runner.py import fixes; **announced strategy pivot** | gpu_model_runner.py still failing |
| **`HEAD` (uncommitted)** | **gpu_model_runner.py reset to aiter-moe — clean baseline** | all targeted imports pass |

The full vLLM build succeeded on vp-dgx-51 with `VLLM_USE_PRECOMPILED=0`
in `/data/home/jinzhao/workspace/tidar/Zvllm-v016/.venv-v016` (~20 min,
torch 2.9.1+cu126, vllm-0.16.1.dev37+gb106800a0.cu128).

## What works now (after the reset)

```python
import vllm                                            # OK
from vllm.config.speculative import SpeculativeConfig  # OK, has use_tidar()
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata  # OK, has draft_probs/draft_logits
from vllm.v1.spec_decode.tidar import TiDARProposer    # OK
from vllm.v1.spec_decode.tidar_single_forward import * # OK (verbatim)
import vllm.v1.attention.backends.flex_attention       # OK (after batch fixes)
from vllm.v1.worker.gpu_model_runner import GPUModelRunner  # OK (aiter-moe's version, NO TiDAR hooks yet)
from vllm import LLM                                   # OK
```

## What still needs to be done

After my reset, `gpu_model_runner.py` is **aiter-moe's pristine version with
no TiDAR hooks**. SF won't actually function until the following blocks are
re-injected. I've labeled them by what they do and roughly where they go.
The reference source for each is `origin/jinzhao/tidar:vllm/v1/worker/gpu_model_runner.py`.

| # | block | size | what | injection point in aiter-moe gpu_model_runner.py |
|---|---|---|---|---|
| 1 | `_draft_probs_by_req_id` + `_draft_logits_by_req_id` dicts | ~12 lines | per-step stash of drafter outputs keyed by req_id | end of `GPUModelRunner.__init__` |
| 2 | `_gather_draft_probs()` method | ~25 lines | reassemble per-req draft_probs in input-batch order for the next step | new method on GPUModelRunner |
| 3 | `_gather_draft_logits()` method | ~25 lines | parallel plumbing for raw drafter logits (mix-logit v1 needs these) | new method on GPUModelRunner |
| 4 | `spec_decode_metadata.draft_probs/draft_logits` population | ~10 lines | wire the gathered dicts onto the metadata dataclass each step | wherever aiter-moe builds the SpecDecodeMetadata for spec-decode |
| 5 | v1 mix-logit + no_bonus sampler block | ~50 lines | the `if _mix_dt_v1` / `elif _drafts_only` / etc. branches that intercept `target_logits` before `self.rejection_sampler(...)` | wherever aiter-moe calls `self.rejection_sampler` |
| 6 | TiDAR SF drafter-prep | ~80 lines | when `tidar_drafter.single_forward_mode`, take the `_propose` early-exit path that calls `extract_drafts_from_hidden` instead of `self.drafter.propose(...)` | wherever aiter-moe invokes the drafter |
| 7 | drafter stash hook | ~20 lines | after each `self.drafter.propose(...)`, copy `tidar_drafter.last_draft_probs` and `last_draft_logits` into the per-req dicts | right after the propose call |

Total estimated injection: ~220 lines spread across 6-7 distinct anchor
points. Each anchor needs to be located in aiter-moe's restructured
file by grepping for the surrounding context (`bonus_logits_indices`,
`rejection_sampler`, `self.drafter.propose`, etc.).

After all blocks are injected:
- Re-add the SF capture-shape override to `vllm/config/vllm.py` (Phase 2; only matters for captured mode)
- Smoke-test SF eager mode on `iter_0012000` AIME25 thinking-off

## Anticipated runtime breakages after injection

Even with all hooks injected, runtime can break because:
1. **`EagleProposer` base class API may have changed** — TiDARProposer extends EagleProposer; if `propose`, `prepare_inputs`, etc. have new signatures, our overrides break.
2. **`SpecDecodeMetadata.make_dummy` signature** — aiter-moe added `cu_num_sampled_tokens`; need a kwarg-default-free call.
3. **FlexAttentionMetadata SF fields** — we add `tidar_single_forward_proposal_acc_levels`; aiter-moe added `logical_block_ids` etc. The two field sets must coexist in the new dataclass.
4. **CCA model arch** — aiter-moe has their own SMoE CCA impl; our `vllm/v1/spec_decode/cca.py` may reference fields/methods their CCA doesn't expose.

## Continuation recipe

```bash
ssh vp-dgx-51  # or 147.68.0.51
cd /data/home/jinzhao/workspace/tidar/Zvllm-v016
git status                                          # should be clean after the reset commit lands
git pull origin jinzhao/tidar_v016 --ff-only

# verify imports still work after pull
source .venv-v016/bin/activate
python -c "from vllm.v1.worker.gpu_model_runner import GPUModelRunner"

# diff aiter-moe vs jinzhao/tidar to find injection contexts:
git diff $(git merge-base origin/smoe-aiter-moe origin/jinzhao/tidar) origin/jinzhao/tidar -- vllm/v1/worker/gpu_model_runner.py | less

# Inject blocks 1-7 per the table above into vllm/v1/worker/gpu_model_runner.py
# (a careful 3-5h task)

# Test: smoke run on iter_0012000
VLLM_TIDAR_PROPOSAL_ACC_LEVELS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
VLLM_ATTENTION_BACKEND=FLEX_ATTENTION \
python scripts/_sf_mmlu_sweep.py \
  --ckpt /data/checkpoints/smoediffusion_128k_64node-hf/iter_0012000 \
  --thinking off --max-tokens 128 --max-model-len 4096 \
  --batch 1 --K 16 --n 1 --mode tidar --eager \
  --t-ar 0.0 --dataset aime25 --tag v016_smoke
```

## Memory references

- [[project_sf_captured_cudagraph_fixes]] — ported
- [[project_sf_kp1_layout_required]] — respected
- [[project_sf_requires_flex_backend]] — respected
- [[project_sf_multi_call_acceptance_regression]] — preserved (eager-only)
