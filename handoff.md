# TiDAR Single-Forward — Handoff

## TL;DR

- **Branch:** `jinzhao/tidar_SF` (this branch). Default behavior = the **K+1 SF layout** with `VLLM_TIDAR_SINGLE_FORWARD=1`. Best per-batch configs reach 88–106% of TF tok/s on AIME25 thinking-off with K=16.
- **What works:** the K+1 SF layout (see "SF input layout" section below). On a 1.5B SMoE diffusion checkpoint `iter_0012000`.
- **What was explored but doesn't work**: K-mask proposal layouts (training/paper-aligned) underperform K+1 by ~22% mean acceptance on this checkpoint. See [SF_proposal_layouts.md](SF_proposal_layouts.md) for the full investigation (ASCII sketches of every layout tried, including failed alignment experiments).
- **Next step (this handoff target):** a **custom kernel for the K+1 layout** to push throughput further. Current ceiling at b=1 is 289 tok/s (vs TF 272 → SF already beats TF). Profile data + kernel-work pointers in the "Kernel optimization context" section right below.

## Kernel optimization context (next-step work)

This branch is being handed off for **custom kernel development targeting the K+1 SF layout**. Goal: improve throughput beyond the current ceiling, which on this checkpoint is set by the SF attention forward + MoE routing.

### Where time is spent (the SF forward at K=16, P=17, b=1)

Per SF step, the model runs ONE forward pass over a per-request tensor of shape
`[verify_len + P * (K+1)] = [17 + 17*17] = 306` tokens — verify segment (17) plus 17 proposal blocks of 17 masks each. The forward includes:

1. **Self-attention** over the 306-token-per-req input, with a custom mask pattern (see "Attention pattern" below). This is the dominant kernel cost — the proposal blocks each require bidirectional attention WITHIN block plus selective causal attention to the verify segment's first `p_j+1` tokens.
2. **MoE routing + experts**: feed-forward through the SMoE layers. Currently single-GPU expert-parallel; routing is data-dependent so the routing pattern varies step-to-step.
3. **LM-head + draft sampling**: one logit projection per mask position (K+1 per proposal × P = 17×17 = 289 mask positions) → drafter softmax/argmax.

The current paged Triton attention kernel (`vllm/attention/ops/sf_attention.py`, gated by `VLLM_TIDAR_SF_TRITON=1`) gave **3.2× speedup over the naive padded-prefix path** and includes `@triton.autotune` over BLOCK_Q/BLOCK_KV/num_warps/num_stages keyed on `(verify_len, Kp1, P_props, num_heads, head_dim, block_size)`. Autotune fires once during eager warmup; cudagraph capture replays the tuned config.

### Attention pattern (the kernel target)

Per-request input layout (RoPE positions and slot indices for K=3 sketch):

```
                                 ←——  verify   ——→  ←——  proposal_1 (acc p_1=0) ——→  ←——  proposal_2 (acc p_2=5) ——→  ...
input:    [ AR prefix in KV ]   [ a, d_1..d_K ]   [ m, m, ..., m   (K+1 masks)  ]   [ m, m, ..., m   (K+1 masks)  ]  ...
RoPE:                           [ b, b+1..b+K ]   [ b+1, b+2, ..., b+K+1        ]   [ b+6, b+7, ..., b+K+6        ]  ...
                                 slot:                  0   1            K              0    1               K
```

The custom attention mask (from `vllm/v1/spec_decode/tidar_single_forward.py::tidar_mask_mod`):

- **Verify queries (slot 0..K of verify)**: causal among themselves + causal to the AR prefix. Standard causal.
- **Proposal-block queries (each of P blocks of K+1 masks)**: 
  - Causal to the AR prefix (full prefix visibility).
  - Causal to verify slots `[0..p_j]` of the verify segment (the assumed-accepted prefix at this acc level).
  - **Bidirectional WITHIN the proposal block** (all K+1 mask slots see each other).
  - **NO attention** to other proposal blocks or to verify slots `> p_j`.

This is a structured block-diagonal-with-causal-prefix mask. The Triton kernel exploits this by skipping blocks that don't attend to each other; further optimization could fuse the per-proposal attention into a single kernel launch (currently P separate launches via the FlexAttention path; the paged Triton variant batches more aggressively).

### Performance baseline (current ceiling to beat)

AIME25 thinking-off, K=16, max_tokens=10000, T_AR=0:

| batch | TF tok/s | SF tok/s (current best, Triton + autotune) | SF acc | best P |
|---:|---:|---:|---:|---|
| 1  | 271.8 | **289**  | 10.06 | P=17 `[0..16]` |
| 8  | 1055.2 | **926**  | 9.28  | P=3  `[0,6,16]` |
| 16 | 1278.2 | **1146** | 10.74 | P=5  `[0,2,5,10,16]` |

SF beats TF at b=1 (+6.4%) because higher draft acceptance (10.06 vs 8.00) more than offsets per-forward cost. At b=8/16, SF reaches 88–90% of TF tok/s with +20% acceptance — these are the kernel-bound regimes where a better attention kernel can push closer to (or past) TF tok/s.

### Where to look in code

- **`vllm/attention/ops/sf_attention.py`** — paged Triton attention kernel for SF. Current home of the autotuned implementation. Main entry point for kernel work.
- **`vllm/v1/attention/backends/flex_attention.py`** — the three SF forward functions (`_sf_split`, `_sf_triton_paged`, `_multi_call_forward`), all of which hardcode `total_per_req = verify_len + P_props * (K_drafts+1)` and `proposal_seg_len = K_drafts+1` for the K+1 layout. Wires the Triton kernel + FlexAttention fallback.
- **`vllm/v1/spec_decode/tidar_single_forward.py`** — `tidar_mask_mod` (the canonical attention mask function), `compute_position_offsets` (RoPE position assignment per slot), `build_single_forward_inputs` (assembles the per-req SF input tensor), `extract_proposal_hidden_states` (post-forward draft extraction).
- **`vllm/v1/spec_decode/cca.py`** — combined causal attention module (handles the SF metadata in the CCA codepath).
- **`vllm/v1/spec_decode/tidar.py`** — `maybe_extend_verify_input` builds the per-step SF input incl. scratch KV slots for proposal blocks; `set_tidar_single_forward_metadata` attaches the metadata for the FA / FlexAttn builders to consume.

### Existing optimization knobs (already shipped)

| env var | effect | gain |
|---|---|---|
| `VLLM_TIDAR_SF_TRITON=1` (now the default) | use paged Triton attention kernel | 3.2× over the historical naive padded-prefix path; ~1.8× over `_sf_multi_call_forward` (the SF_TRITON=0 path) at b=1 P=17 in **eager mode**. The bigger reason to keep it on: multi-call FA was measured to depress mean acceptance ~50% vs Triton on the same config — **but only in eager mode**. In captured (cudagraph) mode, multi-call FA is bypassed (its per-request prefix slicing uses Python ints baked as constants under cudagraph), so SF_TRITON=0 falls through to FlexAttention and the acceptance regression doesn't apply. Triton is still recommended for both modes — also the faster path under capture |
| `VLLM_TIDAR_SF_SPLIT=1` | FA3-verify + Triton-proposals split | neutral on small batches; flips at large `verify_len` / small P |
| Triton autotune | per-shape BLOCK_Q/BLOCK_KV/warps/stages | +4% to +47% across batch×P |

### Suggested directions for kernel work

1. **Fuse per-proposal-block attention** into a single launch with metadata-driven block masking. Currently the multi-call FA path issues P separate FlashAttention calls; the paged Triton kernel batches more but could batch further.
2. **Fuse extract + LM-head + draft sampling** into a single kernel — currently three separate ops (`extract_proposal_hidden_states` → linear → softmax/argmax) that touch the K+1 hidden-states tensor multiple times.
3. **Improve KV-cache layout for the SF scratch blocks** — proposal blocks use scratch KV slots that are allocated lazily; better cache locality between verify-segment KV and proposal-segment KV could reduce L2 pressure.
4. **Audit the autotune key**: currently keyed on `(verify_len, Kp1, P_props, num_heads, head_dim, block_size)`. If `P_props` is variable across batches, the autotune cache might miss frequently; pinning to a small set of pre-tuned shapes could help.

### Correctness check

Any new kernel must reproduce:
- The per-position acceptance curve from the reference K+1 baseline (Scheme 1 in `SF_proposal_layouts.md`).
- AIME25 thinking-off n=8 mt=2000 should give pos0 ≈ 0.74, mean ≈ 6.66, tok/s ≥ 48.67 at K=16 dense P=17.
- Generated outputs should match the reference (or, more leniently, produce sensible reasoning with EOS termination — not degenerate token loops like the failed experiments in `SF_proposal_layouts.md` schemes 6–9).

### Parallel investigation: layout exploration (concluded — don't redo)

`SF_proposal_layouts.md` in this repo documents the investigation of alternative SF layouts (K-mask, NO_ANCHOR, NOSLOT0, LAST_ACC_ANCHOR, etc.). **Conclusion: K+1 is the right layout for this checkpoint; kernel work should target K+1 specifically.** Read that doc for the full empirical comparison + the failed alignment experiments (one of which produced spectacular-looking spec-decode metrics with degenerate token-loop output — useful as a correctness warning).

---

## SF input layout (what we actually ship)

Single forward pass per spec-decode step. Per-request input fed to the model:

```
                                 ←—  verify   ——→  ←——  proposal_1 (acc p_1=0) ——→  ←——  proposal_2 (acc p_2=5) ——→ ...
input:    [ AR prefix in KV ]   [ a, d_1..d_K ]   [ m, m, ..., m   (K+1 masks)  ]   [ m, m, ..., m   (K+1 masks)  ] ...
RoPE:                           [ b, b+1..b+K ]   [ b+1, b+2, ..., b+K+1        ]   [ b+6, b+7, ..., b+K+6        ] ...
                                 slot:                  0   1            K              0    1               K
                                                    ^ bonus position             ^ bonus position
```

Components:
- **Verify segment (K+1 slots):** slot 0 `a` = previous step's anchor (real token: either the committed bonus from previous all-accept, or the recovered token at the first rejected draft); slots 1..K = K drafts proposed by step n-1's drafter. Causal attention.
- **Proposal block at acc level p_j (K+1 slots), repeated for each of P proposals:** all K+1 slots are `mask_token_id=4`. Bidirectional within the proposal block, attends to AR prefix causally + verify[0..p_j] (anchor + first p_j drafts, "assumed accepted") causally. RoPE positions span `[b+p_j+1, ..., b+p_j+K+1]`.
- **Extract:** drop slot 0's hidden state (the bonus-position prediction), keep slots 1..K → K drafts for next step.
- **Verifier (next step):** standard causal rejection sampling. On all-accept, bonus is sampled from the K+1-th logit slot and committed → becomes the next step's anchor `a`.

This **K+1 layout** came from TF Fix 4 (commit `fef304428`, 2026-05-15). Earlier SF tried K masks per proposal with a "gap" at the bonus position — the model collapsed to mask_token_id with 100% acceptance of garbage. TF Fix 4 added the K+1-th mask to make the mask block contiguous; slot 0 functions as the bidirectional-attention anchor that slots 1..K condition on.

## How K+1 differs from training and the paper

### SBD training (`ibm-head:/shared/home/rishi/Megatron-LM-tom-bmoe`)

```python
# cmoe_model.py
parallel_input = torch.where(should_mask, mask_token_id, input_ids)   # masked copy
sbd_input_ids  = torch.cat((input_ids, parallel_input), dim=1)        # length = 2*seq_len
mask_fn = partial(sbd_mask_mod, causal_len=seq_len, block_size=16)    # block_size = K
```

```
                  ←—— causal region ——————→  ←——— mask region ———→
                  block 0  block 1 ... block_{N/K-1}   block 0' block 1' ... block_{N/K-1}'
                  (orig)   (orig)     (orig)            (masked  (masked     (masked
                                                         copy)    copy)       copy)
```

Training attention rules (`sbd_mask_mod`):
- Causal region: standard causal.
- Mask block i (in mask region) → causal region: **block-causal**, sees blocks `0..i-1` only (strict; does NOT see block i = its own "target").
- Mask block i ↔ mask block i: **bidirectional** within block; other mask blocks masked out.

Key contrasts vs what we ship:

| | Training (SBD) | SF inference (K+1 we ship) |
|---|---|---|
| Mask block size | K masks/block | **K+1** masks/block |
| Mask block role | Predict ORIGINAL block i (positions `i*K..(i+1)*K-1`) | Predict FUTURE positions `p_j+1..p_j+K+1` |
| Mask block context | Block-causal to blocks 0..i-1, bidir self | Full causal to AR prefix, visible verify[0..p_j], bidir self |
| Mask block 0 | Sees NO causal blocks (since `0 > kv_block` is empty) | Sees AR prefix + verify[0..p_j] |
| Sequence structure | Doubled `[original | masked copy]` | Single linear `[prefix | verify | proposal_1 | ... | proposal_P]` |
| Anchor token | None | Slot 0 of verify = previous bonus (real token) |
| Slot-0 mask role | N/A | "Bonus-position prediction" — discarded at extract |

**SF inference is not training-faithful in either K or K+1 layout** (training has 1 mask block; inference has P+1 mask blocks at specific future-position offsets). TF Fix 4 is an empirical adjustment that happened to work on this checkpoint.

### TiDAR paper (arxiv:2511.08923) Figure 3

```
[ AR prefix ]  [ d_1..d_K ]  [ m, m, ..., m   (K masks) ]
                  ↑                  ↑
                verify             proposal
                (K slots,         (K masks, BIDIRECTIONAL within;
                 causal)           attends causally to prefix + verify)
```

Paper differences from what we ship:
- **No anchor** in verify (paper: K drafts only; ours: K+1 = anchor + K drafts).
- **No bonus** committed by verifier (paper: bonus emerges implicitly; ours: bonus sampled and committed on all-accept).
- **Single proposal block** in the figure (we ship: P parallel proposals at different acc levels — our SF "multiple-hypothesis" elaboration on top of the paper).
- **K masks per proposal** (paper-aligned to SBD training; we ship K+1 per TF Fix 4).

The paper's "zero-shot K" claim doesn't hold downward on this checkpoint (K<16 has a catastrophic last-3-positions cliff — see "K-generalization sweep"). The paper's "K-mask + no-bonus" gives 0% acceptance on this checkpoint (see "No-bonus exploration").

## Environment variables

**SF is the default mode since the `tidar_TF + tidar_SF` merge.** Set
`VLLM_TIDAR_TWO_FORWARD=1` to switch to TF. The legacy
`VLLM_TIDAR_SINGLE_FORWARD=1` is still recognized but redundant.

| Var | Default | Purpose |
|---|---|---|
| `VLLM_TIDAR_TWO_FORWARD` | `0` (SF mode) | Set to `1` to switch to TF (two-forward TiDAR). When unset (or `0`), runtime uses SF |
| `VLLM_TIDAR_SINGLE_FORWARD` | (legacy) | Redundant since SF is now default. Setting it to `1` is a no-op kept for backward compatibility; setting it to `0` does NOT disable SF (use `VLLM_TIDAR_TWO_FORWARD=1` for that) |
| `VLLM_TIDAR_PROPOSAL_ACC_LEVELS` | `(4,7,10)` (deprecated; see deployment recipe) | Comma-separated proposal acc levels (P inferred from count) |
| `VLLM_TIDAR_SF_TRITON` | **`1` (default since 2026-05-30)** | Paged Triton SF attention kernel. Set to `0` to fall back to `_sf_multi_call_forward` (eager) / FlexAttention (captured). The eager fallback (`_sf_multi_call_forward`) was measured to depress mean acceptance ~50% vs Triton at K=16/P=17 (6.21 vs 9.58 on AIME25 thinking-off; Triton matches handoff's stated 10.19 acceptance, multi-call does not) — **not recommended in eager mode**. The captured fallback (FlexAttention) is correct but slower; Triton still preferred under capture for perf |
| `VLLM_TIDAR_SF_SPLIT` | `0` | FA3-verify + Triton-proposals split path (opt-in; flips trade at large verify_len / small P) |
| `VLLM_ATTENTION_BACKEND` | (auto) | Set to `FLEX_ATTENTION` for SF |
| `VLLM_TIDAR_NO_BONUS` | `0` | (branch `jinzhao/tidar_sf_kmask_no_bonus`) zap verifier bonus; see "No-bonus exploration" |
| `TIDAR_MIX_DRAFT_TARGET_V1` | unset (off) | Set to `1` to enable v1 mix-logit sampling — mixes draft and target logits before the rejection sampler. See "Mix-logit v1 sampler" below |
| `TIDAR_MIX_LOGIT_TARGET_WEIGHT` | `0.5` | `w ∈ [0,1]` for v1 mix; `mixed = w·target + (1−w)·draft`. Only consulted when `TIDAR_MIX_DRAFT_TARGET_V1=1` |

### Mix-logit v1 sampler

Ported from `jinzhao/tidar` commit `583479480` (which itself fixed the
Dirac-drafter q-mismatch introduced by `6597f9151`).

**What it does:** when `TIDAR_MIX_DRAFT_TARGET_V1=1`, the K target rows
fed to the rejection sampler are replaced by

    mixed_logits = w · target_logits + (1 − w) · draft_logits

with `w = TIDAR_MIX_LOGIT_TARGET_WEIGHT` (default `0.5`). The drafter's
`draft_probs` is forwarded **unchanged** to the sampler — it is *not*
recomputed from the mixed logits.

**Why this preserves correctness at greedy drafter (`T_diff=0`):** the
drafter sets `spec_decode_metadata.draft_probs = None` to signal Dirac,
and the rejection sampler's `NO_DRAFT_PROBS=True` kernel branch accepts
the draft argmax with probability `target_prob(argmax)` and resamples
target-masked-at-argmax on reject. **Do not** synthesize a soft
`q = softmax(draft_logits)` to fill the `None` here — that's the exact
bug `583479480` fixed: the fake-q argmax probability is ~0.3, making
`min(1, p/q)` hit the auto-accept short-circuit far more often than the
true delta distribution warrants, biasing accepts toward the drafter
argmax independent of `w`.

**Activation guard:** the v1 branch is silently skipped (target_logits
unchanged) unless all of: env var is set to non-empty/non-`0`/non-`false`;
`spec_decode_metadata.draft_logits is not None`; `sampling_metadata.temperature`
is set; `bonus_logits_indices` is non-empty. This makes the feature
no-op on cold-prefill / empty-spec steps without crashing.

**Scope of what was ported (v1 only):** the v2 direct-sample-from-mixed
path (`TIDAR_MIX_DRAFT_TARGET`, no `_V1`), the entropy / min-entropy
thresholds (v1 & v2), top-P v2, the per-token entropy logger, and the
bonus-entropy threshold were **not** ported from `jinzhao/tidar`. Only
the rejection-sampler-compatible v1 mix lives on `jinzhao/tidar_SF`.

**Side effect of the port:** `spec_decode_metadata.draft_logits` is now
always populated (parallel plumbing to `draft_probs`) so the v1 path
has something to mix when the drafter is Dirac. One extra gather per
step, irrelevant compared to a forward. The field stays `None` if the
drafter didn't run this step (cold prefill / empty spec) and the v1
path is skipped.

**Interaction with the kernel work:** none — the mix happens *after*
the verify forward returns and *before* `self.rejection_sampler(...)`,
so attention kernels are untouched. The kernel-target branch
(`VLLM_TIDAR_SF_TRITON=1`) is still the right path. If you want a
sanity check while iterating on kernels, set
`TIDAR_MIX_DRAFT_TARGET_V1=1 TIDAR_MIX_LOGIT_TARGET_WEIGHT=1.0` (pure
target, zero draft contribution) — should produce numerically identical
decode to the feature-off run.

#### Usage

**Minimum command to turn it on** (composes with the standard SF env
block, doesn't replace anything):

```bash
export VLLM_TIDAR_SINGLE_FORWARD=1
export VLLM_ATTENTION_BACKEND=FLEX_ATTENTION
export VLLM_TIDAR_PROPOSAL_ACC_LEVELS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16

# Enable v1 mix-logit, w=0.5 (default)
export TIDAR_MIX_DRAFT_TARGET_V1=1
# Optional: pick a different mix weight
# export TIDAR_MIX_LOGIT_TARGET_WEIGHT=0.7    # 70% target, 30% draft

python scripts/_sf_mmlu_sweep.py ... --t-ar 0.05 --K 16 ...
```

**How to verify it's actually firing.** The activation guard is
silent — if any condition fails, the mix just doesn't happen and you
get baseline rejection sampling. To confirm the v1 path is hot, the
cheapest check is the `w=1.0` ≡ feature-off identity:

```bash
# Should produce bit-identical output to a feature-off run (same seed,
# same prompts). Any drift means something else is wrong.
TIDAR_MIX_DRAFT_TARGET_V1=1 TIDAR_MIX_LOGIT_TARGET_WEIGHT=1.0 python ...
```

Then sweep `w ∈ {0.0, 0.25, 0.5, 0.75, 1.0}` and look at acceptance
rate / generation quality vs. `w`. The w=0 row is "draft-only target"
— gives the most aggressive accept (the target is replaced by the
drafter's own logits, so `min(1, p_v/p_d)` is closest to 1) but the
*meaning* of the accepted sample is "what would the drafter have
sampled with rejection-against-itself" — the verifier's contribution
to per-token quality vanishes. Useful as an upper bound on accept rate
but not as a production setting.

**Recommended starting sweep** for evaluating whether mix-logit helps
on a given ckpt:

| w | meaning | when to try it |
|---|---|---|
| 1.0 | pure target (= feature off) | sanity check / baseline |
| 0.75 | mostly target, draft nudges | conservative — start here for ckpt-quality eval |
| 0.5 | even mix | the commit's validated point |
| 0.25 | mostly draft | aggressive accept; expect inflated tok/s, watch downstream quality |

**Drafter temperature compatibility.** Works at both `t_diff=0`
(Dirac) and `t_diff>0`. Forwarded `draft_probs` is `None` in the Dirac
case (correct; see above) and the actual soft `q` in the sampling
case. The mix formula on `draft_logits` is the same either way — only
the sampler's reject-recovery branch differs.

**Common pitfalls.**
- Setting `TIDAR_MIX_DRAFT_TARGET=1` instead of `TIDAR_MIX_DRAFT_TARGET_V1=1`:
  silently does nothing on this branch (v2 isn't ported). The env var
  name without `_V1` is a no-op here.
- Setting `TIDAR_MIX_LOGIT_TARGET_WEIGHT` without setting
  `TIDAR_MIX_DRAFT_TARGET_V1=1`: the weight is ignored — the entire
  branch is gated on the V1 flag.
- Reading `w` differently from the spec: the convention is **`w` is
  the target weight** (`mixed = w·target + (1−w)·draft`). So `w=0`
  means "pure draft as target", not "pure target". The variable name
  ends `_TARGET_WEIGHT` to disambiguate.
- Expecting the K=16 SpecDecoding `Per-position acceptance rate`
  numbers to change *just from enabling the feature with w=1.0* —
  they won't. Compare runs at different `w`, not feature-on vs
  feature-off at `w=1.0`.

## Deployment recipe (per-batch best configs)

AIME25 thinking-off, K=16, max_tokens=10000, T_AR=0, with `VLLM_TIDAR_SF_TRITON=1` + autotune:

```bash
# b=1  (SF beats TF, P=17 full coverage)
VLLM_TIDAR_PROPOSAL_ACC_LEVELS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16

# b=8  (sparse optimized bins)
VLLM_TIDAR_PROPOSAL_ACC_LEVELS=0,6,16

# b=16 (sparse optimized bins)
VLLM_TIDAR_PROPOSAL_ACC_LEVELS=0,2,5,10,16
```

All require:
```bash
VLLM_TIDAR_SINGLE_FORWARD=1 VLLM_TIDAR_SF_TRITON=1 VLLM_ATTENTION_BACKEND=FLEX_ATTENTION
```

## Performance results (K=16, thinking-off, AIME25, max_tokens=10000)

### TF baseline

| batch | tok/s | num_acc_tokens |
|---:|---:|---:|
| 1  | 271.8 | 8.00 |
| 8  | 1055.2 | 8.02 |
| 16 | 1278.2 | 8.63 |

### SF — best config per batch (Triton + autotune)

| batch | best P / bins | tok/s | %TF | num_acc_tokens | vs TF accept |
|---:|---|---:|---:|---:|---:|
| 1  | P=17 `[0..16]`               | **288.1**  | **106%** | 10.06 | +25% |
| 8  | P=3 opt `[0,6,16]`           | **926.3**  | **88%**  | 9.28  | +16% |
| 16 | P=5 opt `[0,2,5,10,16]`      | **1145.5** | **90%**  | 10.74 | +24% |

SF beats TF at b=1 because the proposal-with-conditioning produces higher-quality drafts than TF's mask-only drafter, and the larger SF forward isn't bandwidth-bound at low batch. At b>1, FA3 begins to dominate on verify-side attention scaling; SF still recovers ~88-90% of TF tok/s while delivering ~+20% higher acceptance.

### Optimized acc_levels (bins)

Empirical num_accepted distribution at TF AIME25 thinking-off b=1 (15 windows) is bimodal: P(N=0) ≈ 27%, P(N=16) ≈ 28%, ~5% mass in the middle. Distance-minimization with `closest_below` tie-break gives:

| P | original bins | optimized bins | rationale |
|---:|---|---|---|
| 2 | `[0, 5]` | `[0, 14]` | cover the N=16 mode |
| 3 | `[0, 5, 10]` | `[0, 6, 16]` | add level 16 |
| 5 | `[0, 3, 7, 11, 14]` | `[0, 2, 5, 10, 16]` | denser low + cover 16 |
| 9 | `[0, 2,4,6,8,10,12,14,16]` | `[0,1,2,4,6,8,10,13,16]` | minor refinement |
| 17 | `[0..16]` | same | already optimal |

Optimized bins consistently improved tok/s + acceptance at b=8/16 (e.g., P=3: 826 → 918 tok/s at b=8, +11%); at b=1 the gain is small because every config saturates close to TF.

### Thinking-ON SF tradeoff

Same AIME25 b/P sweep with thinking enabled:

| batch | TF tok/s | TF accept | best SF tok/s | best SF accept | %TF |
|---:|---:|---:|---:|---:|---:|
| 1  | 218  | 5.85 | P=17 = 141     | 5.51 | 65% |
| 8  | 1162 | 5.84 | P=3 opt = 601  | 4.32 | 52% |
| 16 | 1865 | 5.93 | P=3 opt = 884  | 4.34 | 47% |

Thinking-ON SF significantly lags TF (vs thinking-off where SF beats TF at b=1). Cause: the bimodal acceptance distribution that powers thinking-off (P(N=16) ≈ 18-65%) collapses into near-geometric decay in thinking-on (P(N=16) ≈ 10%). CoT exploration has high entropy at every token position, so neither TF's drafter nor SF's proposal-with-conditioning gets the templated-tail acceleration. SF's overhead (~3× compute per step for P=5+) becomes pure cost without the all-accept-mode payoff.

Plot at `docs/imgs/acc_dist_p17_thinking.png`.

## K-generalization sweep

Empirical answer to "the model was trained at K=16 — how does inference at different K values behave?"

Setup: TF and SF swept over K ∈ {4, 8, 12, 16, 20, 24, 28, 32}, AIME25 thinking-off b=1, max_tokens=10000, T_AR=0. SF uses P=K+1 (full-coverage `acc_levels = [0..K]`).

### TF results

| K  | tok/s | mean_acc |
|---:|------:|---------:|
|  4 |  60   |  1.44    |
|  8 | 162   |  4.21    |
| 12 | 243   |  6.58    |
| 16 | 338   |  9.68    |
| 20 | 372   | 10.97    |
| 24 | **414** | **12.31** |
| 28 | 430   | 12.77    |
| 32 | 432   | 12.97    |

K=24-32 saturates around 430 tok/s. Marginal gain past K=24 is <2% per +K.

### SF results (P=K+1)

| K  | tok/s | mean_acc |
|---:|------:|---------:|
|  4 |  57   |  1.63    |
|  8 | 152   |  4.76    |
| 12 | 219   |  7.26    |
| 16 | 289   | 10.19    |
| 20 | **309** | **11.15** |
| 24 | 276   | 11.12    |
| 28 | 272   | 11.13    |
| 32 | crashed (3-block scratch overflow) | — |

SF tok/s peaks at K=20 (not K=24+ like TF) because SF's forward grows as (P+1)·(K+1) — past K=20, extra proposal compute swamps the accept gain.

### Thinking-ON TF

Same setup with thinking on:

| K  | tok/s | mean_acc |
|---:|------:|---------:|
|  4 |  58.8 |  1.39    |
|  8 | 128.2 |  3.27    |
| 12 | 191.0 |  5.01    |
| 16 | 217.8 |  5.85    |
| 20 | 226.9 |  6.19    |
| 24 | **230.0** | **6.30** |
| 28 | 229.9 |  6.38    |
| 32 | 224.3 |  6.28    |

Mean_acc saturates ~6.3; thinking-mode CoT lacks the templated-tail mass that drives thinking-off's K-scaling.

### Asymmetric K-generalization

- **K > 16** generalizes smoothly. Per-position rates extend past trained position 15 with continuous decay. Holds for TF/SF and thinking-on/off.
- **K < 16** has a **catastrophic cliff at the last 3 positions**.
  - K=4: positions 1, 2, 3 ≈ 0 (and pos 0 ≈ 0.44, also broken)
  - K=8: positions 5, 6, 7 ≈ 0
  - K=12: positions 9, 10, 11 ≈ 0
  - Pattern: positions K-3..K-1 are uniformly broken for any K<16

Consistent across TF/SF and thinking-on/off → model-structural property.

### Comparison with the TiDAR paper

The paper claims *"we can adjust the block (draft) length during decoding in a zero-shot manner"* (Limitations section). Evidence presented:
- **Figure 4** plots three *separately trained* 1.5B models, each at native K — not one model at multiple K.
- **Table 4** reports K ∈ {4, 8, 16} on the 1.5B model — same set as continual-pretraining ("under block sizes of 4, 8, and 16").
- **8B model** trained at K=16 only; no across-K test shown.

So "zero-shot K" isn't directly demonstrated by the paper. Our SMoE iter_0012000 (single-K=16 training) is the cleanest probe — upward generalizes, downward breaks.

### Hypotheses for the asymmetric cliff (untested)

1. **Bidirectional within-block attention needs N≥3 lookahead.** At K<16 the last 3 mask positions have <3 future positions, below the threshold the model learned.
2. **Learned block-end role at RoPE positions 13-15.** At K=8, end-of-block sits at RoPE 5-7, where no end-of-block role was learned.
3. **Standard upward-only network extrapolation** — smooth along axes scaled during training (seq length, RoPE), breaks on structural expectations (specific block size).

Distinguish (1) vs (2): run K=15 or K=14. If cliff is always "last 3 positions," (1); if always at RoPE 13-15, (2). Not yet tested.

### Deployment recommendations

- **TF:** K=24 → 22% throughput gain (414 vs 338 tok/s) over K=16. K≥28 saturates.
- **SF:** K=20 is the sweet spot (309 vs 289 at K=16, +7%). Beyond K=20, SF compute grows faster than accept gain.
- **Do not** decrease K below 16 — last-3-positions cliff destroys acceptance.
- **Thinking-on**: K=16 is fine; mean_acc plateaus regardless of K.

### Plots

In project root:
- `K_sweep.png` — TF thinking-off, K=4..32 (shows K<16 cliff + smooth K>16)
- `K_sweep_ge16.png` — TF thinking-off, K≥16 only
- `K_sweep_SF_full.png` — SF P=K+1, K=4..28
- `K_sweep_TF_on.png` — thinking-ON TF
- `K_sweep_SF.png` — SF P=K+1 at K=16 vs K=24 acceptance
- `sf_K24_dist.png` — TF vs SF num_accepted distributions
- `acceptance_distribution.md` — pedagogical writeup of P(N=k) modeling

## No-bonus exploration (concluded)

Question: can we drop both the verifier's bonus commit and the K+1-th proposal mask, recovering the paper's K-mask + no-bonus layout? **Short answer: no, this checkpoint requires K+1 + bonus.** All variants tested on AIME25 b=1 P=[0,5,10] thinking-off, 4 prompts × 512 tokens, node 68 GPU 1, captured cudagraph. Baseline K+1 default ~170 tok/s with mean accept ~5.

| variant | mean accept | per-pos rate | tok/s | verdict |
|---|---|---|---|---|
| K+1 default (no flag) | 5.03 | 0.49 → 0.13 | 171 | baseline ✓ |
| K-mask proposal (`prop_seg_len=K`, bidir, prefix-vis) | 1.00 | **0.000** | 10 | broken |
| K-mask + causal-within-block | 1.00 | 0.000 | 10 | broken |
| K-mask + prefix hidden from proposals (training-faithful) | 1.00 | 0.000 | 8 | broken |
| K-mask + CCA off-by-one fix | 1.00 | 0.000 | 8 | broken |
| K-mask + 2 cache blocks per proposal | 1.00 | 0.000 | 7 | broken |
| K+1 layout + extract slots 0..K-1 | 1.05–3.59 (spiky) | 0.06 → 0.24 → 0.14 | 10 | partial — degraded |
| **Zap-only: K+1 layout, `bonus_token_ids = -1`** | **6.22** | **0.50 → 0.23** | **~110** | **ships under `VLLM_TIDAR_NO_BONUS=1`** |

Findings:

1. **K-mask proposal block is OOD on this checkpoint regardless of attention pattern.** Model produces drafts the verifier rejects 100% of the time. TF Fix 4 hard-tuned slots 1..K of the K+1 proposal block as high-quality draft predictions; removing the K+1-th mask breaks that tuning.

2. **Extract-slot-shift on K+1 layout** is the cleanest "K-mask-equivalent" without changing model input. Keep K+1 proposal block (so model sees TF-tuned layout) but under no_bonus extract slots 0..K-1 instead of 1..K. Positions then align with the no-bonus next-step verifier (base_new = base + p_j; verify slots 1..K at RoPE base+p_j+1..base+p_j+K = extract slots 0..K-1's RoPE). **Result: 1-5% per-position acceptance, occasional good prompts hitting 3.59 mean accept.** Positions align; the *model's slot-0 hidden state is degraded* — TF Fix 4 relegated slot 0 to "bonus-position prediction" with less optimization attention.

3. **Zap-only ships.** Keep K+1 layout everywhere. Under `VLLM_TIDAR_NO_BONUS=1`: overwrite `bonus_token_ids = -1` before the rejection sampler; kernel writes PLACEHOLDER at the bonus column on all-accept; `parse_output()` filters it. Drafter bonus is already discarded by `extract_proposal_hidden_states` (slot 0 dropped). Net: no bonus from either, model layout untouched.

4. **The 35 → 67 ms/step slowdown under no_bonus is NOT the zap mechanics.** Bisect: remove the zap line while keeping the flag set → ms/step returns to baseline. The drop from ~170 → ~110 tok/s is **sequence content divergence + MoE routing variability**: no-bonus sequences diverge from default and happen to hit a slower expert-routing path (~67 ms/forward at the same captured shape). Acceptance is *higher* under no_bonus (drafts cluster around model's preferred predictions) but each MoE forward is slower. Likely cause: more concentrated expert load → all-to-all stalls. Fix requires training-side changes (load balancing), not inference-side.

5. **CCA off-by-one fix lands as a separate bug** (commit `452e8fcdf`). `_commit_tidar_cca_state` used `len(tokens) - 1`. Under K+1 all-accept: `len = K + 1` (K drafts + bonus), formula gives K (correct). Under zap-only all-accept: `len = K` (PLACEHOLDER filtered), formula gives K-1 (off by 1). CCA state drifted by 1 per all-accept step. Marginal throughput effect (~110 → ~113 tok/s).

**Future K-mask work** would require retraining the drafter so slot 0 of the proposal block is promoted to a draft slot (or proposal block size is reduced to K with retrained representations). Inference cannot escape TF Fix 4's slot specialization. See [[project_sf_kmask_no_bonus_dead]] memory note for the full bisect.

## Implementation reference

### Files (new + patched)

**New** (no upstream equivalent):
- `vllm/attention/ops/sf_attention.py` — Triton paged kernel (3.2× speedup over padded-prefix; gated by `VLLM_TIDAR_SF_PAGED=1`)
- `vllm/v1/spec_decode/cca.py` — CCA module
- `vllm/v1/spec_decode/tidar_single_forward.py` — `tidar_mask_mod`, `compute_position_offsets`, `build_single_forward_inputs`, `extract_proposal_hidden_states`, `select_proposal_index`
- `scripts/_sf_mmlu_sweep.py` — multi-batch / multi-dataset benchmark

**Patched** (SF-gated code paths):
- `vllm/v1/spec_decode/tidar.py` — SF mode detection, `_ensure_scratch_blocks`, `maybe_extend_verify_input`, `extract_drafts_from_hidden`, `set_tidar_single_forward_metadata`, pre-capture eager warmup
- `vllm/v1/worker/gpu_model_runner.py` — SF inflation path, inflated `uniform_decode_query_len`
- `vllm/v1/attention/backends/flex_attention.py` — SF metadata fields, mask_mod hook, multi-call FA fallback, `num_computed_tokens_cpu`-based prefix_lens
- `vllm/config/__init__.py` — SF cudagraph capture sizes (`Kp1 + P*K` per req); filter spec_sizes by sf_per_req divisibility; raise `_FA3_FULL_CG_MAX` to 8192
- `vllm/env_override.py` — see "env_override.py patches" below

### env_override.py patches required for captured mode

Six monkey-patches needed for SF's verify forward to capture under `FULL_DECODE_ONLY` cudagraph mode (without these, the first capture raises `cudaErrorStreamCaptureUnsupported`):

1. `_sfdp_init()` pre-init at module load (SDPA pattern matcher lazy init crashes in capture)
2. `torch._inductor.config.pattern_matcher = False`
3. `torch._inductor.config.joint_graph_constant_folding = False` (`aten.full` constant folding crashes in capture)
4. `torch._dynamo.config.cache_size_limit = 256` (SF shape variants exceed default 8)
5. `torch.cuda.get_rng_state` / `set_rng_state` no-op during capture (works around dynamo's seed-during-capture bug fired by flex_attention.forward recompilation)
6. `torch.cuda.synchronize` / `torch._C._cuda_synchronize` no-op during capture (inductor Triton precompile calls synchronize after cache hit)

Committed in `6b95abab1`.

### Triton kernel autotune

`@triton.autotune` over `BLOCK_Q ∈ {16,32,64}`, `BLOCK_KV ∈ {64,128,256}`, `num_warps ∈ {4,8}`, `num_stages ∈ {2,3}`, keyed on `(verify_len, Kp1, P_props, num_heads, head_dim, block_size)`. Tuning fires once during eager warmup; cudagraph capture replays the tuned config. Gain ranges +4% (b=8 P=3) to +47% (b=16 P=17).

### FA3-verify + Triton-proposal split (`VLLM_TIDAR_SF_SPLIT=1`)

Routes the verify segment (K+1 causal queries against paged prefix + just-cached verify K/V) to `flash_attn_varlen_func` with `block_table`, and the proposal segment (P*(K+1) queries with the SF mask) to the Triton kernel with `Q_START=verify_len`. Performance neutral at P=3 b=16 (1124 vs 1106 tok/s) — verify is only ~25% of P=3 attention work and the extra launch + verify-query gather/scatter offsets the FA3 savings. Kept as opt-in since the trade may flip at larger verify_len or smaller P.

### Memory file references

- `project_tidar_sf_prefix_lens_bug.md` — use `num_computed_tokens_cpu`, not `seq_lens - inflate`
- `project_tidar_sf_paged_kernel.md` — paged Triton 3.2× speedup
- `project_sf_acceptance_capped.md` — broken-attention diagnostic (pre-2026-05-16)
- `project_tidar_single_forward_sparse_proposals.md` — sparse P preferred over dense
- `project_sf_proposal_levels.md` — optimal acc_levels per batch
- `project_sf_kmask_no_bonus_dead.md` — no-bonus exploration writeup
- `feedback_always_log_stats.md` — pass `--log-stats` for acceptance metrics

### Branches and commits

**`jinzhao/tidar_single_forward_fixed`** (main SF dev branch):
```
b886e39af docs(handoff): K+1 SF layout reference + no-bonus exploration writeup
ca5b16235 docs(tidar-sf): K-generalization sweep findings
d35dee722 docs(tidar-sf): thinking-mode comparison + K-mask attempt findings
cfc66d572 docs(tidar-sf): autotune + opt bins + 10k-cap results
55fd2a2eb feat(tidar-sf): FA3-verify + Triton-proposal split path
07dba8df3 perf(tidar-sf): autotune paged Triton kernel block sizes
171d283fc perf(tidar-sf): wire paged Triton attention kernel (2-3× speedup)
8da219a9a fix(tidar-sf): SF metadata + correct prefix_lens for b≥1 captures
6b95abab1 fix(env_override): patches for cudagraph capture of SF verify forward
e9986f743 feat(tidar-sf): restore K+1 single-forward layout
```
Base: `f272b37c9` (`jinzhao/tidar`).

**`jinzhao/tidar_sf_kmask_no_bonus`** (no-bonus exploration, mergeable):
```
452e8fcdf [SF K-mask] Fix CCA num_accepted off-by-one under no_bonus
9c53a9760 [SF K-mask] Preempt bonus_token_ids = -1; remove post-rejection zap
162ea4aa6 [SF K-mask] Reduce to zap-only: K+1 layout always, only bonus output zapped
... + earlier exploratory commits (K-mask attempts, hybrid layout, dispatch debug)
```

## Test commands

```bash
# Sanity TF (no SF)
CUDA_VISIBLE_DEVICES=2 /data/home/jinzhao/workspace/tidar/.venv/bin/python \
  scripts/smoediffusion_eval.py \
  --ckpt /data/checkpoints/smoediffusion_128k_64node-hf/iter_0012000 \
  --dataset aime26 --no-thinking --t-ar 0.05 --K 16 --n 16 --n-samples 1 \
  --max-tokens 1024 --gpu-mem 0.5 --max-num-seqs 8 --log-stats \
  --out /tmp/sf_timing/sanity_tf.json

# SF best-config benchmark (b=1)
VLLM_TIDAR_SINGLE_FORWARD=1 VLLM_TIDAR_SF_TRITON=1 \
VLLM_TIDAR_PROPOSAL_ACC_LEVELS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
VLLM_ATTENTION_BACKEND=FLEX_ATTENTION \
CUDA_VISIBLE_DEVICES=2 /data/home/jinzhao/workspace/tidar/.venv/bin/python \
  scripts/_sf_mmlu_sweep.py \
  --ckpt /data/checkpoints/smoediffusion_128k_64node-hf/iter_0012000 \
  --dataset aime25 --thinking off --batch 1 --K 16 --n 16 \
  --max-tokens 10000 --max-model-len 12288 --gpu-mem 0.8 \
  --explicit-captures --tag sf_p17_b1

# No-bonus variant (branch jinzhao/tidar_sf_kmask_no_bonus)
VLLM_TIDAR_NO_BONUS=1 VLLM_TIDAR_SINGLE_FORWARD=1 VLLM_TIDAR_SF_TRITON=1 \
VLLM_TIDAR_PROPOSAL_ACC_LEVELS=0,5,10 VLLM_ATTENTION_BACKEND=FLEX_ATTENTION \
... # rest as above
```
