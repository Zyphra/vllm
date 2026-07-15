# TiDAR Single-Forward Spec-Decode: Proposal-Layout Investigation

Captures every layout scheme tested during the session and the empirical
findings. Each scheme has an ASCII sketch of the verify and proposal
segments so the differences between layouts are visible at a glance.

## About this branch (`jinzhao/tidar_SF`)

This branch is **the K+1 baseline** — the shipping default TiDAR SF layout
(K+1 proposal block with bonus anchor; K+1 verify with re-fed bonus at slot 0).
Built on `kmask 162ea4aa6` ('K+1 layout always, only bonus output zapped' —
the latter is opt-in via `VLLM_TIDAR_NO_BONUS=1`, default unset).

**Default behavior (no env vars set): K+1 baseline → mean 6.66, pos0 0.74, 48.67 tok/s** (AIME25 thinking-off, n=8, mt=2000, dense P=17).

The K-mask investigation (`VLLM_TIDAR_TFNA0`, `VLLM_TIDAR_VERIFY_NO_ANCHOR`)
lives on sibling branches — see the 'Branches' section at the end. This branch
holds the K+1 baseline plus the docs (`handoff.md` and this file).

## Working environment

- Voltage Park cluster, shared NFS at `/data`.
- Worktrees: `Zvllm-sf-fixed` (rishi K-mask base + TFNA0 + VERIFY_NO_ANCHOR);
  `Zvllm-kp1` (kmask 162ea4aa6 K+1 base + NOSLOT0 ablation).
- Python: `/data/groups/rl/jinzhao/workspace/tidar/.venv/bin/python` (torch 2.9.1+cu128).
- Bench: `scripts/_sf_mmlu_sweep.py --dataset aime25 --thinking off --max-tokens 2000`
  `--max-model-len 8192 --batch 1 --K 16 --n 8 --mode tidar --eager`.
- Env vars always set: `VLLM_ATTENTION_BACKEND=FLEX_ATTENTION`,
  `VLLM_TIDAR_SINGLE_FORWARD=1`, `VLLM_TIDAR_PROPOSAL_ACC_LEVELS=0,1,...,16`
  (dense P=17). FLEX backend is mandatory — without it SF silently degrades to mean ~1.

## Notation used in sketches (K=3 for compactness; real measurements at K=16)

Each diagram shows the SF input layout per request at a particular step M.
Positions are RoPE (rotary) positions.

```
key:
   PB   = the token at the prefix's last position. In K+1 commit pattern,
          PB = the just-sampled bonus or recovered from step M-1.
          In K-mask commit pattern, PB = the last accepted draft from step M-1.
   A    = anchor token in verify segment (re-fed from PB at same RoPE).
   D_i  = a real draft token (drafter's prediction from step M-1) being verified.
   M_a  = an anchor MASK token in the proposal block (filler, output discarded).
   M_di = a mask token in the proposal block; drafter predicts the i-th draft
          for the NEXT step at this slot.
   L    = last committed position (= prefix length - 1).
```

Bidirectional attention within a block = 'BIDIR{...}'. Causal attention to
the prefix = always-on (omitted from sketches).

## Schemes

---

### 1. K+1 baseline   (THIS BRANCH'S DEFAULT — `jinzhao/tidar_SF`)

The paper's default layout (TiDAR Figure 2). One verify segment per request,
one proposal block per acc-level (P=17 acc-levels in our experiments).

```
RoPE:           L-1     L      L+1    L+2    L+3      (K=3 for sketch)
                ----   ----   ----   ----   ----
prefix:         ...     PB
                        ^ committed: prev step's bonus/recovered token

verify (K+1):           A      D_1    D_2    D_3
                        ^      ^^^^^^^^^^^^^^^^^^
                        re-fed PB    K drafts under spec-decode rejection check
                        at RoPE L

proposal (K+1, level=na):  BIDIR{ M_a    M_d1   M_d2   M_d3 }
                                  ^      ^^^^^^^^^^^^^^^^^^
                                  anchor K mask positions; drafter
                                  at L   predicts K drafts here
                                  (slot 0 output discarded at extract)
```

- verify_len = K+1. Slot 0 of verify is the anchor; slots 1..K are the K drafts.
- Proposal block is K+1 wide; slot 0 is an anchor MASK at the same RoPE as verify slot 0
  anchor (= L). The proposal mask block extends one position 'into' the bonus position.
- All K+1 proposal masks bidirectionally attend to each other; subsequent K drafts at L+1..L+K
  see the slot-0 anchor mask as a left in-block neighbor (block-INTERIOR conditioning for them).
- Extract drops slot 0 (anchor mask's prediction discarded); keeps slots 1..K as the K drafts
  for the next step's verify.
- Commits na+1 tokens per step: na accepted drafts + 1 recovered (na<K) or 1 all-accept bonus (na=K).

**Result (AIME25 thinking-off, n=8, mt=2000): mean 6.66, pos0 0.74, 48.67 tok/s. Clean output.**

Status: **this branch's default**. Run with no env-var changes from the
required core set above.

---

### 2. K-mask + TFNA0   (`jinzhao/tidar_sf_tfna0` default; `VLLM_TIDAR_TFNA0=1`)

The paper-faithful K-mask proposal layout, matching SBD training Figure 3 Left
(K masks per block, no clean leading token within the block). TFNA0 patch
fixes the na=0 hang of rishi's pushed K-mask design (`origin/rishisinglefwd`)
by routing na=0 to a fresh propose() forward.

```
RoPE:           L-1     L      L+1    L+2    L+3
                ----   ----   ----   ----   ----
prefix:         ...     PB
                        ^ committed: prev step's last ACCEPTED draft
                          (recovered was dropped by trailing-drop in
                           rishi K-mask design)

verify (K+1):           A      D_1    D_2    D_3
                        ^      ^^^^^^^^^^^^^^^^^^
                        re-fed PB     K drafts under spec-decode check
                        at RoPE L     (PB at slot 0 is consumed for context;
                                       its output is also a 'bonus' slot
                                       in rejection sampler but column K
                                       of output_token_ids gets DROPPED
                                       by the trailing-drop)

proposal (K-mask, level=na):  BIDIR{ M_d1   M_d2   M_d3 }
                                     ^^^^^^^^^^^^^^^^^^
                                     K masks ONLY at L+1..L+K
                                     (no anchor mask)
                                     drafter's slot 0 mask at L+1 is
                                     BLOCK-START -- no in-block left neighbor.
                                     Extract keeps all K.
```

- verify_len = K+1 (same as K+1 baseline; slot 0 = anchor = re-fed PB).
- Proposal block is K wide; no anchor mask. First mask at L+1 = first draft position.
- On na≥1: trailing-drop drops the trailing recovered. Commits = na tokens. Sequence advances by na (1 less than K+1 layout's na+1).
- On na=0 with TFNA0=1: SF-extract guard bypassed; control falls into `drafter.propose()` which does a fresh TF-style forward to generate K fresh drafts for next step. Commits = 1 (the recovered/bonus from this step). Sequence advances by 1. Costs one extra forward per na=0 step.
- TFNA0 implementation: `_bookkeeping_sync` stashes `_tidar_raw_na = [len(t)-1 for t in valid_sampled_token_ids]` BEFORE the trailing-drop (so na=0 = 0 and na=1 = 1 are distinguishable; without this they collapse to len=1 after drop). Dispatch guard appends `and not _tfna0_fb`.

**Result: mean 5.18, pos0 0.57, 25.17 tok/s. Clean output.**

Status: committed on `jinzhao/tidar_sf_tfna0` (`cc128c539`) and `jinzhao/tidar_single_forward_fixed` (`6c8f9fb0c`). Not on this branch's code.

---

### 3. K-mask + TFNA0 + VERIFY_NO_ANCHOR   (`jinzhao/tidar_sf_tfna0`, `VLLM_TIDAR_VERIFY_NO_ANCHOR=1`)

Same as Scheme 2 (K-mask + TFNA0) but with the verify-segment anchor slot REMOVED.
Tests whether the K+1 anchor convention's verify slot 0 (= re-fed PB) is doing
useful work for the K-mask layout. Per the empirical result: no, it isn't.

```
RoPE:           L-1     L      L+1    L+2    L+3
                ----   ----   ----   ----   ----
prefix:         ...     PB
                        ^ committed: prev step's last ACCEPTED draft
                        verifier's NTP for L+1 comes from this prefix
                        position's output naturally (causal LM forward
                        over the prefix), NO re-feed needed

verify (K):                    D_1    D_2    D_3
                               ^^^^^^^^^^^^^^^^^^
                               verify_len = K
                               no slot 0 anchor

proposal (K-mask, level=na):  BIDIR{ M_d1   M_d2   M_d3 }
                                     (identical to Scheme 2)
```

- verify_len = K (one less than K+1 baseline / TFNA0). `compute_position_offsets` no-anchor convention takes effect (`has_anchor = (verify_len == K+1) = False` → `proposal_offset_base = 0`).
- Proposal block still K wide, drafts at L+1..L+K. Same as Scheme 2.
- This is the most training-faithful inference scheme tested: K-mask proposal (Figure 3 Left) + verify segment containing only the K drafts (no clean leading token anywhere — matches training's pure-mask block structure).

**Result: mean 5.21, pos0 0.548, 27.09 tok/s. Clean output (prompts end with EOS).**

Status: committed on `jinzhao/tidar_sf_tfna0` / `jinzhao/tidar_single_forward_fixed` via `VLLM_TIDAR_VERIFY_NO_ANCHOR=1` env var. Not on this branch.

**Comparison vs Scheme 2 (TFNA0 only):** mean 5.21 vs 5.18, pos0 0.548 vs 0.57 — essentially identical. **The verify-side anchor is functionally redundant for K-mask proposals.**

---

### 4. NOSLOT0 ablation   (`jinzhao/tidar_sf_noslot0_ablation`, `VLLM_TIDAR_NOSLOT0=1`)

Isolates the contribution of the proposal-side slot-0 anchor MASK in the K+1
layout. Same K draft target positions as K+1 baseline (L+1..L+K) but with the
proposal block shifted +1 so the first kept draft is block-START (no preceding
mask in its own block).

```
RoPE:           L-1     L      L+1    L+2    L+3    L+4
                ----   ----   ----   ----   ----   ----
prefix:         ...     PB
                        ^ K+1 commit pattern: PB = prev bonus/recovered

verify (K+1):           A      D_1    D_2    D_3
                        ^      (same as K+1 baseline)

proposal (K+1, shifted): BIDIR{        M_d1   M_d2   M_d3   M_unused }
                                       ^^^^^^^^^^^^^^^^^^   ^^^^^^^^^
                                       K kept drafts        slot K
                                       (slots 0..K-1)       discarded
                                       at L+1..L+K          at extract
                                       BLOCK-START at L+1
                                       (no in-block left neighbor)
```

- `proposal_offset_base = 2` (was 1 in K+1 baseline). Masks land at `p_j+2..p_j+K+2` = L+1..L+K+1 (instead of L..L+K).
- Extract keeps slots 0..K-1 (was 1..K) → drafts at L+1..L+K (same target positions as K+1 baseline).
- The slot-0 anchor mask at RoPE L that K+1 baseline had is GONE. The first kept draft (slot 0 of the shifted proposal block at L+1) is now a block-START mask without an in-block left neighbor.
- Everything else (verify segment, commit pattern, K+1 wide attention block) is identical to K+1 baseline.

**Result: mean 4.74, pos0 0.64, 33.70 tok/s. Clean output.**

Status: committed on `jinzhao/tidar_sf_noslot0_ablation` at `8ca34b875` (built on kmask 162ea4aa6 just like this branch).

**Comparison vs K+1 baseline:** pos0 drops 0.74 → 0.64 = **~0.10 pos0 lost from removing only the proposal slot-0 anchor mask** (everything else held fixed). This is the proposal-side anchor mask's isolated contribution.

---

### 5. NO_BONUS   (THIS BRANCH, native `VLLM_TIDAR_NO_BONUS=1`)

A pre-existing toggle in the kmask 162ea4aa6 branch (= this branch's base).
Hybrid: K-mask proposal + K+1 verify with anchor + zap the all-accept bonus
output column. Different combination from any other scheme; documented for
completeness.

```
RoPE:           L-1     L      L+1    L+2    L+3
                ----   ----   ----   ----   ----
prefix:         ...     PB
                        ^ depending on case: bonus or recovered

verify (K+1):           A      D_1    D_2    D_3
                        ^      ^^^^^^^^^^^^^^^^^^
                        re-fed PB     K drafts; bonus column
                                      of output_token_ids zapped
                                      to PLACEHOLDER on all-accept

proposal (K-mask):     BIDIR{ M_d1   M_d2   M_d3 }
                              ^^^^^^^^^^^^^^^^^^
                              K masks (paper-aligned)
                              same as Scheme 2/3
```

- `verify_len = K+1` (kept).
- `proposal_seg_len = K` (K-mask).
- Bonus column of `output_token_ids` zapped to `PLACEHOLDER_TOKEN_ID` on all-accept; the verify slot 0 anchor input remains the re-fed PB.

**Result: mean 5.32, pos0 0.565, 39.22 tok/s. Clean output.**

Status: native toggle on this branch (= the kmask 162ea4aa6 default behavior under `VLLM_TIDAR_NO_BONUS=1`).

---

## Failed / degenerate experiments (NOT in this branch; reverted)

These were tried, debugged, and reverted because they either crashed pos0
to ~0.15 (broken alignment) or produced degenerate output despite metrics
looking inflated. Documented so future investigators don't redo them.

---

### 6. KMASK_SHIFT (broken)

Idea: shift the K-mask proposal masks forward by 1, so masks land at L+2..L+K+1
instead of L+1..L+K. Tested via `VLLM_TIDAR_KMASK_SHIFT=1` (reverted).

```
RoPE:           L-1     L      L+1    L+2    L+3    L+4
                ----   ----   ----   ----   ----   ----
prefix:         ...     PB

verify (K+1):           A      D_1    D_2    D_3
                        (unchanged from Scheme 2)

proposal (K-mask, SHIFTED):    BIDIR{ ?      M_d1   M_d2   M_d3 }
                                            ^^^^^^^^^^^^^^^^^^
                                            drafts now predicted for
                                            L+2..L+K+1, but the verify
                                            segment expects drafts for
                                            L+1..L+K -- one position OFF
```

- `proposal_offset_base = 2` in the K-mask code path.
- Drafter predicts for shifted positions; next-step verify checks at the original positions. Token-RoPE mismatch.

**Result: mean ~1.8, pos0 ~0.15. Broken.**

---

### 7. KMASK_VERIFY_SHIFT (broken — KV-cache RoPE conflict)

Idea: shift the K-mask verify segment back by 1 (offsets [0..K] → [-1..K-1])
so slot 0 lands at RoPE L-1, matching the natural position of the last accepted
draft. Tested via `VLLM_TIDAR_KMASK_VERIFY_SHIFT=1` (reverted).

```
RoPE:           L-1     L      L+1    L+2    L+3
                ----   ----   ----   ----   ----
prefix:         ...    [PB]
                        ^ prefix HAS a stored KV at L from prev step

verify (K+1, SHIFTED):  [A=draft_1] [draft_2]  ...
                        ^^^^^^^^^^^^
                        verify slot 0 at RoPE L-1
                        verify slot 1 at RoPE L
                        ^
                        These verify-slot KV writes OVERWRITE the prefix's
                        stored L-1 and L KVs with the current step's
                        recomputed K/V (different attention context, possibly
                        different input tokens). KV cache corrupted.
```

- Verify slot 0 RoPE = L-1, but the prefix already has a KV entry at L-1 (computed in a previous step). The verify slot 0's KV write at the same cache slot overwrites the prefix's L-1 KV.
- Even worse for slot 1 at L: prefix's bonus at L has stored KV; verify slot 1's input (K-mask draft_1, a DIFFERENT token) overwrites that KV with the K/V of a different token at the same RoPE.
- Subsequent attention reads see corrupted history. Model goes off-distribution.

**Result: mean ~1.77, pos0 ~0.15. Broken.**

---

### 8. KMASK_ANCHOR (worse than baseline)

Idea: in the kmask 162ea4aa6 worktree, set proposal `offset_base = 0` (instead
of 1), so K+1 proposal masks land at L-1..L+K-1. The slot-0 anchor mask would
be at L-1 (the position right before the just-committed bonus), drafts at
L..L+K-1. Tested via `VLLM_TIDAR_KMASK_ANCHOR=1` (reverted).

```
RoPE:           L-1     L      L+1    L+2    L+3
                ----   ----   ----   ----   ----
prefix:         ...     PB

verify (K+1):           A      D_1    D_2    D_3
                        (unchanged)

proposal (K+1, offset_base=0): BIDIR{ M_a    M_d1   M_d2   M_d3 }
                                      ^      ^^^^^^^^^^^^^^^^^^
                                      anchor drafts at L..L+K-1
                                      mask at  -- but verify expects drafts
                                      L-1     at L+1..L+K -- misaligned
```

- Anchor mask at RoPE L-1 lands on a real-token prefix position (where last_acc lives) — OOD for the drafter.
- Drafts at L..L+K-1 don't match next-step's verify positions (L+1..L+K). Off-by-one.

**Result: mean 3.61, pos0 0.48. Worse than TFNA0 (Scheme 2).**

---

### 9. LAST_ACC_ANCHOR (degenerate output — inflated metric)

Idea: implement the 'last accepted token as verify anchor' scheme by THREE coordinated
changes: (a) shift verify positions back by 1, (b) override verify_ids[:, 0] to the
second-to-last committed token (= last_acc), (c) shift K-mask proposal offset_base
from 1 to 2 so next-step alignment is preserved. Tested via `VLLM_TIDAR_LAST_ACC_ANCHOR=1` (reverted).

```
RoPE:           L-1     L      L+1    L+2    L+3
                ----   ----   ----   ----   ----
prefix:         ...   [last_acc]@L-1, [PB]@L
                        ^         ^
                        both stored in prefix's KV cache from earlier step

verify (K+1, SHIFTED): [A=last_acc] [draft_1] [draft_2] [draft_3]
                        ^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        verify slot 0   verify slots 1..K at RoPE L..L+K-1
                        at RoPE L-1     -- they overwrite prefix's L KV
                        same token as   with K-mask draft K/Vs (DIFFERENT
                        prefix's L-1    tokens at same RoPE!)
                        but recomputed
                        K/V differs
                        from prefix's
                        stored K/V

proposal (K-mask, offset_base=2): BIDIR{ M_d1 M_d2 M_d3 } at L+1..L+K
                                  (shifted to keep next-step alignment)
```

- Two simultaneous KV-cache write conflicts (slot 0 at L-1 with recomputed-but-same-token K/V; slot 1 at L with DIFFERENT-token K/V).
- Result: spec-decode metric goes through the roof (mean 14.77, pos0 0.927, **agg 104.67 tok/s**) — but EVERY prompt hits max_tokens with degenerate output:
```
'Theatic\n the for solve solve solve solve solve...'  (2000 tokens of 'solve' repeated)
```
- The model entered a token-loop attractor. Both drafter and verifier trivially predict the looping token, inflating the spec-decode metric. Per-position curve was uniformly 0.93 for 14 positions then a cliff at 0.80 — characteristic 'degenerate co-prediction' signature, NOT real spec-decode geometric decay.
- **Lesson**: any verify-segment slot whose RoPE overlaps the prefix's existing KV must re-feed the EXACT same token (= what K+1 baseline does correctly), or route to a separate KV cache slot. Putting a different token at an overlapping RoPE corrupts the prefix's stored attention state.

**Result: meaningless metric, useless model output.**

---

## Empirical comparison table (AIME25 thinking-off, n=8, mt=2000, dense P=17, eager, FLEX)

| # | scheme | proposal | verify slot 0 | mean | pos0 | pos15 | tok/s | output |
|---|---|---|---|---|---|---|---|---|
| 1 | **K+1 baseline (THIS BRANCH)** | K+1 mask block (anchor at L + K drafts at L+1..L+K) | bonus token re-fed | **6.66** | **0.74** | 0.148 | **48.67** | clean |
| 2 | TFNA0 K-mask | K masks at L+1..L+K (no anchor mask) | last_acc re-fed | 5.18 | 0.57 | 0.000 | 25.17 | clean |
| 3 | VERIFY_NO_ANCHOR (TFNA0 + new toggle) | K masks at L+1..L+K | none (verify_len=K) | 5.21 | 0.548 | 0.000 | 27.09 | clean |
| 4 | NOSLOT0 | K+1 block shifted (no slot-0 anchor mask) | bonus re-fed | 4.74 | 0.64 | 0.084 | 33.70 | clean |
| 5 | NO_BONUS (this branch toggle) | K masks | bonus, output zapped | 5.32 | 0.565 | 0.143 | 39.22 | clean |
| 6 | KMASK_SHIFT | K masks SHIFTED by +1 (broken alignment) | last_acc | ~1.8 | ~0.15 | 0 | ~10 | dead |
| 7 | KMASK_VERIFY_SHIFT | K masks (paper-aligned) | last_acc at RoPE L-1 (KV-cache conflict) | ~1.8 | ~0.15 | 0 | 8.5 | dead |
| 8 | KMASK_ANCHOR | K+1 masks at L-1..L+K-1 (offset_base=0) | bonus re-fed | 3.61 | 0.48 | 0.05 | 24.8 | clean but worse |
| 9 | LAST_ACC_ANCHOR | K masks at L+1..L+K (shifted via offset_base=2) | last_acc at shifted L-1 + verify shifts back by 1 | 14.77 | 0.927 | 0.00 | **104.67** | **degenerate** (token loops) |

## Key findings

1. **K+1 outperforms training-faithful K-mask by ~0.17 pos0 / ~1.45 mean** at matched scale, despite K-mask matching the SBD training block structure (Figure 3 Left) exactly and K+1 deviating (K+1-wide block with a clean leading token, which training never used).

2. **The verify-side anchor is functionally redundant for K-mask** (Scheme 3 vs Scheme 2: 5.21 vs 5.18, pos0 0.548 vs 0.57 — within noise). The K+1 layout's slot 0 = re-fed bonus is doing essentially no useful work for the K-mask proposal layout. Verifier's NTP for L+1 comes from the prefix's last-position output naturally; re-feeding adds nothing.

3. **The proposal-side slot-0 anchor MASK accounts for ~0.10 of K+1's pos0 advantage** (Scheme 4 isolates this: K+1 → NOSLOT0 = 0.74 → 0.64 with the same K draft target positions and the same verify segment).

4. **Schemes that overlap verify-segment RoPE with prefix-stored KV positions are dangerous** (Schemes 7, 9): any token at the overlapping RoPE that differs from what the prefix has stored (different K/V vectors) silently corrupts the KV cache. The model enters degenerate attractors that LOOK spectacular by spec-decode metrics (pos0 0.93+) but produce token-loop garbage. K+1 baseline avoids this by re-feeding the IDENTICAL bonus token at the overlapping RoPE L — the recomputed K/V is consistent enough that the model handles the redundancy.

5. **commit-the-recovered matters for raw throughput** (separate from per-draft acceptance). K+1 baseline commits na+1 per step (drafts + recovered/bonus); K-mask design drops the trailing → commits na per step for na≥1. That's 1 extra committed token per na≥1 step. Combined with the propose() fallback cost on na=0 steps, this is part of why K-mask's tok/s lags K+1 by more than the mean ratio suggests.

## Open question: WHY does K+1 outperform K-mask given training

Per the TiDAR paper Figure 2/3, training is ONLY the SBD pattern: K masks per block, bidirectional within block + block-causal to AR prefix. The model is NOT trained on TF-mode `[next_token, mask×K]` as a separate objective (confirmed).

Pure first-principles: K-mask SF (matches Figure 3 Left exactly) should outperform K+1 SF (K+1 block with clean leading token = OOD vs training). Empirically the opposite holds, by ~0.17 pos0.

Candidates for explanation (none verified):

- **Inference-time generalization**: the model handles a K+1-wide block with a clean leading token gracefully despite training never having one. The extra real-token context inside the block helps the K subsequent masks' bidirectional predictions, more than the OOD block-size deviation hurts.

- **Subtle attention-mask difference**: the SF K-mask code's attention pattern (`compute_attention_mask` for the proposal segment) may differ from training's `sbd_mask_mod` in a non-trivial way — for example, the proposal block's bidirectional pattern might not match training, or the block-causal mask to the AR prefix might cut off at a different position. Untested. Worth dumping both masks and diffing.

- **Block-boundary RoPE alignment**: training's DIFF blocks start at multiples of K (block boundaries at RoPE 0, K, 2K, ...). Inference K-mask blocks start at L+1 where L is dynamically dependent on the step's na sequence — typically NOT aligned to multiples of K. The K+1 block extends one position earlier (to L) and might have a slightly different alignment with the RoPE-modulo-K phase the model implicitly learned. Speculative.

Not resolved in this session.

## Suggested next experiments

1. **K-mask + commit-the-recovered**: flip the trailing-drop in `_bookkeeping_sync` to KEEP the recovered for na≥1 (in K-mask). Should lift commits from na to na+1 per na≥1 step. Cheap (~5 line patch). Tests how much of the K-mask throughput gap is the dropped-recovered vs the per-draft acceptance gap.

2. **Audit SF K-mask attention mask against training's `sbd_mask_mod`**: dump both masks at matched positions and diff. If they differ structurally, that's likely a major contributor to the K-mask underperformance puzzle.

3. **K-mask + add a slot-0 anchor MASK with no clean leading token**: extend proposal_seg_len to K+1 by adding a MASK (not a clean token) at L+0 = block-start. Tests whether K+1's win comes from 'more bidirectional masks in block' or specifically 'a clean leading token in block'. Implementation: in K-mask code, set proposal_seg_len to K+1 and offset_base to 0; extract slots 1..K. The slot-0 mask at RoPE L would overlap with the prefix's L position — must be careful about KV-cache to avoid Scheme 7/9-style corruption (probably need to route slot 0 to a scratch KV slot).

## Branches

| branch | head | base | default behavior | env-gated toggles available |
|---|---|---|---|---|
| **`jinzhao/tidar_SF`** (THIS BRANCH) | `6b91beb4b` | kmask 162ea4aa6 | **K+1 baseline (Scheme 1)** | `VLLM_TIDAR_NO_BONUS=1` → Scheme 5 |
| `jinzhao/tidar_sf_tfna0` | `cc128c539` | rishi 00928a2c3 (K-mask) | K-mask + TFNA0 default-on (Scheme 2) | `VLLM_TIDAR_VERIFY_NO_ANCHOR=1` → Scheme 3 |
| `jinzhao/tidar_single_forward_fixed` | `6c8f9fb0c` | rishi 00928a2c3 (K-mask) | Same as tfna0 + has `handoff.md` edits | Same as tfna0 |
| `jinzhao/tidar_sf_noslot0_ablation` | `8ca34b875` | kmask 162ea4aa6 | K+1 baseline | `VLLM_TIDAR_NOSLOT0=1` → Scheme 4 |

## Reproducing the main runs

### From this branch (K+1 baseline default):

```bash
git checkout jinzhao/tidar_SF
cd /data/groups/rl/jinzhao/workspace/tidar/Zvllm-kp1     # or fresh checkout of this branch
PY=/data/groups/rl/jinzhao/workspace/tidar/.venv/bin/python
CKPT=/data/checkpoints/smoediffusion_128k_64node-hf/iter_0012000
LEVELS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
COMMON='--ckpt $CKPT --dataset aime25 --thinking off --max-tokens 2000 \
        --max-model-len 8192 --batch 1 --K 16 --n 8 --mode tidar --eager'

# Scheme 1: K+1 baseline (DEFAULT)
CUDA_VISIBLE_DEVICES=6 \
  VLLM_ATTENTION_BACKEND=FLEX_ATTENTION \
  VLLM_TIDAR_SINGLE_FORWARD=1 \
  VLLM_TIDAR_PROPOSAL_ACC_LEVELS=$LEVELS \
  $PY scripts/_sf_mmlu_sweep.py $COMMON --tag kp1_baseline

# Scheme 5: NO_BONUS (hybrid K-mask proposal + bonus output zapped)
CUDA_VISIBLE_DEVICES=6 \
  VLLM_ATTENTION_BACKEND=FLEX_ATTENTION \
  VLLM_TIDAR_SINGLE_FORWARD=1 \
  VLLM_TIDAR_NO_BONUS=1 \
  VLLM_TIDAR_PROPOSAL_ACC_LEVELS=$LEVELS \
  $PY scripts/_sf_mmlu_sweep.py $COMMON --tag no_bonus
```

### From `jinzhao/tidar_sf_tfna0` (or `tidar_single_forward_fixed`):

```bash
git checkout jinzhao/tidar_sf_tfna0

# Scheme 2: TFNA0 K-mask (default-on; just standard SF env)
... VLLM_TIDAR_SINGLE_FORWARD=1 VLLM_TIDAR_TFNA0=1 ...

# Scheme 3: TFNA0 + VERIFY_NO_ANCHOR
... VLLM_TIDAR_SINGLE_FORWARD=1 VLLM_TIDAR_TFNA0=1 VLLM_TIDAR_VERIFY_NO_ANCHOR=1 ...
```

### From `jinzhao/tidar_sf_noslot0_ablation`:

```bash
git checkout jinzhao/tidar_sf_noslot0_ablation

# Scheme 4: NOSLOT0
... VLLM_TIDAR_SINGLE_FORWARD=1 VLLM_TIDAR_NOSLOT0=1 ...
```

All raw logs are at `/tmp/aimeoff_*.log` on node 4 (147.68.0.4) and node 68 (`vp-dgx-68`).
