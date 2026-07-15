# OPD step_32 vs base smoediffusion — SF acceptance comparison

TiDAR Single-Forward (SF) acceptance-rate comparison between the base
smoediffusion checkpoint and an OPD-finetuned snapshot, K+1 layout.

**Short summary:** there *is* a gap at low temperature (T_AR=0.05) on
AIME26 8k thinking-on (5.14 → 4.52, ≈12% drop in mean); it **collapses at
T_AR=1.0 8k** (3.31 vs 3.34, essentially identical) but **re-opens at
T_AR=1.0 40k** in the *other* direction (base 3.69 vs OPD 3.29, ≈12%
favoring base). At T_AR=0.05 the thinking-on gap also **shrinks with
length** — AIME26 24k base 5.46 vs OPD 5.37 (≈1.6%); on hmmt the sign
**flips** and the OPD advantage *grows* with decode budget: 4k +3.8%, 24k
+9.7%, 40k +11.4% (base 7.48, OPD 8.33). The cleanest and largest
degradations show up in **thinking-off**: hmmt 4k off base 9.40 vs OPD
5.50 (-41%); AIME26 4k off base 6.88 vs OPD 5.39 (-22%) with EOS counts
7/8 vs 1/8 — OPD effectively ignores `enable_thinking=False`. The
behavioral "OPD never EOSes" claim only holds at T_AR=0.05 on
thinking-on long contexts — at T_AR=1.0 mt=40k OPD EOSes 5/8 prompts
like base.

## Checkpoints

| label | path |
|-------|------|
| base  | `/data/checkpoints/smoediffusion_128k_64node-hf/iter_0012000` |
| OPD   | `/data/groups/rl/jinzhao/workspace/ckpts/OPD-fsdp2-async-TEACHER-sparse-kl-NLOGPROBS-128-muon-R5-T2-NN1-LR2e-5-N_8-MBsz_16-T_1.0-L_61440-iter_0012000-opd-diffusion-60K-math-rsa-step32-hf/` |

OPD = on-policy distillation against the dense teacher. `step_32` is early —
small step count chosen to look at whether OPD is *stable*, not converged.

## Run setup

- SF K+1 layout (default on `jinzhao/tidar_SF`)
- K=16, dense proposal levels P=17 (`VLLM_TIDAR_PROPOSAL_ACC_LEVELS=0,1,2,...,16`)
- `VLLM_TIDAR_SINGLE_FORWARD=1`
- `VLLM_ATTENTION_BACKEND=FLEX_ATTENTION` (mandatory — see [[project_sf_requires_flex_backend]])
- Greedy drafter (`--t-diff 0.0`); two verifier temperatures tested: `--t-ar 0.05` and `--t-ar 1.0`
- n = 8 prompts, batch = 1, eager, K=16
- Initial 4k/8k sweeps: vp-dgx-68 GPUs 6/7 in parallel
- T=1.0 8k reruns: vp-dgx-44 GPUs 3/4 in parallel
- 24k follow-up sweeps: vp-dgx-68 GPUs 6/7
- 40k hmmt + 40k T=1.0 AIME26 sweeps: vp-dgx-44 GPUs 3/4
- `mean` reported is vLLM's `Mean acceptance length`, defined as
  `1 + Σ_{k=1}^{K} P(na≥k)` = **tokens committed per spec step** (the
  always-committed K+1-layout bonus from the previous step + drafts
  accepted this step). So at K=16 the ceiling is 17.0, not 16.0. The
  number reported here is the true-mean across *all* SpecDecoding
  windows in the log (see [[feedback_true_mean_acceptance]]) — never
  a single window.
- `pos0` = first-position acceptance rate P(na≥1), averaged the same way.

## Results

### hmmt (`/data/datasets/zpo/hmmt.parquet`), thinking-on

| ckpt | mt    | mean      | pos0  | windows | EOSed | tok/s |
|------|-------|-----------|-------|---------|-------|-------|
| base | 4096  | 5.52      | 0.722 | 102     | n/a   | —     |
| OPD  | 4096  | 5.73      | 0.678 | 98      | n/a   | —     |
| base | 24000 | 6.63      | 0.753 | 301     | 4/8   | —     |
| OPD  | 24000 | 7.27      | 0.724 | 365     | 0/8   | —     |
| base | 40000 | **7.48**  | 0.769 | 373     | 4/8   | 59.28 |
| OPD  | 40000 | **8.33**  | 0.742 | 505     | 0/8   | 62.92 |

At 4k they're roughly tied. At 24k and 40k OPD's mean exceeds base, and the
advantage *grows* with length (+3.8%, +9.7%, +11.4%) even though OPD's pos0
stays consistently lower. OPD's pos0 deficit + higher conditional acceptance
is the dominant pattern on hmmt long-context (see Finding 6).

### hmmt thinking-off, mt=4096

| ckpt | mean | pos0  | windows | EOSed |
|------|------|-------|---------|-------|
| base | **9.40** | 0.832 | 22  | many |
| OPD  | 5.50 | 0.685 | 95  | 0/8  |

Base's mean = 9.40 at 22 windows reflects very short decode (it EOSes
quickly when told no-thinking). OPD doesn't honor the flag.

### AIME26, thinking-on, T_AR=0.05

| ckpt | mt    | mean | pos0  | windows | EOSed |
|------|-------|------|-------|---------|-------|
| base | 8192  | 5.14 | 0.717 | 153     | some  |
| OPD  | 8192  | 4.52 | 0.653 | 182     | 0/8   |
| base | 24000 | 5.46 | 0.721 | 352     | 4/8   |
| OPD  | 24000 | 5.37 | 0.673 | 452     | 0/8   |

At 8k the gap is ≈12% (mean) / ≈9% (pos0). At 24k the *mean* gap collapses
to ≈1.6% while *pos0* stays ≈7% lower for OPD — same conditional-acceptance
recovery pattern as hmmt, just less extreme (AIME26 doesn't flip sign).
The 8k snapshot exaggerates the gap.

(An intermediate-window snapshot at the start of the 8k AIME run showed
base 11.25 / OPD 2.64 — that was a 2-window artifact, not the true mean.
The full sweep settled at 5.14 / 4.52.)

### AIME26, thinking-**off**, mt=4096, T_AR=0.05

| ckpt | mean | pos0  | windows | EOSed | elapsed | tok/s |
|------|------|-------|---------|-------|---------|-------|
| base | **6.88** | 0.747 | 19  | 7/8 | 191s | 62.4 |
| OPD  | **5.39** | 0.670 | 59  | 1/8 | 594s | 49.6 |

Mirrors the hmmt thinking-off finding: base mean jumps to 6.88 with
`enable_thinking=False` (vs 5.14 with thinking on), while OPD barely
moves (5.39 with off vs 4.52 with on). 22% mean gap and 10% pos0 gap,
both larger than the thinking-on equivalents. EOS counts (7/8 vs 1/8)
are an even cleaner indicator of OPD ignoring the chat-template flag.

### AIME26, thinking-on, T_AR=1.0

| ckpt | mt    | mean      | pos0  | windows | tok/s (agg) | EOSed |
|------|-------|-----------|-------|---------|-------------|-------|
| base | 8192  | 3.31      | 0.506 | 214     | 30.35       | 1/8   |
| OPD  | 8192  | 3.34      | 0.512 | 212     | 30.65       | 0/8   |
| base | 40000 | **3.69**  | 0.516 | 710     | 29.23       | **5/8** |
| OPD  | 40000 | **3.29**  | 0.510 | 797     | 26.38       | **5/8** |

At 8k the two ckpts are statistically indistinguishable on acceptance and
throughput. At 40k they **diverge in the *opposite* direction from T=0.05**:
base widens to mean 3.69 (+11%), OPD stays flat at 3.29, giving base a
12% mean advantage and ~11% throughput edge. pos0 still ties (~0.51).

Crucially, **at T=1.0 40k OPD EOSes 5/8 prompts** — same as base. The
"OPD never EOSes" finding is a T=0.05 phenomenon only; under enough
sampling stochasticity *and* enough decode budget, OPD does hit stop
tokens at the same rate as base.

## Findings

1. **Low-T acceptance gap is real but narrow.** AIME26 8k T=0.05 thinking-on:
   OPD mean 4.52 vs base 5.14 (≈12% drop), pos0 0.653 vs 0.717. The whole
   acceptance curve sits below base — drafter and verifier disagree more often.

2. **The gap is temperature- *and* length-dependent at T_AR=1.0.**
   - T=1.0 mt=8k: gap collapses (base 3.31, OPD 3.34 — tied; pos0 also tied
     at ~0.51). High-T verifier sampling flattens p_v across both ckpts and
     erases the low-T advantage of the (more confident) base distribution.
   - T=1.0 mt=40k: gap **re-opens favoring base** (3.69 vs 3.29, ≈12%).
     pos0 stays tied (0.516 vs 0.510) — the divergence is on conditional
     acceptance, not first-token agreement. So at T=1.0 base benefits
     more than OPD from extra decode budget, the *inverse* of the
     hmmt-T=0.05 pattern in Finding 3.

3. **The gap is length- and dataset-dependent — and on hmmt it inverts and
   widens.** AIME26 8k thinking-on: mean gap 12%. AIME26 24k: 1.6%. hmmt 4k:
   ≈ tied. hmmt 24k: OPD mean 7.27 > base 6.63 (+9.7%). hmmt 40k: OPD 8.33 >
   base 7.48 (+11.4%). pos0 stays ≈3–9% lower for OPD on every long-context
   row, so the late-decode distribution of `na` converges (or favors OPD)
   even when first-token agreement doesn't — see Finding 6.

4. **OPD EOS is rate-limited but not broken.** At T_AR=0.05 every OPD prompt
   hits `max_tokens` (hmmt 4k/24k, AIME26 8k/24k), while base EOSes a
   meaningful fraction (e.g. AIME26 24k 4/8). At T_AR=1.0 mt=8k base
   1/8 stop, OPD 0/8. **But at T_AR=1.0 mt=40k both ckpts EOS 5/8** —
   given enough sampling stochasticity *and* enough decode budget OPD
   does find stop tokens. So "OPD never EOSes" was an artifact of the
   low-T, short-to-mid context regime, not a structural defect.

5. **`enable_thinking=False` template flag is largely ignored by OPD —
   confirmed on both datasets.**
   - hmmt 4k off: base 9.40 / 22 windows / fast EOS; OPD 5.50 / 95 windows / no EOS.
   - AIME26 4k off: base 6.88 / 19 windows / 7-of-8 EOS in 191s; OPD 5.39 /
     59 windows / 1-of-8 EOS in 594s.

   On both datasets base picks up a large acceptance boost when `thinking=off`
   (drafts are more predictable), while OPD's acceptance barely moves from its
   thinking-on value. Window counts and EOS counts are the most striking signal:
   base finishes its 8 prompts in ~190s and EOSes most of them; OPD grinds for
   ~600s (3× longer) and barely EOSes any. OPD has lost the chat-template-flag
   sensitivity the base ckpt has; output style stays "thinking-like" regardless.
   (Not retested at T=1.0.)

6. **OPD has lower pos0 but recovers (or overtakes) on conditional acceptance**
   at long context.
   - hmmt 24k: pos0 0.724 < base 0.753, mean 7.27 > base 6.63 (+9.7%).
   - hmmt 40k: pos0 0.742 < base 0.769, mean 8.33 > base 7.48 (+11.4%).
   - AIME26 24k: pos0 0.673 < base 0.721, mean 5.37 ≈ base 5.46 (-1.6%).

   So when OPD's first draft *is* accepted, subsequent drafts get accepted at
   a higher conditional rate — its draft distribution is more "all-or-nothing"
   than base's. The 8k AIME26 row is the only case where both metrics dropped
   together for OPD — short-context regimes don't give enough decode budget
   for the conditional-acceptance recovery to dominate.

7. **Greedy drafter + dense P=17 is the high-acceptance regime**; both ckpts
   sit well below an idealized K=16 ceiling. AIME26 specifically is intrinsically
   draft-unfriendly at T=0.05 (base only 5.14), and even more so at T=1.0
   (~3.3 mean for both).

## Reproducing

```bash
cd /data/groups/rl/jinzhao/workspace/tidar/Zvllm-kp1
git checkout jinzhao/tidar_SF

export VLLM_TIDAR_SINGLE_FORWARD=1
export VLLM_TIDAR_PROPOSAL_ACC_LEVELS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
export VLLM_ATTENTION_BACKEND=FLEX_ATTENTION
PYBIN=/data/groups/rl/jinzhao/workspace/tidar/.venv/bin/python

# AIME26 8k thinking-on, T=0.05, base
CUDA_VISIBLE_DEVICES=0 $PYBIN scripts/_sf_mmlu_sweep.py \
  --ckpt /data/checkpoints/smoediffusion_128k_64node-hf/iter_0012000 \
  --dataset aime26 --thinking on \
  --max-tokens 8192 --max-model-len 16384 \
  --batch 1 --K 16 --n 8 --mode tidar --eager \
  --t-ar 0.05 --tag base_aime26_on

# Swap --t-ar 1.0 for the T=1.0 rerun; swap --ckpt to the OPD path for OPD.
# Swap --dataset hmmt --max-tokens 4096 (or 24000) for the hmmt rows.
```

## Caveats

- `step_32` is very early; numbers may be unrepresentative of converged OPD.
- Only K+1 layout tested. K-mask layout (see [[SF_proposal_layouts.md]]) was
  not re-measured on OPD — gap may differ.
- Only pos0 is recorded in the tables. Full per-position curves
  P(na≥k) for k>1 are in the log files (see Logs section).
- T=1.0 was tested only on AIME26 (8k and 40k). hmmt T=1.0 not run.
- Thinking-off was only tested at mt=4096 on both datasets. Longer
  context thinking-off (e.g. 24k/40k) not run; OPD's "no EOS" behavior
  means it would grind to max_tokens for most prompts.

## Logs

Original sweep logs on `vp-dgx-68:/tmp`:
- `hmmt_{base,opd}_{on,off}.log` — 4k hmmt
- `24k_{base,opd}_hmmt_on.log` — 24k hmmt thinking-on
- `aime26_{base,opd}_on.log` — 8k AIME26 T=0.05
- `24k_{base,opd}_aime26_on.log` — 24k AIME26 thinking-on

Follow-up reruns on `vp-dgx-44:/tmp`:
- `t1_{base,opd}_aime26.log` — 8k AIME26 T=1.0 thinking-on
- `t1_40k_{base,opd}_aime26.log` — 40k AIME26 T=1.0 thinking-on
- `40k_{base,opd}_hmmt_on.log` — 40k hmmt T=0.05 thinking-on
- `off4k_{base,opd}_aime26.log` — 4k AIME26 T=0.05 thinking-off

True-mean and pos0 can be reproduced from any log via:
```bash
grep "SpecDecoding metrics:" LOG | \
  sed -E 's/.*Mean acceptance length: ([0-9.]+).*/\1/' | \
  awk '{s+=$1; n++} END {printf "mean=%.4f n=%d\n", s/n, n}'
```
