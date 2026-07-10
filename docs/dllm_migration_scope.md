# Scope: TiDAR → vLLM V2 runner / v0.24.0 dLLM path (AMD)

Goal: get **async-scheduled TiDAR** (the host-overhead ~1.5× win) and, longer term,
align with vLLM v0.24.0's diffusion-LLM (dLLM) decode path. This scopes the work,
the two viable paths, risks, and a recommendation.

## Verified starting point (fork state, 2026-07-01)

- **Base:** Zyphra-internal `v0.16.0` + 144 commits (HEAD `bf8319f50`). Not upstream
  vLLM 0.16; a Zyphra fork lineage. Target dLLM machinery is upstream **v0.24.0**.
- **Two runners coexist:** old monolithic `vllm/v1/worker/gpu_model_runner.py`
  (default) and a **V2 gpu-runner** at `vllm/v1/worker/gpu/model_runner.py`, selected
  by `VLLM_USE_V2_MODEL_RUNNER` (`gpu_worker.py:120,270`).
- **The fork's V2 runner is fairly complete:** `gpu/` has `model_runner.py`,
  `spec_decode/` (eagle.py, eagle_cudagraph.py, rejection_sample.py), `sample/`,
  `metrics/`, `async_utils.py` (`AsyncOutput`), `cudagraph_utils.py`, `states.py`.
- **TiDAR is NOT on the V2 runner.** It lives only in `vllm/v1/spec_decode/tidar.py`
  (`TiDARProposer(EagleProposer)`, SF default + TF via `VLLM_TIDAR_TWO_FORWARD=1`),
  used by the **old** runner. `grep -rl tidar vllm/v1/worker/gpu/` → empty.
- **The v0.24 dLLM machinery is absent:** no `gpu/model_states/` (the `ModelState`
  abstraction), no `config/diffusion.py`, no per-seq-causal `triton_unified_attention`.
- Mature + orthogonal: SMoE model (`models/smoe.py`, CCA conv-attn + FusedMoE),
  EP/EPLB (DeepEP all2all), AITER-FA backend with the TiDAR `causal` field (the mask
  fix, `bf8319f50`). These are the deltas we carry regardless of path.

## What the migration actually buys

TiDAR TF already works on the fork today (old runner, AITER-FA PIECEWISE, **177 tok/s**).
The incremental value of moving forward:

1. **Async scheduling — the real prize.** TiDAR TF on MI300X is host-overhead-bound
   (the rejection-sampler `.cpu()` per-step barrier; ~1.5× on the table, "TiDAR≫AR"
   enabler). The async infra lives in the **V2 runner** (`gpu/async_utils.py`); the old
   runner (where TiDAR runs) is synchronous.
2. **Portable per-seq-causal attention** (v0.24 `triton_unified_attention`, now
   OOB-fixed) → retire the hand-rolled AITER-FA causal path. Marginal (AITER-FA works).
3. **FULL cudagraph** via the V2 path (PIECEWISE=177 today; FULL was the open struggle).
4. **Upstream alignment** with the dLLM/DiffusionGemma family + the `ModelState` API.

Prize #1 dominates. Crucially, **#1 does not require the v0.24 dLLM path** — it
requires TiDAR on *a* V2 runner with async, which the fork already has.

## Path A — port TiDAR onto the fork's existing V2 runner (+ async)

Add TiDAR to the fork's V2 `gpu/spec_decode/` (alongside `eagle.py`), wire it into
`gpu/model_runner.py`, run with `VLLM_USE_V2_MODEL_RUNNER=1` + async scheduling. Keep
AITER-FA (causal fix) as the attention.

Steps: (a) confirm SMoE-AR runs on the V2 runner at all (cheap flag flip — see below);
(b) port `TiDARProposer` → `gpu/spec_decode/tidar.py` mirroring `eagle.py`/`eagle_cudagraph.py`;
(c) wire the 2-forward (draft bidirectional / verify causal) + the CCA stash buffers +
mamba cache into the V2 runner's forward + attn metadata; (d) enable async spec output;
(e) validate accept + tput vs the old-runner 177 baseline.

- **Buys:** async scheduling (prize #1), the V2 runner + FULL cudagraph infra. No version rebase.
- **Skips:** the v0.24 `ModelState`/diffusion abstraction + portable attention (stays AITER-FA).
- **Effort:** ~1–2 weeks. **Risks:** V2 runner may not yet support our mamba/CCA hybrid
  + EP + AITER (unknown — de-risk with the flag flip first); async + spec + SSM-cache
  interaction is intricate; the fork's V2 runner is v0.16-vintage (not identical to v0.24's).

## Path B — full v0.24.0 dLLM migration

Rebase the fork's runner to v0.24.0 (or backport `gpu/model_states/`, `config/diffusion.py`,
the per-seq-causal `triton_unified_attention` + the OOB fix, and the `gpu/model_runner.py`
`ModelState` hooks: `num_new_sampled_tokens_per_step`, sampler-in-`load_model`,
`custom_sampler`), then write a **`TiDARSMoEModelState(MambaHybridModelState)`**:
`num_new_sampled_tokens_per_step=1`, self-draft (copy K drafts → `draft_tokens`, no
speculator), `custom_sampler`=rejection accept, `prepare_attn`=**both** the SSM/conv
cache setup (from MambaHybrid) **and** per-seq causal (verify vs draft) — the integration
crux, `add/remove_request`=K+1 block state.

- **Buys:** everything (async, portable attention, ModelState abstraction, upstream alignment).
- **Effort:** several weeks → ~2 months. **Risks:** v0.16→v0.24 API drift across the
  runner + re-port of SMoE/CCA/EP/AITER; the mamba-hybrid **and** spec `ModelState`
  composition is novel (no upstream precedent for spec-decode on a hybrid model via the
  dLLM path); AMD validation of the whole V2 path (the attention half is already fixed +
  validated here).

## Recommended sequence

1. **De-risk (hours):** run **SMoE AR** (no TiDAR) on the fork's V2 runner
   (`VLLM_USE_V2_MODEL_RUNNER=1`). Does our mamba/CCA + EP + AITER model even load/run on
   V2 today? This single flag flip determines whether Path A is "port TiDAR to V2" (if
   SMoE-AR works) or "make the V2 runner support SMoE first" (much bigger).
2. **Path A** if the model runs on V2 — captures the async host-overhead prize (~1.5×)
   without a version rebase. This is the high-value / moderate-cost move.
3. **Path B** as a later, larger alignment effort — best folded into the fork's normal
   upstream-sync cadence (merge toward v0.24), not run as a special project. The attention
   blocker for it is already removed (per-seq-causal OOB fixed + validated on MI300X).

**Do not** treat "migrate to the dLLM path" as the immediate goal: the immediate goal is
async-scheduled TiDAR, and the fork's existing V2 runner is the shorter road to it.

## Gate result (2026-07-01) — the cheap Path A is dead; paths converge

Ran SMoE **AR** (no TiDAR) on both runners on cnode-5:
- **Old runner:** loads + generates coherently (baseline OK).
- **V2 runner (`VLLM_USE_V2_MODEL_RUNNER=1`):** **`AssertionError` at
  `gpu/attn_utils.py:88 _reshape_kv_cache` → `assert isinstance(kv_cache_spec,
  AttentionSpec)`.** Our SMoE is a mamba/CCA hybrid; its mamba layers produce a
  non-`AttentionSpec` KV spec, and the fork's V2 `_reshape_kv_cache` only handles
  `AttentionSpec`. **The fork's V2 runner cannot run our hybrid model even in plain AR.**

**Implication:** hybrid-model support in the V2 runner *is* what v0.24.0 added via the
`model_states/` refactor (`MambaHybridModelState` + hybrid KV-cache reshaping). So
"get SMoE on a V2 runner" ≈ adopting the v0.24 model_states core = **Path B territory**.
There is no cheap Path A: [port TiDAR to V2] is gated on [backport mamba-hybrid V2
support], which is most of Path B's foundation. Async+spec is also v0.24-era (the
`async_scheduler` `num_sampled_tokens_per_step` / "update draft token ids in the worker"
changes), so backporting async to the *old* runner (option C) is likely not cheaper
either — unconfirmed.

## M1 update (2026-07-01) — SMoE-AR now runs on V2

M1 landed in the remote build repo
`/shared/home/jinzhao/workspace/tidar/vllm-smoe-amd` by extending
`vllm/v1/worker/gpu/attn_utils.py` rather than doing the full v0.24
`model_states/` backport yet:

- Full traceback captured: V2 failed at `_reshape_kv_cache` because it asserted every
  KV group was `AttentionSpec`; SMoE's CCA groups are `MambaSpec`.
- Patch: add `MambaSpec` state-tensor reshape to V2 `_reshape_kv_cache`, then apply the
  old-runner hybrid attention/Mamba KV layout stride fix when attention and Mamba caches
  coexist.
- Validation: old runner and V2 runner, same prompt/greedy/max_tokens=16,
  `ROCM_AITER_FA`, async disabled, matched exactly:
  `[108, 10354, 55733, 3245, 236787, 107, 236772, 34241, 600, 236743, 236778, 236862, 236778, 236784, 236812, 236761]`.
  Text for both: `"\n\n### Goal State:\n- Prove that 2+2=4."`.
- Logs:
  `/shared/home/jinzhao/tidar_m1_v2_assert_full_20260701_165426.log`,
  `/shared/home/jinzhao/tidar_m1_old_vs_v2_ar_20260701_180932.log`.

Next gate is M2: enable/measure async scheduling for SMoE-AR on V2 and verify the
Mamba/CCA state path remains correct under async.

## M2 update (2026-07-02) — async works on V2 SMoE-AR, but AR perf is modest

M2 was tested on `ibm-cnode-123`, GPU 3, with the same SMoE checkpoint and
`VLLM_USE_V2_MODEL_RUNNER=1`, `ROCM_AITER_FA`.

- Async scheduling enables cleanly on SMoE-AR V2. The `mamba_cache_mode` gate does not
  block this plain-AR setup because prefix caching is off and the mode is `"none"`.
- Correctness check: sync and async greedy outputs matched by full token-id hashes on the
  extended matrix.
- Perf check 1: no host-overhead win in the eager AR microbench. Sync was
  `59.9005 tok/s`; async was `59.8910 tok/s`; speedup `0.9998x`.
- Perf check 2: longer eager AR decode (`max_tokens=512`) showed only small wins:
  b=1 `8.3967 -> 8.6431 tok/s` (`1.029x`) and b=8
  `62.0065 -> 64.8567 tok/s` (`1.046x`).
- Perf check 3: after the M2.5 capture fix, captured AR
  (`FULL_AND_PIECEWISE`, AIME thinking-on prompts tokenized with
  `add_special_tokens=False`, warmup excluded, `max_tokens=512`) matched sync/async
  hashes exactly and showed modest wins: b=1 `53.99 -> 54.86 tok/s` (`1.016x`),
  b=8 `343.96 -> 377.41 tok/s` (`1.097x`).
- Initial captured/non-eager check: V2 SMoE started `CUDAGraphMode.FULL_AND_PIECEWISE`
  capture (`[1, 2, 4, 8, 16, 24, 32]`) but failed before generation for both sync and
  async. Root symptom: CCA's Triton path raised
  `torch.AcceleratorError: HIP error: operation not permitted when stream is capturing`
  from `vllm/model_executor/layers/mamba/cca.py:1138` (`forward_triton`, slicing
  `hs_p[start_i:end_i]`).
- Harness note: the first same-process sync→async run failed at async startup because the
  sync LLM had not fully released GPU memory. The corrected harness runs sync and async in
  separate Python processes inside one docker container, preserving JIT cache while freeing
  GPU memory.
- Logs:
  `/shared/home/jinzhao/tidar_m2_v2_ar_async_vs_sync_20260701_183408.log`
  (memory-release harness failure),
  `/shared/home/jinzhao/tidar_m2_v2_ar_async_vs_sync_proc_20260701_185055.log`
  (128-token corrected result),
  `/shared/home/jinzhao/tidar_m2ext_v2_ar_matrix_20260701_192022.log`
  (512-token eager),
  `/shared/home/jinzhao/tidar_m2_captured_ar_sync_async_mt512_20260702_002504.log`
  (512-token captured).

Gate result: async correctness is OK and async does pay on captured plain AR, but the
expected ~1.5x host-overhead win was not confirmed on plain AR. Proceed to M3 only with
calibrated expectations: TiDAR TF must expose a larger host gap than plain AR to beat the
current 177 tok/s production path.

## M2.5 update (2026-07-01) — capture replay is unblocked

Capture-safety patches in the local and remote working trees moved V2 SMoE-AR past the
original HIP stream-capture failure:

- `vllm/model_executor/layers/mamba/cca.py`: graph-capture vectorized prefill path for
  non-spec AR; captured-safe decode path avoiding the Python `causal_conv1d_update`;
  TiDAR/spec stash writes guarded so plain AR can reuse the helper.
- `vllm/distributed/parallel_state.py`: graph-capture depth guard exposed as
  `is_in_graph_capture_context()`.
- `vllm/v1/worker/gpu/attn_utils.py` and `cudagraph_utils.py`: graph-capture setup asks
  attention metadata builders for capture metadata.

This gate is now **passed for the vectorized CCA capture path**:

- Root cause of the prior b=8 zero/NaN replay was AITER attention metadata built for
  capture with the wrong `max_query_len`. `build_attn_metadata()` used
  `query_start_loc_cpu.max()`, which is the total query-token count. For pure b=8
  one-token decode it produced `max_query_len=8`, causing AITER to capture
  `num_decodes=0` / `num_decode_tokens=0`.
- Fix: compute `max_query_len` from per-request adjacent differences in
  `query_start_loc`; b=8 capture now sees `max_query_len=1`, `num_decodes=8`, and
  `num_decode_tokens=8`.
- Validation 1: b=8/max4 captured and eager matched exactly:
  `5779c175883329ea2184a0d5a8ed5943864e9869ef20924f24dc407f4289e73d`.
- Validation 2: b=8/max64 default captured matched compile-with-cudagraph-disabled
  exactly:
  `1aec4f7f7938f1e4afb7d2e287484c3e9ca29bd4bfc8fc61922af10793dbd4ab`.
  Eager differed only from Inductor numerics on two rows late in generation (first
  mismatch row 1, token position 52); compile-no-CG and captured had zero mismatches.
- Validation 3: captured sync/async matched exactly at b=1/b=8, max_tokens=512 in M2.
- Fused CCA still fails capture via `torch.repeat_interleave`:
  `/shared/home/jinzhao/tidar_m2p5_fused_capture_smoke_20260701_200326.log`.
  Use the vectorized capture path for now.
- Relevant logs:
  `/shared/home/jinzhao/tidar_m2p5_trace_b8_20260701_220756.log`,
  `/shared/home/jinzhao/tidar_m2p5_trace_b8_fixmaxq_20260701_221732.log`,
  `/shared/home/jinzhao/tidar_m2p5_eager_b8_fixmaxq_20260701_222602.log`,
  `/shared/home/jinzhao/tidar_m2p5_b8_mt64_sync_async_20260701_224133.log`,
  `/shared/home/jinzhao/tidar_m2p5_b8_mt64_diff_20260701_224604.log`.

Next gate is M3 (TiDAR on V2), but do not carry forward the old 1.5x plain-AR async
assumption: measured captured AR gains were only `1.016x` (b=1) and `1.097x` (b=8).

## M3 update (2026-07-02) — TiDAR sync/eager runs on V2 and matches old runner

M3 used the lower-backport route: add a V2-native `gpu/spec_decode/tidar.py`
self-speculator and wire it into `gpu/spec_decode/__init__.py` before the EAGLE path
(because the current `use_eagle()` helper also returns true for `tidar`).

- The V2 TiDAR speculator reuses the target model for drafting, builds
  `[accepted_token, mask x K]`, uses non-causal/per-pass draft attention metadata, and
  attaches CCA read/write state overrides from V2 block tables.
- V2 model-runner glue commits CCA state from the rejection sampler's accepted-count
  result and copies draft tokens back to the scheduler for the sync path.
- The first V2 smoke ran but diverged from the old runner because the new V2 speculator
  fell back to mask token id `128000`; the old TiDAR proposer falls back to hardcoded
  token id `4`. Aligning the V2 fallback to `4` made the outputs match.
- Validation on `ibm-cnode-123`, GPU 3: old runner and V2 runner, same prompt,
  greedy, `max_tokens=16`, `ROCM_AITER_FA`, sync/eager, both produced token hash
  `a06a38ca906fa474cc4cd1c6e98b30c15679c5534f35cb1de07907d83fd401c1` and identical
  token IDs `[236743, 236812, 108, 7243, 108, 1018, 236770, 236761, 31278, 13768,
  53121, 138, 107, 818, 5498, 1262]`.
- The earlier Triton `arange's range must be a power of 2` failure in
  `vllm/attention/ops/tf_attention.py` is fixed in the working tree by using a
  power-of-two TF attention `block_q` default (`16`).
- Logs:
  `/shared/home/jinzhao/tidar_m3_old_tidar_smoke_short_20260702_032154.log`,
  `/shared/home/jinzhao/tidar_m3_v2_tidar_smoke_short_mask4_20260702_034239.log`.

M3's sync/eager gate is passed. Next gate is M4: allow/validate TiDAR async on V2,
then measure acceptance, coherence, and throughput against the 177 tok/s production path.

## M4 update (2026-07-02) — TiDAR async + cudagraph on V2 beats 177

M4 passed the performance gate after two cudagraph fixes. V2 TiDAR now enables async
scheduling by allowing TiDAR under the V2 async gate and keeping padded drafter batches
enabled only when `VLLM_USE_V2_MODEL_RUNNER=1`; old-runner compatibility is unchanged.

- Correctness smoke: old runner and V2 sync/eager matched exactly on AIME greedy
  `max_tokens=64` (`sha256=a86c7db4ba2b01ceeb79fb2ecf997dab32591c06c15290b23645cb3b38743d41`).
  Warmed V2 sync vs async eager also matched that hash.
- b8/MT512 AIME thinking-on: V2 sync eager `65.951 tok/s`, mean_accept_len `2.359`;
  V2 async eager `75.186 tok/s`, mean_accept_len `2.813` (`+14%`, coherent output).
- Captured b1/64 greedy works after CCA capture fixes: async captured `12.108 tok/s`,
  same greedy hash.
- Default captured b8 initially failed during graph capture because auto capture sizes
  included SF/diffusion-sized totals that built non-uniform dummy rows and fell into
  CCA's host `.tolist()` loop under HIP graph capture.
- Fix 1: TiDAR V2 cudagraph replay now only accepts decode-only (`S=1`) or exact uniform
  verify (`S=K+1`) scheduler rows, capture dummy request count is `tokens/(K+1)` for TF
  verify graph sizes, and `FULL_AND_PIECEWISE` no longer rejects exact TiDAR verify rows
  as generic mixed/prefill batches. This raised b8/MT512 async captured from zero graph
  replays to verify graph replays and `144.917 tok/s`.
- Fix 2: `gpu/spec_decode/tidar.py` now captures exact-size TiDAR drafter graphs using
  persistent input/slot-mapping buffers and falls back to eager for unmatched shapes.
  b8/MT512 async captured with verify+draft graph replay reached **`405.651 tok/s`**
  (`4096` tokens in `10.097s`, mean_accept_len `2.349`, coherent output), beating the
  old-runner production `177 tok/s` by `2.29x`.
- Greedy correctness smoke with the short M4 config matched the earlier passing token
  list exactly; the direct `json.dumps(token_ids)` hash is
  `1aa60bb1012e18127df5d95edab0e6dd5e52386689ca3afbf9b8826c19a19018`.
- Logs:
  `/shared/home/jinzhao/tidar_m4d_v2_tidar_eager_sync_async_b8_mt512_20260702_070126.log`,
  `/shared/home/jinzhao/tidar_m4j_v2_tidar_captured_graphfix_b8_mt512_20260702_081311.log`,
  `/shared/home/jinzhao/tidar_m4l_v2_tidar_draftcg_b8_mt512_20260702_082035.log`,
  `/shared/home/jinzhao/tidar_m4n_v2_tidar_draftcg_greedy64_shortcfg_20260702_082500.log`.

## M4.5 update (2026-07-02) — hardening + perf recheck

Clean-container b3 tests exposed two capture-size policy bugs and one async-state bug,
all fixed in the working tree:

- V2 TF no longer inherits TiDAR SF capture sizes or the SF cap bump; b3 TF now captures
  `[1, 2, 3, 17, 34, 51]`, not SF sizes like `68/136/204`.
- The AR graph grid now appends non-power-of-two `max_num_seqs`, so b3 captures AR size
  `3` instead of trying invalid decode-only size `4`.
- V2 TiDAR latches stable per-request CCA state slots, preventing async scheduling from
  moving a recurrent CCA row between steps.

Validation: b3 sync/eager and async/eager greedy max64 match exactly
(`9753c03b82fe48e399be8937528e970d799e7aadfccb6296286634162ec169be`).
Captured sync and captured async also match each other exactly, but differ from strict
`enforce_eager=True` at prompt 1 token 10; this is consistent with the M2.5
non-eager/Inductor/captured numeric boundary, not async drift.

Perf recheck: b8/MT512 async captured with current hardening reached **`509.515 tok/s`**
(`4096` tokens in `8.039s`, mean_accept_len `2.813`, coherent output), `2.88x` over
the old-runner `177 tok/s`. Log:
`/shared/home/jinzhao/tidar_m45_logs/20260702_184641_b8_async_captured_hardened.log`.

Conclusion: Path B achieved the 177-beating prize. Next useful work is hardening the
remaining compiled/captured parity caveat, broader prompts/seeds/longer MT, and cleaning
the implementation for review.

## M5 update (2026-07-02) — cleanup + broader sweep

Temporary capture-debug hooks were removed from `vllm/v1/worker/gpu/model_runner.py`
and `vllm/v1/worker/gpu/cudagraph_utils.py`; the functional V2 TiDAR state/commit
and graph-selection paths remain. Syntax checks pass in the ROCm container.

After-cleanup b3 greedy parity:

- sync/eager and async/eager match exactly:
  `333e80f17e92657c8e5901b637813fc70666c8ee314edab16b955dc74ef6a1b1`.
- sync/captured and async/captured match exactly:
  `9753c03b82fe48e399be8937528e970d799e7aadfccb6296286634162ec169be`.
- captured vs strict `enforce_eager=True` still differs, consistent with the existing
  Inductor/captured numeric caveat rather than async drift.

Broader production-shaped V2 async-captured sweep on `ibm-cnode-19`, GPU 3, with
`FULL_AND_PIECEWISE`, AITER FA, and TiDAR TF passed b1/b3/b8/b16, two prompt offsets,
two seeds, and MT up to 512. A 2026-07-06 b64 follow-up also passed at MT128:

| Case | Throughput | Mean accept len |
|---|---:|---:|
| b1, MT128, offset 0, seed 0 | `108.635 tok/s` | `4.000` |
| b3, MT128, offset 0, seed 0 | `224.316 tok/s` | `3.048` |
| b8, MT512, offset 0, seed 0 | **`553.839 tok/s`** | `3.103` |
| b8, MT256, offset 8, seed 1 | `438.955 tok/s` | `2.485` |
| b16, MT128, offset 0, seed 0 | `799.495 tok/s` | `2.560` |
| b3, MT256, offset 12, seed 1 | `298.601 tok/s` | `4.000` |
| b64, MT128, offset 0, seed 0 | `570.355 tok/s` | `2.723` |

Main logs:
`/shared/home/jinzhao/tidar_m5_logs/20260702_191356_cnode19_v2_async_captured_sweep.log`,
`/shared/home/jinzhao/tidar_m5_logs/20260702_192418_cnode19_b3_parity_after_cleanup.log`,
`/shared/home/jinzhao/tidar_m5_logs/20260706_181316_cnode19_v2_async_captured_b64_mt128.log`.

This b64 V2 point did **not** beat the older historical AMD TF b64 number
(`862.6 tok/s`) and was temporarily left below the historical result. It is now
superseded by the 2026-07-07 long/cap-fixed b64 run (`2743.088 tok/s`, clean
rerun `2699.349 tok/s`) documented below.
2026-07-06 scaling follow-up: the old b64 V2 point was graph-cap limited. Its log shows
`max_cudagraph_capture_size=510`, while a b64 TiDAR TF verify/draft batch is
`64 * (K + 1) = 1088` tokens, so exact b64 TiDAR graph replay could not happen.
`vllm/config/vllm.py` now bumps the V2 TF TiDAR graph cap to at least
`max_num_seqs * (K + 1)`. The 2026-07-07 run below is that b64 rerun and confirms
capture sizes include `1088`.

2026-07-06 acceptance follow-up: the short V2 probe used target temperature `0.0`
and `tidar_diff_temperature=0.0`, so acceptance is greedy argmax equality, not a
sampling-temperature effect. The probe's older `total_tokens / (batch * propose_calls)`
formula undercounts when requests finish early; `/shared/home/jinzhao/tidar_m5_logs/probe_v2_tidar_accept.py`
now reports active-request `mean_accept_len` from actual V2 `num_sampled` and keeps the
old value as `legacy_mean_accept_len`. The TF paged op now honors
`VLLM_TIDAR_TF_PAGED_NO_SPLITS=1`/`VLLM_TIDAR_FA_NO_SPLITS=1`, and the AITER-FA decode
route is gated on TiDAR exact K+1 and defaults on for TiDAR TF, replacing the older
unconditional `_dmql > 1` custom branch. `VLLM_TIDAR_USE_TF_PAGED_ATTENTION=0` or
`VLLM_TIDAR_DISABLE_TF_PAGED_ATTENTION=1` still opt out for debugging. Corrected b16/MT256 eager diagnostic
on Slurm cnode-26 GPU3, `iter_0012600`, AIME first16, `PATCH_PROBE_REPEATS=2`, confirmed
the helper for both draft (`causal=False`) and verify (`causal=True`). Warmed repeat-2:
default split-K `272.221 tok/s`, corrected `mean_accept_len=5.542`
(`legacy_mean_accept_len=3.821`); no-splits `286.869 tok/s`, corrected
`mean_accept_len=5.613` (`legacy_mean_accept_len=3.765`). This validates the route and
explains the apparent low accept; it does not replace the captured V2 TF throughput table.
Log: `/shared/home/jinzhao/tidar_m5_logs/20260706_214938_cnode26_v2_tfpaged_repeats2_b16_mt256_default_vs_nosplits.log`.

2026-07-07 AMD/NVIDIA parity trace: the remaining AMD low-accept symptom was a missing
TF-paged route in one run tree, not the checkpoint or rejection sampler. Without it on
`ibm-cnode-107` GPU1, b1 eager V2 TF corrected `mean_accept_len=1.000`, draft tokens
collapsed to constant `47599`, and output was incoherent. With
`VLLM_TIDAR_USE_TF_PAGED_ATTENTION=1`, the same AMD probe produced coherent output and
corrected `mean_accept_len=2.913`; the matching NVIDIA H100 trace was `2.870`.
Logs: `/shared/home/jinzhao/tfscope/accept_parity_logs/20260707_054222_amd_cnode107_gpu1_b1_eager_trace.log`,
`/shared/home/jinzhao/tfscope/accept_parity_logs/20260707_055619_amd_cnode107_gpu1_b1_eager_trace_tfpaged.log`,
`/data/home/jinzhao/nv_v2_tidar_logs/accept_parity/20260706_233626_nv_b1_eager_trace_gpu7_nomproc.log`.

2026-07-07 AMD long-config follow-up: ran the NVIDIA apple-style prompt/window/capture
setup on MI300X (`ibm-cnode-107` GPU1): AIME25 thinking-off, `MT=2000`, K=16,
target/draft temp `0.0`, prompt-token IDs with single BOS, ignore-EOS,
`ROCM_AITER_FA`, `FULL_AND_PIECEWISE`, TF-paged AITER attention, and no-splits. The
only mismatch is checkpoint because AMD lacks `iter_0012000` under `/shared`, so the
run used `iter_0012600`. Results: b1 `100.146 tok/s`, accept `5.208`; b8
`482.468`, accept `5.400` (second repeat accept `6.499`); b16 `940.229`, accept
`6.548`; b64 `2743.088`, accept `6.120` from the full sweep, plus a clean b64-only
rerun best `2699.349`, accept `6.552`. The b64 log confirms cap/capture sizes include
`1088 = 64 * (K+1)`, so the old cap-limited `570 tok/s` point is superseded. Logs:
`/shared/home/jinzhao/tfscope/amd_long_v016/20260707_062913_cnode-107_gpu1_v2_tf_fp_long_iter12600_b1_b8_b16_b64_mt2000.log`,
`/shared/home/jinzhao/tfscope/amd_long_v016/20260707_065110_cnode-107_gpu1_v2_tf_fp_long_iter12600_b64_mt2000_rerun.log`.

2026-07-07 same-checkpoint AMD/NVIDIA profile follow-up: reran the long sweep with
`iter_0012600` on both platforms and matched the real NVIDIA config: AIME25
thinking-off, `MT=2000`, warmup 64, K=16, target/draft temp `0.0`, prompt-token IDs,
ignore-EOS, `FULL_AND_PIECEWISE`, `n_sample=10`, `num_prompts=bsz`,
`max_num_seqs=bsz`, and `num_input_sequences=bsz*10`. NVIDIA H100 (`FLASH_ATTN` v3)
measured b1 `148.951 tok/s`, acc `4.978`; b8 `1029.468`, acc `5.590`; b16
`1656.976`, acc `5.735`; b64 `2127.865`, acc `5.881`. AMD MI300X
(`ROCM_AITER_FA`, TF-paged, no-splits, warmed image `tidar-v024-built:iter12600-aiterwarm`)
measured b1 `100.010 tok/s`, acc `5.208`; b8 `510.885`, acc `5.471`; b16
`990.631`, acc `5.986`; b64 `1682.274`, acc `5.886`. This supersedes the earlier
AMD-only `n_sample=1` sweep for parity: AMD acceptance is now roughly matched, but
throughput is lower. Matched b16 profile (`n_sample=1`, `MT=512`) shows why:
NVIDIA full-shape target/draft/reject is `20.067/17.579/0.989 ms`; AMD is
`28.821/26.314/1.175 ms`. The gap is mainly target/draft forward latency on
`ROCM_AITER_FA`, not rejection-sampler logic. Logs:
`/data/home/jinzhao/nv_v2_tidar_logs/iter12600_matched/`,
`/data/home/jinzhao/nv_v2_tidar_logs/iter12600_profile/20260707_014246_nv_gpu6_iter12600_fp_profile_b16_mt512.log`,
`/shared/home/jinzhao/tfscope/amd_iter12600_matched/20260707_075944_*_warmimg.log`,
`/shared/home/jinzhao/tfscope/amd_iter12600_profile/20260707_081033_amd_cnode107_gpu4_iter12600_fp_profile_b16_mt512_warmimg.log`.

2026-07-07 checkpoint-control + AR-baseline follow-up: exact same NVIDIA V2 TF
long script/config, paired by batch on the same H100 GPU, shows `iter_0012000`
has higher TiDAR accept than `iter_0012600`: b1 `188.467 tok/s`, acc `6.339`
vs `157.188`, acc `4.978`; b8 `1139.612`, acc `6.623` vs `984.326`, acc
`5.554`; b16 `1846.769`, acc `6.475` vs `1685.288`, acc `5.628`; b64
`2280.917`, acc `6.556` vs `2098.162`, acc `5.799`. This confirms the older
v0.16 handoff's higher accept is mostly checkpoint/workload, not a V2 rejection
sampler regression. Follow-up tokenizer check: loaded `AutoTokenizer.chat_template`
differs only by the leading BOS (`iter_0012000` prepends `{{ bos_token }}`,
`iter_0012600` does not), but this benchmark path used already-templated
AIME25 `prompt_token_ids` with `add_special_tokens=False`; all 30 prompt token
sequences are identical across checkpoints. Forced-BOS retest on both H100 and
MI300X changed `0/30` prompts and logged `leading_bos_count=1`; `iter_0012600`
accept stayed in the same band (NVIDIA b1/b8/b16/b64 `4.978/5.479/5.642/5.774`,
AMD `5.208/5.622/5.887/5.902`), so missing BOS is not the accept gap here.
Captured AR baselines on `iter_0012600` (`n_sample=1`,
`MT=2000`, V2 async, `FULL_AND_PIECEWISE`, no spec) measured NVIDIA H100
`82.649/552.746/1035.511/3420.864 tok/s` for b1/b8/b16/b64 and AMD MI300X
`53.369/419.094/774.441/2622.038 tok/s`. Logs:
`/data/home/jinzhao/nv_v2_tidar_logs/iter12000_vs_12600_backtoback_20260707_030645/`,
`/data/home/jinzhao/nv_v2_tidar_logs/ar_v2_iter12600_captured_20260707_034419/`,
`/shared/home/jinzhao/tfscope/amd_ar_v2_iter12600_captured_20260707_091143/`.

Conclusion after M5: the implementation is past the perf/correctness gate for this
Path-B slice. Remaining pre-commit work is review cleanup of the broader diff, a decision
on whether to document/accept the strict eager-vs-captured numeric boundary, and EP/EPLB
checks on the V2 runner.

## M6 update (2026-07-03) — landing prep plus DP+EP eager+captured smoke

Review cleanup started after the M5 sweep:

- V2 TiDAR CCA layer lookup is now centralized behind
  `GPUModelRunner._get_tidar_cca_layers()`, removing stale private `_cca_layers_cache`
  use and keeping capture/reset/commit paths on one cache.
- Added short comments for the two non-obvious V2 TiDAR state cases: CCA state must be
  reset after graph capture, and async scheduling needs stable per-request CCA slots
  because scheduler rows can compact between steps.
- Cleaned stale config comments: the old handoff doc name was replaced with
  `docs/TIDAR_AMD_HANDOFF.md`, and the SF capture-size formula now says
  `(P + 1) * (K + 1)`.
- Local checks passed: `git diff --check` and `py_compile` for the touched config/V2
  files.
- ROCm-container checks passed on `ibm-cnode-19` after editable install and again on
  the two-GPU EP smoke container: `py_compile` for the touched TiDAR/V2/config files.

Runtime checks:

- Old-runner guard: the attempted run confirmed in logs that old-runner TiDAR still
  disables async (`Async scheduling not supported ... will be disabled`) and uses the
  monolithic `gpu_model_runner.py`, but generation was interrupted after long AITER JIT
  and model load when SSH to `ibm-head` / `ibm-cnode-*` briefly failed with
  `Permission denied (publickey)`. Cleanup has since completed.
- EP/EPLB: this cannot be meaningfully validated with a single-GPU `LLM(...)` flag flip;
  the current `LLM` entrypoint explicitly rejects `data_parallel_size > 1` for
  single-process offline use, and vLLM EP size is `TP * DP`. A real V2 DP+EP eager
  smoke was run on `ibm-cnode-20`, physical GPUs 1/2, via `vllm serve` with
  `VLLM_USE_V2_MODEL_RUNNER=1`, `--data-parallel-size 2`, `--enable-expert-parallel`,
  `--disable-nccl-for-dp-synchronization`, and AMD-safe
  `--all2all-backend allgather_reducescatter`.
- The first EP server attempt failed before model load because `GLOO_SOCKET_IFNAME=eth0`
  names a nonexistent interface in this container. Retrying with `GLOO_SOCKET_IFNAME=lo`
  booted successfully through the DP coordinator, V2 runner, TiDAR self-speculator,
  AgRs all2all, expert split (`8/16` experts per rank), AITER MoE JIT, KV-cache init,
  and API startup.
- The first `/v1/completions` request initially failed at runtime in CPU/Gloo DP
  padding sync: `op.preamble.length <= op.nbytes. 12 vs 4`, immediately after
  `Using CPU all reduce to synchronize DP padding between ranks.`
- Root cause: V2 main forward coordinated DP padding with the V2 GPU runner's
  `[2, dp]` CPU fold, but TiDAR's nested draft forward passed
  `num_tokens_across_dp=None`, so it fell back to the old shared
  `vllm/v1/worker/dp_utils.py` `[6, dp]` fold. Idle ranks also needed a dummy
  TiDAR draft pass so EP all-to-all counters match when only a peer rank has
  live requests.
- Fix: V2 `gpu/dp_utils.py` now folds a third TiDAR-draft-token row; the V2
  model runner computes/stores the per-step draft fold, passes it into
  `TiDARSpeculator.propose()`, and runs `speculator.dummy_run()` on idle ranks
  when a peer advertised draft work. The V2 speculator accepts the DP vector and
  uses it in `run_model()`.
- Second bug found during the retest: V2 `execute_model()` returns before DP
  collectives when `total_num_scheduled_tokens == 0`, but the DP busy-loop guard
  treated any `execute_model()` call as if it had already joined the model-forward
  collectives. That put one rank in the finish-sync all-reduce while the other
  rank ran an idle dummy model-forward all-reduce. Fix: the guard now skips the
  extra dummy batch only after a non-empty model batch was dispatched.

Validation on `ibm-cnode-20`, physical GPUs 1/2, fresh container
`tidar_m6_dpfix2_codex_022953`, log
`/shared/home/jinzhao/tidar_m6_logs/20260703_022953_cnode20_v2_dp2_ep_eager_dpfix2_server.log`:

- Remote `py_compile` and `git diff --check` passed for the touched V2 files.
- Server booted through DP coordinator, V2 runner, TiDAR self-speculator, AgRs
  all2all, EP expert split (`8/16` per rank), AITER JIT, KV init, and API
  startup.
- Sequential completions all passed with HTTP 200 and server load returning to 0:
  max_tokens 8 (`13.49s`, first-request JIT), 16 (`1.89s`), 17 (`1.97s`), and
  32 (`3.36s`).
- Two concurrent 32-token `/v1/completions` requests passed with HTTP 200 in
  `2.79s` and `3.40s`; outputs were coherent and usage matched requested lengths.
- Post-idle sanity passed after 45s idle: health `200` and a new 8-token request
  returned in `1.84s`.
- No `ERROR`, traceback, `RuntimeError`, or `EnforceNotMet` appeared in the
  fresh server log after these requests.

Captured validation used `FULL_AND_PIECEWISE` on `ibm-cnode-9`, physical GPUs
1/2, main-process container `tidar_m6_cg_codex_9_053131`, log
`/shared/home/jinzhao/tidar_m6_logs/20260703_053131_cnode9_v2_dp2_ep_captured_mainproc_server.log`:

- Startup completed with async scheduling, V2 runner, DP=2, EP=2, AgRs all2all,
  AITER FA, explicit capture sizes `[1, 2, 17, 34]`, and
  `CUDAGraphMode.FULL_AND_PIECEWISE`.
- Cold boot paid AITER/torch compile, then captured all four target graphs and
  TiDAR drafter graphs; the log reports `Graph capturing finished in 38 secs`.
- Sequential completions max_tokens 8/16/17/32 all returned HTTP 200; server
  load returned to 0 after each request.
- Two concurrent 32-token `/v1/completions` requests both returned HTTP 200 in
  `0.54s` after warmup; outputs and usage lengths matched the requests.
- Post-idle sanity passed after 45s idle: health `200` and a new 8-token request
  returned HTTP 200 in `0.60s`.
- Final log sweep found no `ERROR`, traceback, `RuntimeError`, `EnforceNotMet`,
  `AssertionError`, or `ValueError`.
- Infra notes: a prior `ibm-cnode-20` captured attempt reached AITER JIT but was
  lost when the long-lived `sleep infinity` container disappeared; the successful
  `ibm-cnode-9` run made the build+serve process PID 1. An `ibm-cnode-123` retry
  failed before model load because a `verl-wrapper-train` job had already taken
  about 62% VRAM on all GPUs, leaving only ~72.5 GiB free per visible device.

Conclusion after M6/M6.5: the single-GPU V2 TiDAR path is cleaned/syntax-checked,
V2 DP+EP eager and captured request handling now work for sequential,
concurrent, and post-idle smoke requests, and DP+EP+EPLB eager now survives
post-idle smoke under the env-gated DP keepalive workaround. EPLB is no longer
untouched, but the keepalive workaround still busy-spins one-token dummy batches
while idle and should remain opt-in until the DP coordinator pause/resume path is
fixed directly.

## Revised plan: Path B, done incrementally (not "A then B")

1. **Milestone 1 (foundation): DONE 2026-07-01.** V2 now handles SMoE-AR via the
   fork's existing lower-level MambaSpec cache machinery plus the hybrid KV layout fix.
   A full v0.24 `model_states/` backport may still be the cleaner M3/TiDAR shape, but
   it is no longer required just to run SMoE-AR on V2.
2. **Milestone 2 (gate): CHECKED 2026-07-02.** V2 async composes with SMoE-AR/Mamba
   state and matches sync. Plain-AR throughput improves modestly: eager b8 +4.6%,
   captured b8 +9.7% at max_tokens=512.
   **M2.5 capture gate is unblocked:** graph capture completes and b=8 captured replay
   matches the compiled no-cudagraph control exactly after the AITER `max_query_len` fix.
   M3 can start, but the perf case now depends on TiDAR TF exposing more host overhead
   than plain AR.
3. **Milestone 3 (gate): DONE 2026-07-02.** TiDAR TF now runs on the V2 runner in
   sync/eager mode via a V2-native `gpu/spec_decode/tidar.py` self-speculator, and
   matches old-runner greedy output on the same prompt.
4. **Milestone 4: PERF GATE PASSED 2026-07-02.** TiDAR async + captured verify and
   drafter graphs run on V2.
5. **M4.5/M5 hardening: PASSED 2026-07-02.** Latest b8/MT512 async captured is
   `553.839 tok/s`, above the old-runner `177 tok/s` baseline. Broader b1/b3/b8/b16
   sweep passed after removing debug trace hooks. Per-seq-causal Triton attention
   (+ the OOB fix) remains optional for TiDAR TF and belongs to the broader
   dLLM-diffusion path.
6. **M6 landing prep / DP+EP eager+captured smoke: PASSED 2026-07-03.** Review
   cleanup and local/container syntax checks passed. V2 DP+EP eager and captured
   server initialization/request handling now work on two MI300X GPUs after the
   V2 TiDAR DP draft-fold, idle dummy-draft, and zero-token busy-loop guard
   fixes. Sequential 8/16/17/32 token requests, two concurrent 32-token
   requests, and post-idle sanity returned HTTP 200 with no fresh log errors.
   Captured `FULL_AND_PIECEWISE` verified V2/TiDAR drafter graph capture under
   DP+EP.
7. **M6.5 DP+EP+EPLB eager smoke: PASSED 2026-07-03.** On `ibm-cnode-123`,
   GPUs 6/7, `VLLM_USE_V2_MODEL_RUNNER=1`, async scheduling, DP=2, EP=2,
   AgRs all2all, AITER FA, TiDAR TF, and `--enable-eplb` booted and served
   post-idle requests when `VLLM_TIDAR_V2_DP_KEEPALIVE=1` was enabled. First
   8-token completion returned HTTP 200 in `13.941s`; after 75s idle, health was
   200 and the next 8-token completion returned in `1.579s`; after another 30s
   idle, a third 8-token completion returned in `7.646s`. Log
   `/shared/home/jinzhao/tidar_m6_logs/20260703_110128_cnode123_v2_dp2_ep_eplb0_debug12b_keepalive_eager_server.log`
   shows profile EPLB rearrange, live-request EPLB steps/rearranges, idle
   `eplb_skip_all_dummy`, keepalive markers on both DP ranks, and no runtime
   errors/tracebacks/assertions. Caveat: keepalive busy-spins idle dummy batches,
   so the production-quality fix is still to repair coordinator pause/resume.
8. **M6.6 DP+EP+EPLB captured smoke: BLOCKED 2026-07-03.** Captured
   `FULL_AND_PIECEWISE` EPLB with keepalive boots and serves sequential
   max_tokens 8/16/17/32, but the two-concurrent-requests smoke crashes in
   AITER MoE at the first concurrent 32-token pair. Baseline captured run:
   `ibm-cnode-123`, GPUs 6/7, container
   `tidar_m6_eplb_dbg14_keepalive_captured_codex_123_20260703_170011`, log
   `/shared/home/jinzhao/tidar_m6_logs/20260703_170011_cnode123_v2_dp2_ep_eplb0_dbg14_keepalive_captured_server.log`;
   concurrent clients disconnected after `6.46s`, and the log shows MoE shape
   `(304, 32, 2048, 2048, ...)` / `estimated_m_per_expert=4`, `Memory access
   fault`, then an engine death. Isolation on `ibm-cnode-19` showed eager EPLB
   concurrency passes (`dbg16`, log
   `/shared/home/jinzhao/tidar_m6_logs/20260703_173016_cnode19_v2_dp2_ep_eplb0_dbg16_keepalive_eager_conc_server.log`).
   Adding diagnostic env `VLLM_TIDAR_V2_DISABLE_DRAFT_CUDAGRAPH=1` to skip
   TiDAR drafter graph capture did not fix captured EPLB; `dbg17` still passed
   sequential requests then crashed on the same concurrent `(304,32,...)` AITER
   MoE shape (log
   `/shared/home/jinzhao/tidar_m6_logs/20260703_174602_cnode19_v2_dp2_ep_eplb0_dbg17_keepalive_cg_nodraftcg_server.log`).
   Therefore the next investigation is captured main graph / EPLB state or
   rearrangement interaction under concurrent DP+EP, not drafter graph replay
   alone. The no-keepalive ADD-wake probe (`dbg13`) also still hung post-idle,
   so `VLLM_TIDAR_V2_DP_KEEPALIVE=1` remains required while EPLB idle
   pause/resume is repaired directly.

Path B is still the 177-beating route in the working tree for single-GPU and
DP+EP without EPLB. The current blocked gate is captured EPLB concurrency; do not
move EPLB captured to production until that crash is fixed or EPLB capture is
deliberately disabled.

## Open questions to resolve before committing

- Does SMoE-AR run on the fork's V2 runner today? Yes — M1 landed 2026-07-01.
- Is TiDAR's SF/TF proposer portable to the V2 spec-decode interface as cleanly as EAGLE? Yes for TiDAR TF: sync/eager, async/eager, and captured verify+drafter graph paths run. The performance gate is passed at b8/MT512 (`553.839 tok/s` latest; first M4 pass was `405.651 tok/s`).
- Why can the latest short V2 accept numbers look lower than the v0.16 6-7 handoff table? Mostly checkpoint/workload/accounting: v0.16 used `iter_0012000`, AIME thinking-off, `MT=2000`, and metric windows; current parity probes use `iter_0012600`, and the old denominator undercounted once requests finished early. The back-to-back H100 control above confirms `iter_0012000` accepts about `0.75-1.36` more tokens than `iter_0012600` under the exact same current V2 TF script. Target/draft temps are both `0.0`. If AMD accept collapses to ~1.0 with incoherent text, check that ROCm logs show `Using TiDAR TF paged attention in ROCm AITER-FA` for draft and verify.
- Does the V2 runner's async path compose with the CCA stash + mamba state updates? Yes. Sync/async hashes matched on eager and captured plain AR. Perf is modest on plain AR: eager b8 +4.6%, captured b8 +9.7%. Captured graph construction and replay now work on the vectorized CCA path after the AITER `max_query_len` metadata fix; fused CCA capture is still unsafe.
- EP/EPLB (DeepEP/allgather-reducescatter all2all) behavior on the V2 runner on MI300X. M6 showed DP+EP eager/captured server init and request handling work with AgRs all2all and expert sharding after the V2 TiDAR DP draft-fold/dummy-draft fixes. M6.5 showed DP+EP+EPLB eager post-idle smoke works with `VLLM_TIDAR_V2_DP_KEEPALIVE=1`, but that workaround busy-spins dummy batches while idle. M6.6 showed captured EPLB boots and sequential requests pass, but concurrent captured requests crash with an AITER MoE GPU memory access fault at the `(304,32,...)` shape; eager EPLB concurrency passes, and disabling TiDAR drafter graph capture does not fix the captured crash.
