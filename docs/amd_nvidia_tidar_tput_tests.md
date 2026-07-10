# TiDAR TF on AMD: Throughput Gap and Reproducer

_Last updated: 2026-07-10. Audience: AMD and vLLM performance engineers._

Repository: [Zyphra/vllm-smoe-amd](https://github.com/Zyphra/vllm-smoe-amd),
branch `jinzhao/tidar_v024`. Checkpoint:
[Zyphra-staging/smoediffusion_128k-hf_iter_0012600](https://huggingface.co/Zyphra-staging/smoediffusion_128k-hf_iter_0012600).

## Executive Summary

TiDAR two-forward (TF) is correct and fully operational on a single MI300X:

- vLLM V2 GPU runner with async scheduling
- `FULL_AND_PIECEWISE` cudagraph replay for target and draft passes
- `ROCM_AITER_FA` attention and AITER unquantized MoE
- bidirectional draft attention and causal verify attention
- correct CCA recurrent-state handling and rejection sampling

The open issue is performance, not acceptance. In the primary matched run, AMD
AR reaches `0.70-0.81x` NVIDIA AR at b8-b64, but AMD TF reaches only
`0.44-0.46x` NVIDIA TF. NVIDIA gets a `1.25-1.37x` TF/AR speedup at b8-b16;
AMD gets `0.83x` and `0.76x`, respectively.

A matched b16 profile localizes the gap inside the two model forwards. Relative
to H100, the MI300X target pass is `44%` slower and its draft pass is `50%`
slower, while rejection sampling differs by only `0.19 ms`. The best next work
is therefore AITER MoE, AITER attention, and CCA kernel tuning at TiDAR's
expanded graph shapes.

## Branch Provenance and v0.24 Adoption

`jinzhao/tidar_v024` is based directly on `jinzhao/tidar_v016` at commit
`12cb2e780`. The main V2 TiDAR implementation and graph-shape support landed in
`6bbca0313`, followed by the fast LM-head restoration, AMD paged-attention
default, scheduler/profiling refinements, and benchmark documentation.

Despite the branch name, this is not stock vLLM v0.24 and it was not rebased
onto the upstream v0.24 source tree. It is a selective backport into the
Zyphra v0.16-based fork. In practical terms, it cherry-picked the useful
v0.24/DiffusionGemma architecture while preserving the fork's SMoE, CCA,
AITER, EP, and TiDAR code. The capabilities were ported into branch-local code;
they are not a set of preserved upstream cherry-pick commits.

The starting fork already contained a fairly complete V2 GPU runner with
`AsyncOutput`, EAGLE-style speculative decode, scheduler integration, and
cudagraph management. What it lacked was hybrid Mamba/CCA support and TiDAR.
A plain SMoE AR load initially failed because V2 assumed every KV-cache group
was an `AttentionSpec`; the model also has `MambaSpec` CCA state groups.

The v0.24/DiffusionGemma design informed these branch changes:

| v0.24/DiffusionGemma capability | Adoption in `jinzhao/tidar_v024` |
|---|---|
| V2 async scheduling | Reuses the fork's V2 `AsyncOutput` path and permits TiDAR speculative decoding under the async gate. GPU outputs copy to host on a separate stream while request state is updated. |
| Hybrid Mamba model state | Adds `MambaSpec` SSM/conv-cache reshaping and the hybrid attention/Mamba layout fix in `gpu/attn_utils.py`. TiDAR then adds stable request-keyed CCA slots and post-rejection state commit. |
| V2 speculative-decode data path | Adds a V2-native self-draft `TiDARSpeculator` next to EAGLE and feeds its K draft IDs through the normal V2 scheduler and input expansion path. |
| Causal and bidirectional attention | Preserves the important per-pass behavior: target verify is causal and draft is bidirectional. AMD uses the fork's TF paged AITER-FA helper and its causal flag. |
| Cudagraph-friendly execution | Extends the V2 graph manager for exact uniform TiDAR verify shapes and gives the self-speculator its own persistent-buffer draft graphs. |
| Custom acceptance | Reuses the V2 sampler plus its GPU Triton prefix-rejection kernel. The accepted count remains on device long enough to commit the correct CCA candidate state before async output completion. |

The full upstream dLLM framework was deliberately not copied. This branch has
no `gpu/model_states/`, `MambaHybridModelState`, or `config/diffusion.py`.
Instead of implementing a `TiDARSMoEModelState` with `prepare_attn`,
`postprocess_state`, and `custom_sampler` hooks, it implements the equivalent
minimum behavior in `gpu/attn_utils.py`, `gpu/model_runner.py`, CCA, and
`gpu/spec_decode/tidar.py`. This was the smaller route to the performance prize
without re-porting the entire Zyphra stack onto upstream v0.24.

The v0.24 per-sequence-causal unified-attention path is also not required by
two-forward TiDAR. TF runs draft and verify as separate forwards, so an AITER-FA
causal flag per pass is sufficient. Per-sequence causal attention remains
relevant to the optional single-forward diffusion path, not this benchmark.

## Current TiDAR TF Implementation

TiDAR is a self-drafter: the same SMoE target model performs both forwards. For
K=16, the steady-state V2 loop is:

1. **Schedule and verify.** The scheduler supplies the K draft IDs produced by
   the preceding iteration. The normal V2 main runner expands each request into
   the proposal plus one target slot and runs the causal target forward.
2. **Sample and reject.** The regular V2 sampler produces target tokens at the
   K+1 verifier positions. A Triton kernel accepts the matching draft prefix;
   on all-accept it emits the target's bonus token. `num_sampled` therefore
   includes the normal `+1` token used by the acceptance metric.
3. **Commit CCA state.** CCA stashes the candidate recurrent states created by
   verify. The runner maps `num_sampled - 1` to the accepted candidate and
   commits that SSM/conv state to the request's stable AR slot. Stable slots are
   necessary because async scheduling can compact or reorder batch rows.
4. **Self-draft.** `TiDARSpeculator` builds
   `[last_sampled_token, mask, ..., mask]`, assigns K+1 consecutive positions,
   and runs the same model with `causal=False`. The draft pass reads the
   committed AR CCA state and writes separate scratch state. Logits from the K
   mask positions produce the next draft IDs.
5. **Propagate asynchronously.** Draft IDs are stored in V2 request state and
   passed back to the scheduler for the next verify iteration. Sampled output
   IDs and accepted counts are copied through `AsyncOutput` without imposing
   the old synchronous per-step host barrier.

The production path uses `tidar_diff_temperature=0`, so draft IDs are argmax.
The V2 prefix matcher is correct for this deterministic draft distribution.
Non-zero draft temperature can generate draft probabilities in the speculator,
but those probabilities are not yet wired into the V2 rejection kernel and
remain experimental. Target sampling temperature, including the primary
temperature-0.6 benchmark below, is supported.

Both forwards have the uniform expanded size `B * (K + 1) = 17B` in steady
state. At b64 this is 1,088 tokens. Verify graphs replay through the V2
`CudaGraphManager`; draft graphs are captured and replayed inside
`TiDARSpeculator` using persistent input, position, block-table, and slot-mapping
buffers. Unmatched or mixed shapes fall back to eager execution.

On AMD, the two attention modes must appear independently in the log:

```text
Using TiDAR TF paged attention in ROCm AITER-FA (S=17, causal=True).
Using TiDAR TF paged attention in ROCm AITER-FA (S=17, causal=False).
```

The fallback TiDAR mask token is ID `4` when the checkpoint does not define one.
This matches the old runner and was required for old/V2 greedy output parity.
The old TiDAR implementation remains in `vllm/v1/spec_decode/tidar.py`; the new
path is selected with `VLLM_USE_V2_MODEL_RUNNER=1` and lives under
`vllm/v1/worker/gpu/`.

The single-GPU path described here is the validated throughput target. DP+EP
has passed eager and captured smoke without EPLB, and eager DP+EP+EPLB has also
passed. Concurrent DP+EP+EPLB with main-graph capture still has an independent
AITER MoE memory fault; it is not part of the single-GPU throughput gap below.

For historical context, the old AMD runner reached `177 tok/s` at b8/MT512.
The V2 async and captured path reached `553.839 tok/s` on the same broad
production shape, a `3.13x` improvement. That milestone removed the original
host-overhead bottleneck; the tables below isolate the remaining AMD device-side
gap with matched AMD/NVIDIA workloads.

## Primary Matched Result

This is the collaborator-facing comparison: checkpoint `iter_0012600`, AIME25
thinking-off prompts, exactly one BOS, MT4000, target temperature `0.6`, argmax
draft, K=16, ignore EOS, one request per active sequence, and no refill.
NVIDIA b1/b8 are three-seed medians; NVIDIA b16/b64 and all AMD rows are seed 0.

The public checkpoint already has the metadata used by both platforms:
`residual_in_fp32=false` and `mamba_cache_dtype=float32`.

Both AR and TF use V2 async scheduling and `FULL_AND_PIECEWISE`. H100 uses
`FLASH_ATTN` v3. MI300X uses `ROCM_AITER_FA`, AITER unquantized MoE, and the TF
paged no-splits path. The AR baseline uses the same platform backend as TF.

Throughput is output tokens/second. Acceptance uses the corrected active-request
denominator and includes TiDAR's normal `+1` sampled token.

| bsz | NVIDIA AR | NVIDIA TF / acc | TF/AR | AMD AR | AMD TF / acc | TF/AR | AMD/NVIDIA AR | AMD/NVIDIA TF |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `82.258` | `132.427` / `4.619` | `1.61x` | `60.702` | `60.753` / `4.209` | `1.00x` | `0.74x` | `0.46x` |
| 8 | `542.912` | `680.086` / `5.317` | `1.25x` | `377.326` | `314.923` / `5.422` | `0.83x` | `0.70x` | `0.46x` |
| 16 | `975.975` | `1333.164` / `5.263` | `1.37x` | `773.367` | `587.371` / `5.444` | `0.76x` | `0.79x` | `0.44x` |
| 64 | `3141.139` | `4043.730` / `5.942` | `1.29x` | `2532.184` | `1788.693` / `5.002` | `0.71x` | `0.81x` | `0.44x` |

The b8 and b16 rows are the clearest evidence. Acceptance is essentially equal
across platforms, but AMD turns TF into a slowdown while NVIDIA gets a material
speedup. The b64 AMD graph includes the full 1,088-token TiDAR shape, so this is
not the old cudagraph-cap failure.

The probes do not apply a chat template. They tokenize with
`add_special_tokens=False`, remove duplicate leading BOS IDs, and force exactly
one BOS. This avoids a known template difference between checkpoint revisions.

Source logs:

- NVIDIA: `/data/home/jinzhao/nv_v2_tidar_logs/iter12600_iter12000meta_mt4k_t06_d0_n1_vp16_20260709_190123/`
- AMD: `/shared/home/jinzhao/tfscope/amd_iter12600_iter12000meta_mt4k_t06_d0_n1_c17_20260710_010137/`

## Greedy Diagnostic

The MT5000 greedy run shows the same high-batch conclusion under a second
sampling configuration. Both platforms use target/draft temperature `0.0`,
warmup 64, ignore EOS, seed 0, and `n_sample=1`.

| bsz | NVIDIA AR | NVIDIA TF / acc | TF/AR | AMD AR | AMD TF / acc | TF/AR | AMD/NVIDIA AR | AMD/NVIDIA TF |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `79.723` | `207.554` / `7.562` | `2.60x` | `53.828` | `62.296` / `4.703` | `1.16x` | `0.68x` | `0.30x` |
| 8 | `539.015` | `972.872` / `6.136` | `1.80x` | `382.078` | `375.808` / `6.746` | `0.98x` | `0.71x` | `0.39x` |
| 16 | `968.246` | `1603.001` / `6.802` | `1.66x` | `772.428` | `661.769` / `8.325` | `0.86x` | `0.80x` | `0.41x` |
| 64 | `3093.504` | `3482.665` / `6.936` | `1.13x` | `2509.826` | `2102.265` / `8.250` | `0.84x` | `0.81x` | `0.60x` |

AMD acceptance is higher than NVIDIA at b8-b64, yet AMD TF remains slower than
AMD AR. This rules out rejection quality as the cause of the high-batch gap.
The b1 acceptance difference is a single-prompt trajectory effect and is not a
useful platform-level acceptance comparison.

Source logs:

- NVIDIA: `/data/home/jinzhao/nv_v2_tidar_logs/iter12600_forcebos_mt5k_n1_vp16_20260709_175900/`
- AMD: `/shared/home/jinzhao/tfscope/amd_iter12600_forcebos_mt5k_n1_c17_20260710_002037/`

All runs reached exactly 5,000 output tokens per request with no probe errors.
Each b64 mode generated 320,000 output tokens.

## Profile Evidence

This matched b16/MT512 profile uses the same checkpoint, prompt-token path,
K=16, `n_sample=1`, and platform-specific production attention backend:

| Platform and backend | Throughput / acc | Target forward | Draft forward | Reject sampler |
|---|---:|---:|---:|---:|
| H100, `FLASH_ATTN` v3 | `1394.609` / `6.307` | `20.067 ms` | `17.579 ms` | `0.989 ms` |
| MI300X, `ROCM_AITER_FA` | `1197.474` / `5.796` | `28.821 ms` | `26.314 ms` | `1.175 ms` |

The target plus draft model time is `37.646 ms` on H100 and `55.135 ms` on
MI300X, a `46%` increase. Rejection sampling adds only `0.186 ms` to the AMD/NV
difference. Optimize inside the captured model graphs before changing the
sampler or acceptance algorithm.

Profile logs:

- NVIDIA: `/data/home/jinzhao/nv_v2_tidar_logs/iter12600_profile/20260707_014246_nv_gpu6_iter12600_fp_profile_b16_mt512.log`
- AMD: `/shared/home/jinzhao/tfscope/amd_iter12600_profile/20260707_081033_amd_cnode107_gpu4_iter12600_fp_profile_b16_mt512_warmimg.log`

## Already Fixed

- The AITER-FA causal-mask bug is fixed: draft is bidirectional and verify is
  causal on separate passes.
- TiDAR TF defaults to the paged AITER attention path. Falling back to generic
  ROCm AITER extend attention caused incoherent drafts and acceptance near one.
- V2 async scheduling works with stable per-request CCA state slots.
- Target and drafter cudagraphs replay. The graph cap grows to at least
  `max_num_seqs * (K + 1)`, including 1,088 tokens at b64.
- The acceptance probe uses active requests as its denominator and includes the
  sampled bonus token.
- The fast LM-head path is present. Full-shape LM-head calls are below 1 ms and
  a BF16-output experiment did not improve throughput.

These checks matter because older logs can otherwise look like the same issue
while exercising a different attention path, an undersized graph cap, or the
legacy acceptance denominator.

## Optimization Targets

1. **AITER unquantized MoE.** Tune the E=16, H=2048, top-1 workload at TiDAR's
   uniform 17B graph shapes. The current path reports a generic two-stage AITER
   default and is the highest-priority kernel target.
2. **AITER attention by pass.** Measure and tune causal verify and bidirectional
   draft separately at b8, b16, and b64. Preserve the per-pass causal flag and
   the TF paged-attention path.
3. **CCA uniform-K+1 path.** A TiDAR-aware fused HIP kernel may remove CCA
   overhead, but it must preserve candidate-state stashing and separate drafter
   scratch writes.
4. **Graph-level breakdown.** Add per-layer or per-op device timings inside the
   target and draft graphs. Compare the AMD share of MoE, attention, CCA, norms,
   and LM head against the matched NVIDIA profile.
5. **Steady state versus drain.** For serving-style tests, report full-batch
   windows separately from low-active-request tail windows. Keep the pure-decode
   table above as the stable kernel baseline.

An earlier AMD screen found no win from switching to Triton MoE, the alternate
SMoE AITER op, removing router padding, enabling the CCA Triton fusion, or using
CCA unfold/einsum. Those variants should remain off unless their kernels change.

Success means TF is faster than the paired AMD AR baseline at b8, b16, and b64,
without lowering acceptance. Closing AMD/NVIDIA TF toward the corresponding AR
ratio is the cross-platform target.

## Code Map

| Area | File |
|---|---|
| V2 runner and TiDAR graph execution | `vllm/v1/worker/gpu/model_runner.py` |
| Async GPU-to-host output | `vllm/v1/worker/gpu/async_utils.py` |
| V2 self-draft implementation | `vllm/v1/worker/gpu/spec_decode/tidar.py` |
| Draft-token scheduler handoff | `vllm/v1/worker/gpu/spec_decode/utils.py` |
| Verify cudagraph selection | `vllm/v1/worker/gpu/cudagraph_utils.py` |
| Per-pass AITER attention setup | `vllm/v1/worker/gpu/attn_utils.py` |
| ROCm AITER-FA backend | `vllm/v1/attention/backends/rocm_aiter_fa.py` |
| TF paged attention helper | `vllm/attention/ops/tf_attention.py` |
| CCA implementation | `vllm/model_executor/layers/mamba/cca.py` |
| SMoE routing, MoE, and LM head | `vllm/model_executor/models/smoe.py` |
| Graph-cap configuration | `vllm/config/vllm.py` |
| Corrected rejection metrics | `vllm/v1/sample/rejection_sampler.py` |
| Matched benchmark driver | `benchmarks/tidar/run_iter12600_tput.sh` |

## Reproducer

Clone the tested branch and download the public checkpoint:

```bash
git clone --branch jinzhao/tidar_v024 \
    https://github.com/Zyphra/vllm-smoe-amd.git
cd vllm-smoe-amd

python3 -m pip install -q huggingface_hub
export CKPT=/shared/home/$USER/checkpoints/smoediffusion_128k-hf_iter_0012600
huggingface-cli download Zyphra-staging/smoediffusion_128k-hf_iter_0012600 \
    --local-dir "$CKPT"
```

The tested AMD environment is `zyphra/rocm-primus:aiter_pa_swa` with Torch
2.10, Triton 3.7, and ROCm 7.2. Mount the repo at `/work` and the checkpoint
filesystem at `/shared`, then build inside a fresh container:

```bash
cd /work
pip install -q "setuptools>=77.0.3,<81.0.0" "setuptools-scm>=8"
pip install -q --no-build-isolation -e .
```

The first model load in a fresh image can spend about 12 minutes compiling
AITER kernels. Use a free GPU; on the shared IBM nodes, do not use GPU 0.

Run the primary MT4000, target-temperature-0.6 comparison:

```bash
export CKPT=${CKPT:-/shared/home/$USER/checkpoints/smoediffusion_128k-hf_iter_0012600}
export DATA=benchmarks/tidar/aime25_zpo_texts.json
export BACKEND=ROCM_AITER_FA
export GPU=1
export BATCHES="1 8 16 64"
export MAX_TOKENS=4000
export MAX_MODEL_LEN=12000
export TARGET_TEMP=0.6
export DRAFT_TEMP=0
export N_SAMPLE_AR=1
export N_SAMPLE_TF=1
export LOGROOT=/shared/home/$USER/tfscope/iter12600_mt4k_t06_$(date +%Y%m%d_%H%M%S)

bash benchmarks/tidar/run_iter12600_tput.sh
```

For a quick validation, use `BATCHES="1 16"`. For the greedy diagnostic, keep
the same command and set `MAX_TOKENS=5000 TARGET_TEMP=0 DRAFT_TEMP=0`.

The driver selects this AMD path:

```text
VLLM_USE_V2_MODEL_RUNNER=1
VLLM_ATTENTION_BACKEND=ROCM_AITER_FA
VLLM_ROCM_USE_AITER=1
VLLM_ROCM_USE_AITER_MHA=1
VLLM_ROCM_USE_AITER_MOE=1
VLLM_ROCM_MOE_PADDING=1
VLLM_TIDAR_TWO_FORWARD=1
VLLM_TIDAR_USE_TF_PAGED_ATTENTION=1
VLLM_TIDAR_TF_PAGED_NO_SPLITS=1
VLLM_TIDAR_FA_NO_SPLITS=1
```

For the H100 reference, use the same script and parameters with:

```bash
BACKEND=FLASH_ATTN VLLM_FLASH_ATTN_VERSION=3 GPU=0 \
    bash benchmarks/tidar/run_iter12600_tput.sh
```

## Result Acceptance Checklist

Every accepted run must satisfy all of the following:

- `PATCH_PROBE_CONTEXT` reports `force_bos=true`,
  `leading4=[2,105,9731,107]`, and `leading_bos_count=1`.
- AMD logs say `Using Aiter Flash Attention backend` and
  `Using ROCm AITER backend for Unquantized MoE`.
- AMD TF logs show the paged path for both
  `S=17, causal=True` and `S=17, causal=False`.
- The b64 startup log includes cudagraph capture size `1088`.
- AR and TF use the same checkpoint, prompts, active batch, output length,
  target temperature, EOS policy, and request count.
- Acceptance remains in the expected band and output is coherent.

If acceptance collapses toward one, first check BOS and the two paged-attention
causal signatures. If those are correct, compare target and draft forward time
before changing rejection sampling.

Expected greedy AMD ballpark:

```text
b1:  AR 53.8 tok/s, TF 62.3 tok/s, accept 4.70
b8:  AR 382.1 tok/s, TF 375.8 tok/s, accept 6.75
b16: AR 772.4 tok/s, TF 661.8 tok/s, accept 8.33
b64: AR 2509.8 tok/s, TF 2102.3 tok/s, accept 8.25
```

For migration history and distributed EPLB status, see
`docs/TIDAR_AMD_HANDOFF.md`. Those topics are intentionally outside this
single-GPU throughput handoff.
