# TiDAR-on-AMD Handoff

_Last updated: 2026-07-13. Owner: jinzhao. Scope: TiDAR two-forward (TF)
decode on MI300X for the 80B SMoE/Zaya checkpoint._

## 1. Current State

Path B is working on one GPU: TiDAR TF runs on the fork's V2 GPU runner
(`vllm/v1/worker/gpu/model_runner.py`, selected by
`VLLM_USE_V2_MODEL_RUNNER=1`) with native async scheduling and cudagraph replay
for both target verification and self-drafting.

This is not stock vLLM v0.24. Branch `jinzhao/tidar_v024` is based on the
v0.16-family SMoE fork and adopts the useful v0.24/DiffusionGemma architecture:
V2 runner execution, speculative-decode data flow, hybrid recurrent state,
async output, and graph-friendly model-state handling.

Current performance claims use **thinking-on**, with EOS enabled and
`max_tokens=10000` as a safety cap. The probes replace the pre-rendered closed
`<think>\n</think>\n\n` suffix with an open `<think>\n` suffix before direct
tokenization. Thinking-on trajectories are long, and several rows reach the
10k cap before EOS; the focused report labels every result as `EOS` or `CAP`.

Matched single-GPU results use `iter_0012600`, one forced BOS, target
temperature 0.6, argmax draft, K=16, seed 0, V2 async, and
`FULL_AND_PIECEWISE`:

| bsz | H100 AR / TF | H100 TF/AR | MI300X AR / TF | MI300X TF/AR | MI300X acc |
|---:|---:|---:|---:|---:|---:|
| 1 | `84.6 / 171.0` | `2.02x` | `62.9 / 146.2` | `2.33x` | `5.495` |
| 8 | `623.0 / 1171.3` | `1.88x` | `444.6 / 843.4` | `1.90x` | `4.690` |
| 16 | `1247.9 / 1916.8` | `1.54x` | `848.5 / 1745.4` | `2.06x` | `5.227` |
| 32 | `2124.2 / 3588.0` | `1.69x` | `1641.8 / 2456.5` | `1.50x` | `4.662` |
| 64 | `3649.2 / 5315.1` | `1.46x` | `3157.2 / 3409.7` | `1.08x` | `4.725` |

Rates are device-event output tok/s. Acceptance includes the bonus token.
MI300X matches or exceeds H100's observed TF/AR gain at b1-b16, but is lower at
b32-b64. The large-b result is confounded by different sampled context-length
trajectories: at b64, MI300X AR ends at 5,599 output tokens while MI300X TF
runs to 7,998, whereas H100 AR and TF run to 10,000 and 9,581. A matched-context
profile is required before assigning that difference to an AMD kernel. Full
lengths, completion status, backend details, logs, and the reproducer are in
`docs/amd_nvidia_tidar_tput_tests.md`.

## 2. Migration Status

### M1: SMoE-AR on V2

Complete. The hybrid SMoE loads and decodes on the V2 runner with CCA
recurrent-state cache handling and AITER unquantized MoE.

### M2: Async SMoE-AR

Complete. V2 async scheduling and captured AR run correctly with the hybrid
state cache.

### M3: TiDAR TF on V2

Complete for single GPU. The implementation uses a self-draft
`TiDARSpeculator` integrated with the existing V2 speculative-decode path.
Target verification is causal; drafting is bidirectional. Prefix rejection,
the bonus token, accepted-state commit, and mask-token handling are wired.

### M4: Validate and Measure

Complete for the EOS-enabled, thinking-on single-GPU matrix above. MI300X's
observed TF/AR gain matches or exceeds H100 at b1-b16. The lower b32-b64 ratios
need a matched-context profile because AR and TF sampled different output
lengths.

### M5-M6: Distributed Smoke

- DP=2 + EP passes eager and captured smoke without EPLB.
- DP+EP+EPLB eager passes with `VLLM_TIDAR_V2_DP_KEEPALIVE=1`.
- Captured DP+EP+EPLB concurrency remains blocked. It reaches health and serves
  sequential requests, then faults in AITER MoE on the first two-request
  concurrent smoke. Eager EPLB concurrency and captured DP+EP without EPLB
  pass; disabling TiDAR draft graph capture does not fix it.

The distributed blocker is therefore captured main-graph interaction with
EPLB state or expert rearrangement, not the single-GPU TiDAR path.

## 3. Build and Run

Remote development repo:

```bash
/shared/home/jinzhao/workspace/tidar/vllm-smoe-amd
```

Local repo:

```bash
/Users/jz/Documents/Diffusion RL/vllm-smoe-amd
```

Checkpoint:

```text
https://huggingface.co/Zyphra-staging/smoediffusion_128k-hf_iter_0012600
```

AMD image:

```text
zyphra/rocm-primus:aiter_pa_swa
```

Build inside a fresh container:

```bash
git config --global --add safe.directory \
    /shared/home/jinzhao/workspace/tidar/vllm-smoe-amd
pip install -q "setuptools>=77.0.3,<81.0.0" "setuptools-scm>=8"
pip install -q --no-build-isolation -e .
```

Operational notes:

- Use a fresh `docker run --rm`; long-lived containers may be reaped.
- Never use GPU 0 on shared IBM nodes.
- Check free nodes with `~/.ssh/check_gpus_ids ibm` and reservations with
  Slurm.
- AITER JIT can take roughly 12 minutes on the first model load.
- Two-GPU DP runs worked with `HIP_VISIBLE_DEVICES=6,7`.

TiDAR TF environment:

```bash
VLLM_USE_V2_MODEL_RUNNER=1
VLLM_ENABLE_V1_MULTIPROCESSING=0
VLLM_ATTENTION_BACKEND=ROCM_AITER_FA
VLLM_ROCM_USE_AITER=1
VLLM_ROCM_USE_AITER_MHA=1
VLLM_ROCM_USE_AITER_MOE=1
VLLM_ROCM_MOE_PADDING=1
VLLM_TIDAR_TWO_FORWARD=1
VLLM_TIDAR_USE_TF_PAGED_ATTENTION=1
VLLM_SKIP_SDPA_PREINIT=1
VLLM_CCA_TRITON=1
VLLM_TIDAR_ROUTER_PAD=1
```

Use `benchmarks/tidar/slurm_lockstep_steady.sh` on AMD and
`benchmarks/tidar/run_lockstep_nvidia.sh` on H100. The exact thinking-on
commands are maintained in `docs/amd_nvidia_tidar_tput_tests.md`.

## 4. Implementation

For K=16, each TF iteration performs:

1. Causal target verification over K draft IDs plus one target slot.
2. Target sampling and prefix rejection, including the bonus token.
3. Commit of the accepted target CCA candidate state to a stable request slot.
4. Bidirectional self-draft over `[last_token, mask, ..., mask]`.
5. Async output propagation and scheduler handoff of the next K draft IDs.

Both forwards have size `B * (K + 1) = 17B`. TiDAR TF does not require
per-sequence-causal unified attention because target and draft are separate
forwards. AITER-FA selects causality per pass.

Core files:

- `vllm/v1/worker/gpu/model_runner.py`
- `vllm/v1/worker/gpu/spec_decode/tidar.py`
- `vllm/v1/worker/gpu/attn_utils.py`
- `vllm/v1/worker/gpu/async_utils.py`
- `vllm/v1/attention/backends/rocm_aiter_fa.py`
- `vllm/attention/ops/tf_attention.py`
- `vllm/model_executor/layers/mamba/cca.py`
- `vllm/model_executor/models/smoe.py`

Key choices:

- V2 TiDAR uses a focused self-draft speculator rather than a complete
  backport of upstream `gpu/model_states/`.
- CCA state uses stable per-request slots so async scheduling cannot move
  recurrent state between rows.
- Verify and draft cudagraphs use exact TiDAR shapes.
- The fallback mask token is ID 4 when the checkpoint lacks a configured mask
  token.
- On AMD, a correct run logs the TF paged AITER path for both `causal=True`
  target verification and `causal=False` drafting.

## 5. Scope and Next Work

The thinking-on data does not show an AMD-specific TiDAR speedup regression at
b1-b16. The b32-b64 gap must first be separated from the different sampled
context-length trajectories. Do not pursue cross-vendor bitwise matching or
production deterministic GEMM.

Priority work:

1. Profile matched-context b32-b64 target and draft steps, including
   long-prefix `ROCM_AITER_FA` with `S=17`.
2. Improve ordinary SMoE forward throughput on MI300X: normal BF16 dense GEMM
   and AITER unquantized MoE at `M=17B`.
3. Review the batch-invariant CCA convolution for production adoption and
   broader coverage.
4. Debug captured DP+EP+EPLB concurrency, focusing on main-graph and EPLB
   expert-state interaction.
5. Refine DP pause/resume so EPLB does not require keepalive busy-spin.
6. Decide whether to submit the independent v0.24 mixed-causal OOB fix in
   `docs/tidar_amd_handoff/`.

Per-sequence-causal unified attention and its OOB fix belong to the optional
single-forward diffusion path. They are not required for TiDAR TF's two
separate forwards.

## 6. References

- Focused thinking-on report: `docs/amd_nvidia_tidar_tput_tests.md`
- Migration scope: `docs/dllm_migration_scope.md`
- OOB materials: `docs/tidar_amd_handoff/`
- TF split-K design: `docs/tf_full_splitk_design.md`
- Memory entries: `reference_vllm_v024_dllm_is_tidar_tf.md`,
  `project_tf_on_aiter_fa.md`, `project_amd_tidar_host_overhead.md`, and
  `reference_tidar_amd_env.md`
