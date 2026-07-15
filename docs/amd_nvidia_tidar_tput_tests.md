# TiDAR TF on AMD: Thinking-On Throughput

_Last updated: 2026-07-13. Audience: AMD and vLLM performance engineers._

Repository: [Zyphra/vllm-smoe-amd](https://github.com/Zyphra/vllm-smoe-amd),
branch `jinzhao/tidar_v024`. Checkpoint:
[Zyphra-staging/smoediffusion_128k-hf_iter_0012600](https://huggingface.co/Zyphra-staging/smoediffusion_128k-hf_iter_0012600).

> **Prompt mode: thinking is ON.** The probes transform each pre-rendered
> `aime25_zpo_texts.json` prompt from the closed
> `<think>\n</think>\n\n` suffix to an open `<think>\n` suffix before direct
> tokenization. Every result logs `thinking_on=true` and the exact open suffix
> in `PATCH_PROBE_CONTEXT`.

## Result

TiDAR two-forward (TF) is operational on one MI300X with the vLLM V2 GPU
runner, async scheduling, target and draft cudagraph replay,
`ROCM_AITER_FA`, AITER unquantized MoE, CCA recurrent-state handling, and
prefix rejection.

EOS is enabled and `max_tokens=10000` is a safety cap. Thinking-on produces
much longer trajectories than thinking-off; several rows reach the cap before
EOS. The table marks every mode as `EOS` or `CAP` so cap-limited output cannot
be mistaken for natural completion.

One AIME25 prompt and seed are replicated across every request in a row. All
requests within a row produce the same length and exit together, so there is
no shrinking tail batch. Device-event rates exclude startup and prefill.

Both platforms use:

- Checkpoint `iter_0012600` and exactly one forced BOS.
- Target temperature `0.6`, argmax draft, K=16, and seed 0.
- V2 async scheduling and `FULL_AND_PIECEWISE` cudagraph capture.
- Explicit `--thinking-on` prompt transformation.
- EOS enabled; no `--ignore-eos`.

H100 uses `FLASH_ATTN` v3 and its default CCA convolution. MI300X uses
`ROCM_AITER_FA`, AITER unquantized MoE, and the batch-invariant CCA
convolution. The reported AMD TF rows use adaptive paged-attention split-K at
all batch sizes.

| bsz | NVIDIA AR (len/status) | NVIDIA TF / acc (len/status) | TF/AR | AMD AR (len/status) | AMD TF / acc (len/status) | TF/AR | AMD/NVIDIA AR | AMD/NVIDIA TF |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `84.637` (10000/CAP) | `171.047` / `5.435` (10000/CAP) | `2.02x` | `62.881` (10000/CAP) | `146.205` / `5.495` (10000/CAP) | `2.33x` | `0.74x` | `0.85x` |
| 8 | `623.028` (9406/EOS) | `1171.264` / `5.250` (10000/CAP) | `1.88x` | `444.620` (7441/EOS) | `843.351` / `4.690` (10000/CAP) | `1.90x` | `0.71x` | `0.72x` |
| 16 | `1247.850` (4987/EOS) | `1916.782` / `4.764` (8578/EOS) | `1.54x` | `848.493` (8058/EOS) | `1745.360` / `5.227` (7452/EOS) | `2.06x` | `0.68x` | `0.91x` |
| 32 | `2124.208` (10000/CAP) | `3588.006` / `5.095` (9581/EOS) | `1.69x` | `1641.767` (6279/EOS) | `2456.543` / `4.662` (8124/EOS) | `1.50x` | `0.77x` | `0.68x` |
| 64 | `3649.225` (10000/CAP) | `5315.146` / `5.046` (9581/EOS) | `1.46x` | `3157.244` (5599/EOS) | `3409.732` / `4.725` (7998/EOS) | `1.08x` | `0.87x` | `0.64x` |

All rates are device-event output tok/s. Length is output tokens per request,
and acceptance includes TiDAR's normal bonus token. The AMD b1 no-splits
control reached only `47.352 tok/s`, acceptance `5.431`, and 10000/CAP because
its mean TF step was `114.738 ms`; adaptive split-K is the selected b1 result.

## Interpretation

- At b1/b8/b16, observed MI300X TF/AR is comparable to or better than H100:
  `2.33x/1.90x/2.06x` AMD versus `2.02x/1.88x/1.54x` H100.
- At b32/b64, observed MI300X TF/AR falls to `1.50x/1.08x`, versus
  `1.69x/1.46x` on H100.
- The large-b ratios are not a matched-context kernel comparison. At b64,
  MI300X AR ends at 5,599 tokens while MI300X TF runs to 7,998; H100 AR and TF
  run to 10,000 and 9,581. TF/AR therefore averages different context-length
  distributions on each platform.
- Acceptance is broadly comparable: AMD spans `4.662-5.495` and H100 spans
  `4.764-5.435`. Acceptance alone does not explain the large-b throughput
  difference.
- Changing B changes model-kernel shapes and can alter the sampled trajectory.
  Plain AR lengths vary as strongly as TF lengths, so this is not evidence of
  rejection-sampler corruption.
- Bitwise AMD/NVIDIA agreement is neither expected nor required. Expensive
  deterministic GEMM is not a production recommendation.

This table is the requested same-configuration sampled-throughput comparison.
To isolate hardware efficiency at b32/b64, the next experiment should compare
fixed context buckets or matched engine iterations while still allowing EOS at
the request level.

## Useful AMD Work

1. Profile matched-context b32/b64 target and draft steps to separate the
   longer TF context distribution from kernel latency.
2. Tune long-prefix `ROCM_AITER_FA` for TiDAR's `S=17` target and draft passes.
3. Tune normal BF16 dense GEMMs and AITER unquantized MoE for `M=17B`:
   `M=17,136,272,544,1088` at b1/b8/b16/b32/b64.
4. Review the batch-invariant CCA convolution for production adoption and
   broader model coverage.
5. Recommend current AITER kernels and tuned configurations for this SMoE
   architecture on MI300X.

Do not require bitwise NVIDIA parity or enable the diagnostic deterministic
GEMM path for production.

## Branch Context

`jinzhao/tidar_v024` is not stock vLLM v0.24. It is based on the v0.16-family
SMoE fork and adopts the v0.24 V2-runner features needed by this model and by
DiffusionGemma-style execution:

- V2 GPU runner and async scheduler integration.
- Hybrid CCA recurrent-state cache preparation and commit.
- V2 speculative-decode data flow and rejection sampling.
- Verify and self-draft cudagraph replay.
- Per-pass causal selection for TiDAR's two forwards.

The complete upstream `gpu/model_states/` framework was not copied. Equivalent
minimum hooks live in `gpu/attn_utils.py`, `gpu/model_runner.py`, CCA, and
`gpu/spec_decode/tidar.py`.

For K=16, each TF iteration performs:

1. A causal target verify over K draft IDs plus one target slot.
2. Target sampling and prefix rejection, including the bonus token.
3. Commit of the accepted target CCA candidate state.
4. A bidirectional self-draft over `[last_token, mask, ..., mask]`.
5. Async output propagation and scheduler handoff of the next K draft IDs.

Both forwards have size `B * (K + 1) = 17B`. The target and draft are separate
forwards, so AITER-FA selects causality per pass.

On AMD, a correct run logs both:

```text
Using TiDAR TF paged attention in ROCm AITER-FA (S=17, causal=True).
Using TiDAR TF paged attention in ROCm AITER-FA (S=17, causal=False).
```

## Code Map

| Area | File |
|---|---|
| V2 runner and verify graph | `vllm/v1/worker/gpu/model_runner.py` |
| Async output | `vllm/v1/worker/gpu/async_utils.py` |
| V2 self-draft | `vllm/v1/worker/gpu/spec_decode/tidar.py` |
| Per-pass attention setup | `vllm/v1/worker/gpu/attn_utils.py` |
| ROCm AITER-FA backend | `vllm/v1/attention/backends/rocm_aiter_fa.py` |
| TF paged attention | `vllm/attention/ops/tf_attention.py` |
| CCA | `vllm/model_executor/layers/mamba/cca.py` |
| SMoE/MoE/LM head | `vllm/model_executor/models/smoe.py` |
| AR probe | `benchmarks/tidar/probe_v2_ar.py` |
| TF probe | `benchmarks/tidar/probe_v2_tidar_nv.py` |
| AMD Slurm driver | `benchmarks/tidar/slurm_lockstep_steady.sh` |
| NVIDIA driver | `benchmarks/tidar/run_lockstep_nvidia.sh` |

## Thinking-On Reproducer

Clone the tested branch and download the checkpoint:

```bash
git clone --branch jinzhao/tidar_v024 \
    https://github.com/Zyphra/vllm-smoe-amd.git
cd vllm-smoe-amd

python3 -m pip install -q huggingface_hub
export CKPT=/shared/home/$USER/checkpoints/smoediffusion_128k-hf_iter_0012600
huggingface-cli download Zyphra-staging/smoediffusion_128k-hf_iter_0012600 \
    --local-dir "$CKPT"
```

The tested AMD image is `zyphra/rocm-primus:aiter_pa_swa` with Torch 2.10,
Triton 3.7, and ROCm 7.2. A fresh image may spend about 12 minutes compiling
AITER kernels. On shared IBM nodes, do not use GPU 0.

Run AR and adaptive-split TF separately:

```bash
CKPT="$CKPT" BATCHES="1 8 16 32 64" RUN_AR=1 RUN_TF=0 \
    MAX_TOKENS=10000 CCA_BATCH_INVARIANT=1 IGNORE_EOS=0 THINKING_ON=1 \
    sbatch benchmarks/tidar/slurm_lockstep_steady.sh

CKPT="$CKPT" BATCHES="1 8 16 32 64" RUN_AR=0 RUN_TF=1 \
    MAX_TOKENS=10000 CCA_BATCH_INVARIANT=1 TF_PAGED_NO_SPLITS=0 \
    IGNORE_EOS=0 THINKING_ON=1 \
    sbatch benchmarks/tidar/slurm_lockstep_steady.sh
```

For H100, use:

```bash
CKPT="$CKPT" MAX_TOKENS=10000 IGNORE_EOS=0 THINKING_ON=1 \
    CASES="ar:1:1 tf:1:2 ar:8:3 tf:8:4 ar:16:5 tf:16:6" \
    bash benchmarks/tidar/run_lockstep_nvidia.sh
```

Run additional batches on free GPUs with the same settings. A valid context
record must show `thinking_on=true`, an open `<think>\n` prompt suffix, exactly
one leading BOS, EOS enabled, and the expected backend. A valid AMD TF run must
also report both paged-attention causal signatures and exact capture size
`17B`.

## Source Logs

- NVIDIA TF: `/data/groups/rl/jinzhao/nv_v2_tidar_logs/eos_thinkon_tf_iter12600_20260713/`
- NVIDIA AR: `/data/groups/rl/jinzhao/nv_v2_tidar_logs/eos_thinkon_ar_iter12600_20260713/`
- AMD AR: `/shared/home/jinzhao/tfscope/amd_lockstep_steady_271771/`
- AMD adaptive TF: `/shared/home/jinzhao/tfscope/amd_lockstep_steady_271809/`
- AMD b1 no-splits control: `/shared/home/jinzhao/tfscope/amd_lockstep_steady_271810/`
