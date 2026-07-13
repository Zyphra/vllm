# TiDAR TF on AMD: Natural-EOS Throughput

_Last updated: 2026-07-13. Audience: AMD and vLLM performance engineers._

Repository: [Zyphra/vllm-smoe-amd](https://github.com/Zyphra/vllm-smoe-amd),
branch `jinzhao/tidar_v024`. Checkpoint:
[Zyphra-staging/smoediffusion_128k-hf_iter_0012600](https://huggingface.co/Zyphra-staging/smoediffusion_128k-hf_iter_0012600).

> **Prompt mode: template-level thinking is OFF.** The tests use the
> pre-rendered `benchmarks/tidar/aime25_zpo_texts.json` prompts, which end with
> an empty, closed `<think>\n</think>\n\n` block before generation. The probes
> encode those strings directly and do not call `apply_chat_template` or set
> `enable_thinking=True`. The user text says "Let's think step by step," so the
> model may emit visible reasoning, but these are not thinking-on results.

## Result

TiDAR two-forward (TF) is operational on one MI300X with the vLLM V2 GPU
runner, async scheduling, target and draft cudagraph replay,
`ROCM_AITER_FA`, AITER unquantized MoE, CCA recurrent-state handling, and
prefix rejection.

These measurements enable EOS. `max_tokens=10000` is only a safety cap; every
run completes naturally after roughly 686-776 output tokens. One AIME25 prompt
and seed are replicated across every request in a row. All requests within a
row produce the same length and exit together, so there is no shrinking tail
batch. Device-event rates exclude startup and prefill.

Both platforms use:

- Checkpoint `iter_0012600` and exactly one forced BOS.
- Target temperature `0.6`, argmax draft, K=16, and seed 0.
- V2 async scheduling and `FULL_AND_PIECEWISE` cudagraph capture.
- Template-level thinking-off prompt token IDs.
- Natural EOS; no `--ignore-eos`.

H100 uses `FLASH_ATTN` v3 and its default CCA convolution. MI300X uses
`ROCM_AITER_FA`, AITER unquantized MoE, and the batch-invariant CCA
convolution. AMD TF uses adaptive paged-attention split-K at b8-b64; b1 uses
the faster no-splits configuration.

| bsz | NVIDIA AR (len) | NVIDIA TF / acc (len) | TF/AR | AMD AR (len) | AMD TF / acc (len) | TF/AR | AMD/NVIDIA AR | AMD/NVIDIA TF |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `80.733` (686) | `216.386` / `7.711` (686) | `2.68x` | `55.203` (687) | `191.585` / `7.408` (718) | `3.47x` | `0.68x` | `0.89x` |
| 8 | `612.671` (721) | `1435.108` / `6.505` (693) | `2.34x` | `453.177` (724) | `1404.922` / `6.901` (687) | `3.10x` | `0.74x` | `0.98x` |
| 16 | `1221.915` (772) | `3108.160` / `7.798` (686) | `2.54x` | `792.916` (777) | `2711.642` / `6.804` (693) | `3.42x` | `0.65x` | `0.87x` |
| 32 | `2337.489` (747) | `6071.548` / `8.070` (686) | `2.60x` | `1614.817` (770) | `4796.072` / `7.010` (722) | `2.97x` | `0.69x` | `0.79x` |
| 64 | `4508.023` (693) | `9731.729` / `7.977` (686) | `2.16x` | `3491.091` (693) | `7464.847` / `7.118` (776) | `2.14x` | `0.77x` | `0.77x` |

All rates are device-event output tok/s. Length is output tokens per request,
and acceptance includes TiDAR's normal bonus token. The AMD b1 adaptive-split
control reached `178.582 tok/s`, acceptance `5.953`, and length 749; no-splits
is better for this workload.

## Takeaways

- AMD's relative TiDAR gain is not worse than NVIDIA's. MI300X TF/AR is higher
  at b1-b32 and essentially equal at b64 (`2.14x` AMD versus `2.16x` H100).
- At b64, AMD/NVIDIA is `0.77x` for both AR and TF. The remaining absolute TF
  gap therefore tracks the base SMoE/AR platform gap rather than a separate
  TiDAR penalty.
- At b8, MI300X TF reaches `0.98x` H100 TF despite MI300X AR being `0.74x`
  H100 AR.
- Acceptance is healthy on both platforms. It varies across batch sizes
  because changing B changes model-kernel shapes and can eventually alter a
  sampled trajectory. Plain AR output lengths also vary with B, so this is not
  evidence of rejection-sampler corruption.
- Bitwise AMD/NVIDIA agreement is neither expected nor required. Expensive
  deterministic GEMM is not a production recommendation.

TF and AR can naturally sample different token trajectories and output
lengths. Their ratio compares each mode's natural-completion device rate; it
does not assert token-for-token output identity between the two modes.

## Useful AMD Work

The current data does not support asking AMD to fix a TiDAR-specific relative
speedup regression. The useful remaining work is absolute model-forward
performance:

1. Tune normal BF16 dense GEMMs and AITER unquantized MoE for TiDAR's exact
   `M=17B` shapes: `M=17,136,272,544,1088` at b1/b8/b16/b32/b64.
2. Improve ordinary SMoE V2 AR throughput; both TiDAR target and draft
   forwards inherit those gains.
3. Review the batch-invariant CCA convolution for production adoption and
   broader model coverage.
4. Recommend current AITER kernels and tuned configurations for this SMoE
   architecture on MI300X.

Do not require bitwise NVIDIA parity or enable the diagnostic deterministic
GEMM path for production.

## Branch Context

`jinzhao/tidar_v024` is not a stock vLLM v0.24 branch. It is based on the
v0.16-family SMoE fork and adopts the v0.24 V2-runner features needed by this
model and by DiffusionGemma-style execution:

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

## Natural-EOS Reproducer

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
    MAX_TOKENS=10000 CCA_BATCH_INVARIANT=1 IGNORE_EOS=0 \
    sbatch benchmarks/tidar/slurm_lockstep_steady.sh

CKPT="$CKPT" BATCHES="1 8 16 32 64" RUN_AR=0 RUN_TF=1 \
    MAX_TOKENS=10000 CCA_BATCH_INVARIANT=1 TF_PAGED_NO_SPLITS=0 \
    IGNORE_EOS=0 sbatch benchmarks/tidar/slurm_lockstep_steady.sh
```

Run the faster b1 no-splits control separately:

```bash
CKPT="$CKPT" BATCHES=1 RUN_AR=0 RUN_TF=1 MAX_TOKENS=10000 \
    CCA_BATCH_INVARIANT=1 TF_PAGED_NO_SPLITS=1 IGNORE_EOS=0 \
    sbatch benchmarks/tidar/slurm_lockstep_steady.sh
```

For H100, use:

```bash
CKPT="$CKPT" MAX_TOKENS=10000 IGNORE_EOS=0 \
    CASES="ar:1:1 tf:1:2 ar:8:3 tf:8:4 ar:16:5 tf:16:6" \
    benchmarks/tidar/run_lockstep_nvidia.sh
```

Run additional batches on free GPUs with the same settings. The wrappers omit
`--ignore-eos` when `IGNORE_EOS=0`. The pre-rendered dataset is intentionally
thinking-off, and `--force-bos` normalizes it to exactly one leading BOS.

A valid AMD run must report one forced BOS, AITER Flash Attention, AITER
unquantized MoE, both paged-attention causal signatures, and exact capture size
`17B`.

## Source Logs

- NVIDIA TF: `/data/home/jinzhao/nv_v2_tidar_logs/eos_iter12600_20260713/`
- NVIDIA AR: `/data/home/jinzhao/nv_v2_tidar_logs/eos_ar_iter12600_20260713/`
- AMD adaptive TF: `/shared/home/jinzhao/tfscope/amd_lockstep_steady_271217/`
- AMD b1 no-splits TF: `/shared/home/jinzhao/tfscope/amd_lockstep_steady_271218/`
- AMD AR: `/shared/home/jinzhao/tfscope/amd_lockstep_steady_271222/`
