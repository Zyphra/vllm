# AMD/NVIDIA TiDAR Throughput: `iter_0012600`

_Last updated: 2026-07-10. Audience: AMD/vLLM performance engineers. Repo:
`Zyphra/vllm-smoe-amd`, branch `jinzhao/tidar_v024`._

This is the focused performance handoff for TiDAR two-forward (TF) on AMD
MI300X. Every primary result and reproducer in this note uses checkpoint
`iter_0012600`. Prompts are passed as token IDs with exactly one leading BOS,
independent of the checkpoint's chat template.

TiDAR TF is functionally working on the V2 GPU runner with async scheduling,
cudagraph replay, ROCm AITER Flash Attention, ROCm AITER MoE, and per-pass
causal attention. AMD acceptance is healthy at b8-b64, but TF throughput still
trails AR there. The remaining gap is the cost of the two expanded ROCm model
forwards, especially at high batch.

## Reference Configuration

| Setting | Value |
|---|---|
| Checkpoint | `smoediffusion_128k_64node/iter_0012600` |
| Dataset | `benchmarks/tidar/aime25_zpo_texts.json` (AIME25 thinking-off) |
| Prompt path | token IDs, `add_special_tokens=False`, force exactly one BOS |
| Expected first IDs | `[2, 105, 9731, 107]`, `leading_bos_count=1` |
| Runner | V2, `VLLM_USE_V2_MODEL_RUNNER=1`, async scheduling |
| Capture | `FULL_AND_PIECEWISE` |
| Generation | MT5000, warmup 64, ignore EOS, seed 0 |
| TiDAR | K=16, target/draft temperature `0.0` |
| TF workload | `num_prompts=bsz`, `n_sample=1` |
| AR workload | `num_prompts=bsz`, `n_sample=1` |
| NVIDIA | H100, `FLASH_ATTN` v3 |
| AMD | MI300X, `ROCM_AITER_FA`, AITER MoE, TF paged attention, no-splits |

The AR and TF tests have the same active batch, prompt set, MT5000 length,
request count, sampling, runner, and capture mode. This makes the TF/AR ratios
direct speedup measurements for a fixed, pure-decode workload.

### BOS Control

The checkpoint chat templates differ: the older comparison checkpoint's
template prepends BOS while `iter_0012600`'s template does not. These tests do
not call `apply_chat_template`. The probes tokenize with
`add_special_tokens=False`, remove any duplicate leading BOS IDs, and prepend
exactly one BOS when `--force-bos` is set. This makes the input path explicit
and idempotent even though the included AIME25 file already starts with BOS.

Always verify the emitted context contains:

```text
"force_bos": true
"leading4": [2, 105, 9731, 107]
"leading_bos_count": 1
```

## MT4000 at Target Temperature 0.6

Latest matched sampling run: MT4000, `n_sample=1`, target temperature `0.6`,
and argmax draft (`tidar_diff_temperature=0`). NVIDIA b1/b8 use three-seed
medians; NVIDIA b16/b64 and all AMD rows are seed 0.
Both platforms use the `iter_0012600` weight shards plus the metadata variant
matching `iter_0012000` (`residual_in_fp32=false`,
`mamba_cache_dtype=float32`).

| bsz | NVIDIA AR | NVIDIA TF / acc | TF/AR | AMD AR | AMD TF / acc | TF/AR | AMD/NVIDIA AR | AMD/NVIDIA TF |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `82.258` | `132.427` / `4.619` | `1.61x` | `60.702` | `60.753` / `4.209` | `1.00x` | `0.74x` | `0.46x` |
| 8 | `542.912` | `680.086` / `5.317` | `1.25x` | `377.326` | `314.923` / `5.422` | `0.83x` | `0.70x` | `0.46x` |
| 16 | `975.975` | `1333.164` / `5.263` | `1.37x` | `773.367` | `587.371` / `5.444` | `0.76x` | `0.79x` | `0.44x` |
| 64 | `3141.139` | `4043.730` / `5.942` | `1.29x` | `2532.184` | `1788.693` / `5.002` | `0.71x` | `0.81x` | `0.44x` |

The AMD/NVIDIA AR ratios are `0.74/0.70/0.79/0.81x`; TF ratios are
`0.46/0.46/0.44/0.44x`. Acceptance is broadly comparable, but AMD TF remains
slower than AMD AR at b8/b16/b64. NVIDIA median TF is faster than AR at b1/b8.

Run this variant with:

```bash
MAX_TOKENS=4000 MAX_MODEL_LEN=12000 N_SAMPLE_AR=1 N_SAMPLE_TF=1 \
TARGET_TEMP=0.6 DRAFT_TEMP=0 \
    bash benchmarks/tidar/run_iter12600_tput.sh
```

Logs:

- NVIDIA: `/data/home/jinzhao/nv_v2_tidar_logs/iter12600_iter12000meta_mt4k_t06_d0_n1_vp16_20260709_190123/`
- AMD: `/shared/home/jinzhao/tfscope/amd_iter12600_iter12000meta_mt4k_t06_d0_n1_c17_20260710_010137/`

## MT5000 Greedy Throughput

Throughput is output tokens/second. Mean acceptance includes TiDAR's normal
`+1` sampled token.

| bsz | NVIDIA AR | NVIDIA TF / acc | NVIDIA TF/AR | AMD AR | AMD TF / acc | AMD TF/AR | AMD TF / NVIDIA TF |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `79.723` | `207.554` / `7.562` | `2.60x` | `53.828` | `62.296` / `4.703` | `1.16x` | `0.30x` |
| 8 | `539.015` | `972.872` / `6.136` | `1.80x` | `382.078` | `375.808` / `6.746` | `0.98x` | `0.39x` |
| 16 | `968.246` | `1603.001` / `6.802` | `1.66x` | `772.428` | `661.769` / `8.325` | `0.86x` | `0.41x` |
| 64 | `3093.504` | `3482.665` / `6.936` | `1.13x` | `2509.826` | `2102.265` / `8.250` | `0.84x` | `0.60x` |

Platform parity for the same mode:

| bsz | AMD/NVIDIA AR | AMD/NVIDIA TF | AMD-NVIDIA TF accept delta |
|---:|---:|---:|---:|
| 1 | `0.68x` | `0.30x` | `-2.859` |
| 8 | `0.71x` | `0.39x` | `+0.610` |
| 16 | `0.80x` | `0.41x` | `+1.524` |
| 64 | `0.81x` | `0.60x` | `+1.314` |

### Takeaways

- AMD AR reaches `0.68-0.81x` of NVIDIA AR, while AMD TF reaches only
  `0.30-0.60x` of NVIDIA TF.
- AMD TF is slightly slower than AMD AR at b8 and materially slower at b16/b64,
  while NVIDIA TF remains faster than NVIDIA AR at every batch.
- AMD acceptance is higher than NVIDIA at b8/b16/b64, so the high-batch
  throughput gap is not caused by rejection quality. The b1 row is one prompt;
  platform-specific numeric trajectory divergence makes its acceptance noisy.
- The matched b16 result is especially diagnostic: AMD accepts `8.33` tokens
  versus NVIDIA's `6.80`, yet delivers only `0.41x` NVIDIA TF throughput.
- The b64 graph-cap fix is active: the captured TiDAR shape includes
  `64 * (K + 1) = 1088` tokens.

## Source Logs

- NVIDIA: `/data/home/jinzhao/nv_v2_tidar_logs/iter12600_forcebos_mt5k_n1_vp16_20260709_175900/`
- AMD: `/shared/home/jinzhao/tfscope/amd_iter12600_forcebos_mt5k_n1_c17_20260710_002037/`

The source logs report no probe errors. All AR and TF runs reached exactly
5,000 output tokens per request and logged one leading BOS. Each b64 mode
generated 320,000 output tokens.

## Matched b16 Profile

This shorter MT512 profile uses the same checkpoint, prompt-token path,
backend, capture mode, and K=16, with `n_sample=1`:

| Platform | Throughput / acc | Target forward | Draft forward | Reject sampler |
|---|---:|---:|---:|---:|
| NVIDIA H100, `FLASH_ATTN` v3 | `1394.609` / `6.307` | `20.067 ms` | `17.579 ms` | `0.989 ms` |
| AMD MI300X, `ROCM_AITER_FA` | `1197.474` / `5.796` | `28.821 ms` | `26.314 ms` | `1.175 ms` |

The sampler difference is small. The AMD target and draft model forwards are
about `44%` and `50%` slower in this profile, which points optimization work
inside the captured model graphs rather than at rejection sampling.

Profile logs:

- NVIDIA: `/data/home/jinzhao/nv_v2_tidar_logs/iter12600_profile/20260707_014246_nv_gpu6_iter12600_fp_profile_b16_mt512.log`
- AMD: `/shared/home/jinzhao/tfscope/amd_iter12600_profile/20260707_081033_amd_cnode107_gpu4_iter12600_fp_profile_b16_mt512_warmimg.log`

## AMD Reproducer

The repo now contains both probes and a matched runner:

```text
benchmarks/tidar/probe_v2_ar.py
benchmarks/tidar/probe_v2_tidar_nv.py
benchmarks/tidar/run_iter12600_tput.sh
benchmarks/tidar/aime25_zpo_texts.json
```

On an AMD node, build the repo in the standard ROCm image, then run:

```bash
cd /work
pip install -q --no-build-isolation -e .

export CKPT=/shared/home/henry/checkpoints/hf/smoediffusion_128k_64node/iter_0012600
export DATA=benchmarks/tidar/aime25_zpo_texts.json
export BACKEND=ROCM_AITER_FA
export GPU=1
export BATCHES="1 8 16 64"
export MAX_TOKENS=5000
export MAX_MODEL_LEN=12000
export N_SAMPLE_AR=1
export N_SAMPLE_TF=1
export LOGROOT=/shared/home/$USER/tfscope/iter12600_forcebos_$(date +%Y%m%d_%H%M%S)

bash benchmarks/tidar/run_iter12600_tput.sh
```

To run only a quick validation before the full sweep:

```bash
BATCHES="1 16" bash benchmarks/tidar/run_iter12600_tput.sh
```

The runner selects the production AMD path:

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

For NVIDIA, use the same script with:

```bash
BACKEND=FLASH_ATTN VLLM_FLASH_ATTN_VERSION=3 GPU=0 \
    bash benchmarks/tidar/run_iter12600_tput.sh
```

Expected AMD ballpark from the reference sweep:

```text
b1:  AR 53.8 tok/s, TF 62.3 tok/s, accept 4.70 tokens
b8:  AR 382.1 tok/s, TF 375.8 tok/s, accept 6.75 tokens
b16: AR 772.4 tok/s, TF 661.8 tok/s, accept 8.33 tokens
b64: AR 2509.8 tok/s, TF 2102.3 tok/s, accept 8.25 tokens
```

Each log's `PATCH_PROBE_CONTEXT` must show exactly one BOS before its throughput
result is accepted.

## Required Log Signatures

AMD TF logs should contain:

```text
Using Aiter Flash Attention backend.
Using ROCm AITER backend for Unquantized MoE
Using TiDAR TF paged attention in ROCm AITER-FA (S=17, causal=True).
Using TiDAR TF paged attention in ROCm AITER-FA (S=17, causal=False).
TiDAR detected: setting cudagraph_capture_sizes ... 1088
```

If acceptance falls near one or text becomes incoherent, first verify the BOS
context and both paged-attention causal modes. If those are correct, compare
target/draft forward time before changing the rejection sampler.

## Optimization Priorities

1. Tune AITER's unquantized E=16, H=2048, top-1 MoE kernels at the TiDAR graph
   shapes; the current path reports its generic two-stage default.
2. Profile causal verify and bidirectional draft AITER FA separately at b8,
   b16, and the full b64 `1088` shape.
3. Build a TiDAR-aware fused HIP CCA uniform-K+1 path that preserves every
   candidate-state stash and separate drafter scratch writes.
4. Recheck full-batch steady-state windows separately from final-tail windows.
5. Keep acceptance, prompt IDs, checkpoint, and AR baselines fixed while
   evaluating kernel changes.
