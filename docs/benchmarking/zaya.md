# Zaya performance branch

`rob/zaya-v26-fused` is a reviewable Zaya-8B/74B inference branch based on
vLLM v0.26.0 and the exact official Zaya PR history.

## Source and scope

| Role | Revision |
|---|---|
| vLLM release | `vllm-project/vllm:releases/v0.26.0@568afb3a1` |
| Official Zaya PR | `Zyphra/vllm:zaya1-pr@b1b99a08b` |
| Integration merge | `142009e35` |
| Qualified Zyphra-kernels API | `Zyphra/zyphra_kernels:rob/zaya-v26-fused@5727de4b6` |

There is one runtime model: `ZayaForCausalLM`. Zaya-8B checkpoints already use
its 40 combined attention+MoE blocks. Legacy Zaya-74B checkpoints use 120
alternating attention/MoE layers; convert them offline into 60 combined blocks:

```bash
python tools/model_converters/convert_smoe_to_zaya.py \
  <legacy-74b-checkpoint> <converted-checkpoint>
```

The converter validates the complete key map before reading tensors, stacks
the per-expert weights into the official Zaya layout, and writes a standard
`ZayaForCausalLM` checkpoint. A metadata-only preflight is also available:

```bash
python tools/model_converters/convert_smoe_to_zaya.py \
  <legacy-74b-checkpoint> --validate-only
```

The branch adds two runtime features after the official model:

1. A generic post-load hook for non-persistent derived weight views.
2. A default-off adapter for Zyphra-kernels fused CCA prefill and decode+RoPE.

The adapter fails closed if either operator or the requested shape is
unavailable. Both operators share one cached grouped-weight layout, refreshed
after initial load and live weight updates. Temperature remains a canonical
FP32 parameter; no per-step copy is maintained.

The prefill adapter uses the official model's compact projected recurrent
state. It asks the kernel to skip the legacy raw-hidden shift while retaining
the fused Q/K convolution, mean, normalization, temperature, and conv-state
update.

## Validation

The legacy 74B production checkpoint has `4,683` tensors. The converter
preflight consumes all of them, rejects duplicate targets, and verifies all
24 experts for both projections in every one of the 60 combined blocks.
Conversion job `330111` rewrote it to `1,923` tensors in four minutes;
validation job `330129` passed key/header coverage and bitwise checks of
direct residual mappings plus first/last-layer expert slices. The converted
config loads as `ZayaForCausalLM`, 60 layers, hidden size 4096, 24 experts.

Exact-branch server smokes then exercised the same sole runtime class:

- Job `330175` loaded Zaya-8B DP1, selected AITER and fused CCA+RoPE,
  captured four FULL decode graphs, and returned a finite completion. Its
  wrapper alone failed on an obsolete log-string assertion.
- Job `330179` loaded the converted Zaya-74B checkpoint in DP8/EP8, selected
  AITER and fused CCA+RoPE, captured four FULL decode graphs, returned a finite
  completion, and completed `0:0`.

The fused adapter and refresh lifecycle passed focused tests on MI300X,
covering Q8/GQA4 and Q16/GQA8, BF16 I/O with FP32 convolution state,
caller-owned output, exactly-once RoPE, initial load, and live weight updates.

A matched Zaya-8B DP1 directional run on `cnode-184` used the same vLLM
source, image, model, AITER backend, and 128-token workload:

| Path | Job | Post-first tok/s | Tokens |
|---|---:|---:|---:|
| Official CCA | `330850` | 66.87 | 128/128 |
| Fused prefill + decode/RoPE | `330865` | 126.22 | 128/128 |

Job `330865` logged both fused paths, captured FULL decode graphs, and
returned 543 bytes of nonempty text. This is a single-run directional result;
concurrency and counterbalanced promotion measurements remain open.

## Run

Install the qualified Zyphra-kernels revision, then enable the path explicitly:

```bash
git clone --branch rob/zaya-v26-fused \
  https://github.com/Zyphra/zyphra_kernels.git
git -C zyphra_kernels checkout 5727de4b6
ZK_BUILD_FAMILIES=cca ZK_PYTORCH_ROCM_ARCH=gfx942 \
  uv pip install -e ./zyphra_kernels --no-build-isolation
```

```bash
export VLLM_CCA_ZK_DECODE=1
vllm serve <zaya-checkpoint> \
  --dtype bfloat16 \
  --mamba-cache-dtype float32 \
  --compilation-config.cudagraph_mode FULL_AND_PIECEWISE
```

The explicit cache dtype preserves recurrent CCA state in FP32 while the
operator keeps BF16 projections and outputs; no per-step state cast is added.

Required receipts are:

- `Using Zyphra fused CCA decode and RoPE`;
- `Using Zyphra fused CCA prefill` on prefill-only steps;
- FULL graph replay;
- finite output;
- the intended Zaya-8B DP1 or Zaya-74B EP8 topology.

With `VLLM_CCA_ZK_DECODE=0` (the default), the official source path is
unchanged. The model and converter are device-neutral. The optional fused CCA
operator is currently qualified only on AMD MI300X/gfx942.
