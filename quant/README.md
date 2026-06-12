# ZAYA1-8B INT8 (W8A8) quantization

Offline, GPU-free INT8 quantization of ZAYA1-8B for faster RSA serving on the
RX 7900 (gfx1100). Part of the performance plan
(`/home/pat/.claude/plans/to-align-with-the-cryptic-fairy.md`, Tier 2).

## Why a custom quantizer

ZAYA1 ships only `config.json` + safetensors — **no `transformers` modeling
code** (`architectures: ZayaForCausalLM`, no `auto_map`). vLLM supplies the
modeling code, but `llm-compressor` loads models through
`AutoModelForCausalLM.from_pretrained`, which has no `ZayaForCausalLM` class to
instantiate. So the standard calibrate-on-GPU flow can't run here.

`quantize_int8.py` instead rewrites the weight tensors **directly in the
safetensors**, emitting a `compressed-tensors` checkpoint vLLM loads natively.
No model instantiation, no calibration, no GPU — so it does not touch the shared
inference container.

## What it does (v1)

- Quantizes the **MoE experts** (`zaya_block.experts.local_experts.*.linear_fc1`
  and `linear_fc2`) — the dominant share of the 8B weights — to per-output-channel
  symmetric **INT8**, storing a `weight_scale` per row.
- Declares **dynamic per-token INT8 activations** in the config (quantized at
  runtime, nothing stored).
- Leaves everything else in bf16: attention/CCA projections, the MoE **router**
  (accuracy-sensitive top-1 routing), norms, and the tied **embedding / lm_head**.

The scheme matches vLLM's `CompressedTensorsW8A8Int8` (Linear) and
`CompressedTensorsW8A8Int8MoEMethod` (experts). On gfx1100 both run through
**Triton** kernels (`triton_scaled_mm`, `TritonExperts`) over native RDNA3 WMMA
int8 — a memory *and* compute win. Measured single-tensor dequant error is
~0.9% (per-channel), so accuracy impact should be small, but it **must** be
validated (see runbook).

## Run it

```bash
# CPU-only venv (kept separate from the rsa client venv):
uv venv --python 3.12 .venv-quant
uv pip install -p .venv-quant/bin/python --torch-backend=cpu torch safetensors numpy

SNAP=$(find ~/.cache/huggingface/hub/models--Zyphra--ZAYA1-8B/snapshots \
        -maxdepth 1 -mindepth 1 -type d | head -1)
.venv-quant/bin/python -m quant.quantize_int8 --src "$SNAP" --dst ~/models/ZAYA1-8B-int8
```

Output: `~/models/ZAYA1-8B-int8/` (int8 safetensors + index + rewritten
`config.json` with `quantization_config` + tokenizer files).

## Deploy + validate — **needs a coordination window**

The single GPU can't host a second instance next to the live bf16 server, so
deploy/benchmark only when the other agent has paused RSA testing. A staged
compose override (`quant/docker-compose.int8.yml`) makes the swap fast.

1. **Coordinate & snapshot.** Confirm RSA testing is paused; record the current
   `docker-compose.yml` serve flags so the bf16 config can be restored.
2. **Baseline (bf16).** From a quiet server, capture numbers with the harness:
   `.venv/bin/python -m bench.run --prometheus http://localhost:9090 \
   --rsa-n 16 --rsa-k 4 --rsa-t 2 --repeat 3 --out bf16.json`
3. **Deploy INT8.**
   `docker compose -f docker-compose.yml -f quant/docker-compose.int8.yml up -d vllm`
   Confirm `/health`; watch startup logs for the compressed-tensors INT8 scheme
   being applied to the experts (and for any expert-scale loader errors — the
   MoE scale-mapping in `vllm/model_executor/models/zaya.py` is the most likely
   place to need a fix, since this is a custom-arch loader path).
4. **Measure INT8.** Re-run the harness `--out int8.json`. Compare `latency`,
   `accuracy`, and `server.peak_running_seqs` against `bf16.json`. With freed
   VRAM, raise `--max-num-seqs` toward 16 so a round of rollouts runs as one
   wave.
5. **Accuracy gate.** Run the AIME-style set (`bench.run --questions aime.jsonl`)
   on both; require no regression to adopt INT8.
6. **Restore or keep.** Either revert to bf16 or hand the INT8 config to the
   other agent, and confirm the server is healthy before releasing the window.

## Follow-ups (future iterations)

- Extend quantization to the attention `o_proj` (standard Linear, low risk) once
  the MoE path is validated.
- Evaluate INT8 on the CCA input projections only if accuracy holds — the CCA
  state path is precision-sensitive (the report keeps mamba state in fp32).
- If per-channel RTN loses accuracy on AIME, a GPTQ pass (needs a GPU window +
  calibration data) is the next lever.
