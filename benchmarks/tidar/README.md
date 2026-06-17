# TiDAR SF throughput reproducer (AMD MI300X)

Reproduces the two headline single-forward (SF) numbers in
[`docs/amd_tidar_perf.md`](../../docs/amd_tidar_perf.md):

| Config | proposal levels | tok/s | accept |
|---|---|---:|---:|
| SF `[0,4,7,11]` (P=4, fastest) | `0,4,7,11` | ~803 | ~5.6 |
| SF dense `[0..16]` (P=17, max accept) | `0,1,2,...,16` | ~544 | ~7.6 |

The **only** difference between the two is `VLLM_TIDAR_PROPOSAL_ACC_LEVELS`.
Everything else is the shared matched config (ckpt `iter_0012600`, AIME25 30
chat-template prompts, n=4, T=0.5, max_tokens=8192, max_model_len=10000,
b=16, captured `FULL_DECODE_ONLY`, Triton CCA).

## Requirements

- An MI300X box (gfx942), ROCm 7.x.
- A vLLM build of this branch. Either build into the base image:
  ```bash
  docker run --rm -v $PWD:/w -w /w zyphra/rocm-primus:aiter_pa_swa \
    bash -c 'pip install "setuptools>=77,<81" setuptools_scm && pip install --no-build-isolation -e .'
  ```
  …or reuse a prebuilt image (commit the built container once to skip the
  ~5-min rebuild on every run).

## Run

`VLLM_ATTENTION_BACKEND=FLEX_ATTENTION` and `VLLM_TIDAR_SF_TRITON=1` are
required — SF only runs correctly on Flex, and the Triton paged kernel is
the AMD win (it skips the full-KV-cache reshape copy the generic Flex path
does). Set `CKPT=` if your checkpoint lives elsewhere.

```bash
# SF [0,4,7,11]  -> ~803 tok/s
docker run --rm --device /dev/dri --device /dev/kfd --group-add video \
  --network host --ipc host --shm-size 32G -v /shared:/shared -v $PWD:/vllm -w /vllm \
  -e HIP_VISIBLE_DEVICES=0 -e PYTHONUNBUFFERED=1 \
  -e VLLM_SKIP_SDPA_PREINIT=1 -e VLLM_ATTENTION_BACKEND=FLEX_ATTENTION \
  -e VLLM_TIDAR_SF_TRITON=1 -e VLLM_TIDAR_PROPOSAL_ACC_LEVELS=0,4,7,11 \
  <image> python -u benchmarks/tidar/bench_amd_sf.py

# SF dense [0..16] -> ~544 tok/s : same command, change one env var:
#   -e VLLM_TIDAR_PROPOSAL_ACC_LEVELS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
```

The script prints `TOTAL: <tok/s>`. For acceptance, the SpecDecoding metric
windows are in the log (`disable_log_stats=False`); true-mean accept =
`1 + Σaccepted / Σ(drafted/16)` over all windows.

## Notes / gotchas (from `docs/amd_tidar_perf.md`)

- Proposal levels **must include 0** — the codebase default `(4,7,10)`
  collapses acceptance to ~1.0 on this checkpoint.
- `VLLM_SKIP_SDPA_PREINIT=1` avoids an intermittent `import vllm` segfault
  on this stack.
- Keep Triton CCA (`VLLM_CCA_TRITON=1`, default); pytorch CCA is
  capture-unsafe on ROCm at b=16.
- Other AMD attention backends (AITER-FA, Triton, AITER-Unified) break
  TiDAR's K+1 spec layout — accept collapses. SF needs `FLEX_ATTENTION`.
