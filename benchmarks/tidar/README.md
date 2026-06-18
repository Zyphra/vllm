# TiDAR SF throughput reproducer (AMD MI300X)

Reproduces the two headline single-forward (SF) numbers in
[`docs/amd_tidar_perf.md`](../../docs/amd_tidar_perf.md). `bench_amd_sf.py` is
**a single self-contained file** — the 30 AIME25 chat-template prompts are
embedded in it, so you can copy just that one file into a built checkout. You
only supply (a) an MI300X box and (b) a smoediffusion checkpoint.

(`aime25_zpo_texts.json` is kept here only as the source of the embedded
prompts; the script does not read it.)

| Config | proposal levels | tok/s (doc) | accept |
|---|---|---:|---:|
| SF `[0,4,7,11]` (P=4, fastest) | `0,4,7,11` | ~803 | ~5.6 |
| SF dense `[0..16]` (P=17, max accept) | `0,1,...,16` | ~544 | ~7.6 |

`benchmarks/tidar/bench_amd_sf.py` **forces** every knob that selects the fast
code path (FLEX_ATTENTION + SF Triton + the proposal levels + captured), so
you cannot land on the slow path by forgetting an env var. The two configs
differ only by `--dense`.

## 1. Build the image (from this repo)

```bash
git checkout jinzhao/tidar_v016        # or this branch
docker run --rm -v "$PWD":/vllm -w /vllm zyphra/rocm-primus:aiter_pa_swa \
  bash -c 'pip install "setuptools>=77,<81" setuptools_scm && \
           pip install --no-build-isolation -e .'
# commit the result once so reruns skip the ~5-min rebuild:
#   docker commit <container> vllm-tidar:latest
```
If you can't pull `zyphra/rocm-primus:aiter_pa_swa`, any ROCm 7.x image with
torch 2.10 + aiter + a ROCm flash-attn works as the base.

## 2. Run

```bash
# SF [0,4,7,11]  -> ~760-800 tok/s
docker run --rm --device /dev/dri --device /dev/kfd --group-add video \
  --network host --ipc host --shm-size 32G \
  -v /path/to/checkpoints:/ckpts -v "$PWD":/vllm -w /vllm \
  -e HIP_VISIBLE_DEVICES=0 -e PYTHONUNBUFFERED=1 \
  <image> python -u benchmarks/tidar/bench_amd_sf.py --ckpt /ckpts/iter_0012600

# SF dense [0..16] -> ~510-545 tok/s : add --dense
#   ... python -u benchmarks/tidar/bench_amd_sf.py --ckpt /ckpts/iter_0012600 --dense
```

The script prints its forced config, then `TOTAL: <tok/s>`. You do **not**
set any `VLLM_*` env vars yourself — the script sets them before importing
vllm. Run **one config per node** (b=16/8192-token co-runs contend ~6-7%).

## 3. Confirm you're on the fast path

If you previously saw ~10% of the expected throughput, you were not on the
SF Triton path. With this script that can't happen silently — but to confirm,
check the engine log near startup for:

```
Using FlexAttention backend
TiDAR single-forward mode ENABLED with K=16, P=4, acc_levels=(0, 4, 7, 11)
```

and that the `SpecDecoding metrics` per-position acceptance is ~0.8 decaying
to ~0.2 (a flat ~1.0 mean = collapsed → wrong levels). The script also prints
a loud warning if throughput is < 150 tok/s.

True-mean accept (the metric in the doc) =
`1 + Σaccepted / Σ(drafted/16)` over all SpecDecoding windows in the log.

## Why the slow path happens (the three traps)

1. **Backend must be `FLEX_ATTENTION`.** Other AMD backends (AITER-FA,
   Triton, AITER-Unified) break TiDAR's K+1 spec layout — accept collapses
   to ~1.2 and you get AR-ish speed. SF only runs correctly on Flex.
2. **Proposal levels must include 0.** The codebase default `(4,7,10)`
   is cascade-degenerate on this checkpoint (accept ≈ 1.0).
3. **Must be captured** (`cudagraph_mode=FULL_DECODE_ONLY`), not eager.

Plus on ROCm: keep `VLLM_CCA_TRITON=1` (pytorch CCA is capture-unsafe at
b=16) and `VLLM_SKIP_SDPA_PREINIT=1` (avoids an intermittent import segfault).
The script sets all of these.

## Validation (2026-06-17, cnode-2 GPU5, solo, fresh build at tip `67a63fc7d`)

| Config | doc | measured | accept (doc / measured) |
|---|---:|---:|---|
| SF `[0,4,7,11]` | 803 | **762** | 5.63 / **5.53** |
| SF dense `[0..16]` | 544 | **509** | 7.57 / **7.26** |

Acceptance matches the doc → the config is faithful. The ~5-6% throughput
gap is build-image variance (doc used prebuilt `jinzhao/vllm-tidar-amd:latest`
@ `3f1a680f2`; this was a fresh source build @ `67a63fc7d`).
