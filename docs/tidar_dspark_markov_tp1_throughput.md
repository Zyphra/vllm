# TiDAR DSpark+Markov TP1 throughput

This report reproduces the 32K live-training workload on one NVIDIA H100,
without data parallelism, using the V2 TiDAR runner on
`jinzhao/tidar_v024`. It is intentionally separate from the AMD/NVIDIA
microbenchmark report because the workload, model, and output-length
distribution are different.

## Result

The run completed all 128 requests without an error, OOM, or
preemption/recompute warning.

| Phase | Duration | Output tok/s | Mean accept length | Draft accept | Running / waiting | Mean KV |
|---|---:|---:|---:|---:|---:|---:|
| Refill-controlled | 46m 20s | **1,074.2** | 8.26 | 45.36% | 15.74 / 16.24 | 94.97% |
| Terminal drain | 24m 10s | 488.4 | 7.44 | 40.22% | 7.69 / 1.71 | 49.76% |
| Full finite batch | 70m 30.7s | **873.6** | **8.09** | **44.30%** | 12.98 / 11.26 | 79.47% |

The refill-controlled phase ends at the last ten-second telemetry window with
32 outstanding requests. Its engine throughput distribution is 628.2 / 1,010.9
/ 1,626.7 tok/s at p10 / median / p90, with a maximum ten-second window of
1,971.5 tok/s. The terminal drain begins after request 96 completes, when the
client can no longer refill to concurrency 32.

The 24-minute drain consumes 34% of total wall time while producing only about
19% of the output tokens. This is why the complete finite-batch result is 19%
below the refill-controlled result even though both come from the same server
run.

### Output distribution

| Metric | Value |
|---|---:|
| Total output tokens | 3,695,937 |
| Mean output length | 28,874.5 |
| p10 / median / maximum | 12,767.7 / 32,768 / 32,768 |
| Reached 32,768-token cap | 102 / 128 (79.7%) |
| Stopped on EOS before cap | 26 / 128 (20.3%) |

This deterministic sample is longer than the two cited live steps, whose mean
responses were 21,997 and 24,503 tokens. It is therefore a harsher KV and drain
case than those live captures, despite using the same source dataset and
sampling policy.

### Latency

| Metric | Median | Mean | p99 |
|---|---:|---:|---:|
| TTFT | 9.39 s | 59.92 s | 388.82 s |
| TPOT | 23.74 ms | 29.65 ms | 111.22 ms |
| End-to-end latency | 795.48 s | 801.95 s | 1,623.51 s |

The high TTFT tail comes from KV admission: at the requested 0.55 GPU-memory
utilization, long-context plateaus commonly had only 9-16 running requests and
the rest waiting. Refill bursts temporarily restored all 32 running requests
and 1.4-1.9K tok/s; context growth then reduced the runnable batch again.

## What was run

| Item | Value |
|---|---|
| Host / GPU | `vp-dgx-85`, NVIDIA H100 80 GB, GPU 0 |
| vLLM | fork version `0.24.0`, V2 GPU model runner |
| Parallelism | TP1, PP1, **DP1** |
| Checkpoint | `SMOE_DIFFUSION_sftv2_tv_dspark+markov-hf/iter_0005600` |
| Precision | BF16 |
| TiDAR | two-forward, K16, async scheduling |
| Draft | DSpark head + Markov bias, global-position reset |
| Draft sampling | `tidar_diff_temperature=0.0` (argmax after Markov bias) |
| Target sampling | temperature 0.6, top-p 1.0, top-k -1 |
| EOS | enabled |
| Maximum output | 32,768 tokens |
| Maximum model length | 33,824 tokens |
| Concurrency | 32 HTTP requests and `max_num_seqs=32` |
| KV allocation | `gpu_memory_utilization=0.55` |
| Graph mode | `FULL_DECODE_ONLY`, target and TiDAR draft graphs captured |
| Attention / CCA / MoE | FlashAttention 3 / Triton / Triton |
| Extra output work | chosen-token logprobs, AR verifier logprobs, routed experts |

The engine startup log confirms `tensor_parallel_size=1` and
`data_parallel_size=1`; no data-parallel server flags were used.
The model occupied 17.92 GiB, captured graphs occupied 1.51 GiB, and the engine
reported 8.46 GiB of available KV memory and a 110,848-token GPU KV cache.

## Workload

The source is the live-training parquet:

```text
/data/datasets/zpo/training_datasets/zpo_math_rsa_only_geq0_no_skywork_no_dapo_no_explicit_images_puzzle_math_cont_step_116_instruct_with_pass_rate_filtered_geq_0.25.parquet
```

The deterministic workload uses NumPy seed 0, 16 prompt groups, and eight
responses per prompt, for 128 responses total. Prompts are rendered with the
checkpoint chat template and `enable_thinking=True`, contain exactly one BOS,
and are interleaved by response index so refills retain a balanced prompt mix.
Prompt lengths are 69 minimum, 124.7 mean, and 359 maximum tokens. Seven
prompt groups are AceReason and nine are DeepMath.

Unlike a fixed-output synthetic test, EOS remains enabled. This preserves the
live response-length distribution and makes it necessary to report both the
saturated interval and the full finite-batch result.

## V2 support added for this run

The checkpoint already carried DSpark and Markov weights, but the V2 TiDAR
speculator was still drafting through the ordinary LM head. The V2 path now:

1. Computes DSpark logits with the untied diffusion output head.
2. Applies the sequential Markov bias at each draft position.
3. Resets Markov state on the true global block grid and otherwise chains from
   the preceding committed or drafted token.
4. Uses argmax because the requested draft temperature is zero.

The V2 runner also lacked the routed-expert capturer lifecycle required by
`--enable-return-routed-experts`. It now creates and binds the sidecar after KV
cache sizing, saves target-pass routes in the scheduler's selected dense KV
group, and disables capture during the self-draft forward.

The benchmark ran before remote commit `efa5c1435` landed its dense-group
anti-aliasing fix. A live shared-memory inspection confirmed that capture and
copy overhead were active, so the throughput result remains representative;
however, the raw routed-expert payload from this run should not be used for
correctness validation. The final branch applies `efa5c1435` to V2 as well as
the old runner.

## Reproducer

The checked-in entry point contains the complete TP1 server and client flags:

```bash
export MODEL=/data/checkpoints/SMOE_DIFFUSION_sftv2_tv_dspark+markov-hf/iter_0005600
export DATASET=/data/datasets/zpo/training_datasets/zpo_math_rsa_only_geq0_no_skywork_no_dapo_no_explicit_images_puzzle_math_cont_step_116_instruct_with_pass_rate_filtered_geq_0.25.parquet
export WORKLOAD=/data/home/jinzhao/tidar_live_tp1_20260713/workload_16x8.jsonl

python3 benchmarks/tidar/build_live_workload.py \
  --dataset "$DATASET" \
  --model "$MODEL" \
  --output "$WORKLOAD"

# Terminal 1
benchmarks/tidar/run_live_tp1_dspark.sh serve

# Terminal 2, after /health is ready
benchmarks/tidar/run_live_tp1_dspark.sh bench
```

The benchmark client sends 128 requests at infinite request rate with a
32-request concurrency cap. `/metrics` is sampled every ten seconds so fill,
KV saturation, refill, and terminal drain can be analyzed separately.

Raw artifacts from this run are retained on `vp-dgx-85` under:

```text
/data/home/jinzhao/tidar_live_tp1_20260713/
```

The important files are `result.json`, `server.log`, `client.log`,
`metrics.log`, `workload_16x8.jsonl`, and `workload_manifest.json`.

## Reference comparison

The supplied live reference used DP8 with 32 requests distributed across eight
rollout engines: 1,111.3 generated tok/s aggregate, 138.9 tok/s per GPU, and
983.3 tok/s end-to-end aggregate goodput. This TP1 run instead concentrates all
32 requests on one GPU. It therefore tests single-engine saturation, not a
drop-in per-GPU comparison with the roughly four-request live engines.

The refill-controlled TP1 result is only 3.3% below the live DP8 aggregate raw
throughput, while the full finite-batch result is 21.4% below it. The latter gap
is primarily terminal drain. Compared with the live per-GPU raw number, TP1 is
7.7x faster during refill because it executes a much larger per-GPU batch; it
also runs at 95% mean KV instead of the live capture's 38%.

Acceptance is materially better here: 8.09 tokens/block overall versus 4.815
in the supplied live capture. That does not by itself identify a rejection
sampler defect. This run explicitly uses the DSpark output head and global-grid
Markov state in the V2 path, while the supplied live result came from a
different runner/version and a different prompt sample.

## Takeaways

1. The requested no-DP V2 DSpark+Markov configuration works end to end, while
   retaining async scheduling, captured target/draft forwards, logprobs, and
   routed-expert output.
2. One H100 can nearly match the reference DP8 aggregate while 32 requests are
   continuously supplied, but it does so at near-full KV and substantial
   scheduler waiting.
3. The 873.6 tok/s finite-batch figure is not the engine saturation limit. For
   throughput comparisons, use the 1,074.2 tok/s refill-controlled result.
4. For this exact 32K workload, raising the 0.55 memory allocation or reducing
   per-engine concurrency would trade KV waiting against compute batch size.
   That is a separate tuning experiment, not part of this exact-config run.
