# TiDAR DSpark+Markov TP1 throughput

This report reproduces the 32K live-training workload on one NVIDIA H100,
without data parallelism, using the V2 TiDAR runner on
`jinzhao/tidar_v024`. It is intentionally separate from the AMD/NVIDIA
microbenchmark report because the workload, model, and output-length
distribution are different.

## Result

### Stochastic V2 production follow-up

A later no-DP H100 run used stochastic DSpark and target sampling
(`T_d=T_AR=0.6`) with real rejection sampling. This is the current production
path and supersedes the earlier V2 token-equality approximation for stochastic
drafts.

The matched benchmark uses `iter_0005600`, 128 prompts, concurrency 32, a
4,096-token client cap, natural EOS, `gpu_memory_utilization=0.75`, K16,
`FULL_AND_PIECEWISE`, async V2, BF16, `logprobs=1`, routed-expert output, and
no data parallelism on one H100.

| V2 implementation | Output tok/s | Mean accept, +1 | Duration | Output tokens |
|---|---:|---:|---:|---:|
| Token match, old draft sampler | 1,831.58 | 4.435 | 284.52 s | 521,119 |
| Token match, fused draft sampler | 1,914.92 | 4.452 | 271.41 s | 519,726 |
| Exact rejection, fused draft sampler | 2,256.72 | 5.247 | 231.11 s | 521,544 |
| Exact rejection + fused top-1 logprobs | **2,320.01** | 5.109 | 225.76 s | 523,770 |

Exact rejection raises throughput by 17.9% over the fused token-match path.
The top-1 logprob kernel adds another 2.8%; its verified request-block rate
rises from 430.5 to 454.6 blocks/s. From the old draft sampler to the current
path, aggregate throughput improves by 26.7%.

The acceptance improvement is a correctness fix, not a tuning trick. V2 had
been computing compact proposal state but discarding it, then accepting only
when a target Gumbel sample matched the draft token. It now persists proposal
state by stable request slot, accepts with `min(1, p(x)/q(x))`, and samples the
first rejected token from `max(p-q, 0)`.

The updated profile measures the fused top-1 logprob kernel at 0.520 ms per
B32 TF iteration versus 2.495 ms for the previous top-k/log-softmax/rank path.
Residual recovery is only 0.117 ms per iteration and is not a current
bottleneck. FP32 Gumbel noise was also tested as an opt-in: it reduced sampler
microkernel time but did not improve aggregate throughput (`2,314.39 tok/s`),
so production retains the default FP64-noise behavior.

Raw H100 artifacts are under:

```text
/tmp/jinzhao_tidar_075_ab_20260714/
/tmp/jinzhao_tidar_exact_rs_20260714/
/tmp/jinzhao_tidar_top1_20260714/
/tmp/jinzhao_tidar_top1_profile_20260714/
```

Relevant commits are `80147f94b` (fused stochastic draft sampling),
`825a38dfc` (exact V2 rejection), and `51a2de6c6` (fused top-1 logprobs).

### Latest verifier and drafter hot-path optimization

The next optimization pass removed work that was still scaling with K or the
full vocabulary even though exact rejection did not need it:

1. The verifier now Gumbel-samples only one bonus row per request instead of
   all K+1 rows. Draft-row target probabilities and log-normalizers still use
   every required row.
2. One Triton reduction computes target `p(draft)`, target log-normalizers,
   and deferred raw/processed top-1 statistics. Deferred `logprobs=1` then
   scans logits only for rank.
3. Draft logits remain in proposal-batch order. A small stable-request-to-row
   map replaces the per-step copy into a `[max_reqs,K,V]` cache.
4. The DSpark loop no longer materializes a separate FP32 `[B,V]` tensor for
   each of 16 positions. It also reuses row, temperature, and Markov
   workspaces and stacks compact outputs once.

Focused H100 tests matched the prior all-row Gumbel samples on bonus rows,
matched target probabilities/log-normalizers and top-1 logprobs, exercised
request reordering in exact rejection, and passed the optimized DSpark loop.
The full server then passed every draft-graph size used during a natural-EOS
run without an error, preemption, or recomputation.

The same 32K production contract was rerun after these changes. For the most
robust saturated comparison, strict B32 includes only telemetry intervals
whose two endpoints both report 32 running requests. The near-full window is
an interpolated 180-second window with 31-32 running, zero waiting, and mean
occupancy 31.84; the baseline's equivalent window remained at 32 throughout.

| Saturated metric | Pre-optimization | Latest | Change |
|---|---:|---:|---:|
| Strict B32 output tok/s | 2,686.47 | **2,843.54** | **+5.85%** |
| Strict B32 mean accept, +1 | 7.194 | 7.065 | -0.129 |
| Near-full 180-second tok/s | 3,077.13 | **3,372.00** | **+9.58%** |
| Near-full 180-second mean accept, +1 | 7.784 | 7.098 | -0.686 |

The lower acceptance in both faster windows rules out acceptance inflation as
the source of the gain. At the historical AR reference of 1,665.47 tok/s, the
latest near-full TF window is **2.025x** AR.

The complete finite batch generated 3,308,106 tokens in 1,497.23 seconds:
2,209.48 tok/s, mean acceptance 7.23, and 38.92% draft-token acceptance. That
whole-run rate is only 0.23% above the earlier 2,204.39 tok/s because the long
low-batch terminal drain dominates it; it is not a saturation measurement.
The 4K regression screen was also within the prior run-to-run range at
2,315.71 tok/s and mean acceptance 5.17.

Raw latest-run artifacts are on `vp-dgx-86` under:

```text
/tmp/jinzhao_tidar_opt_4k/
/tmp/jinzhao_tidar_opt_32k_full/
/tmp/jinzhao_tidar_opt_32k_full_metrics.tsv
```

### FA3 split-KV optimization

A delayed, pass-separated H100 profile at B32 showed that sampler work was no
longer the limiting factor. The target and draft model forwards accounted for
89.8% of GPU time, while the complete DSpark Markov loop was 3.94%. Flash
Attention kernels alone accounted for about 40%.

The production launcher had retained `VLLM_TIDAR_FA_NO_SPLITS=1` from the
historical argmax-draft path. That setting preserves a particular BF16
reduction order, but prevents FA3 from parallelizing long KV sequences. The
stochastic production path was rerun with FA3's captured split-KV scheduler
enabled. Both profiles started after 1,800 engine iterations and captured 100
iterations.

| GPU time per B32 iteration | No split | Split-KV | Change |
|---|---:|---:|---:|
| Target forward | 33.627 ms | **25.687 ms** | **-23.6%** |
| Draft forward | 31.970 ms | **25.313 ms** | **-20.8%** |
| DSpark Markov loop | 2.880 ms | 2.877 ms | -0.1% |
| Target sampler | 1.289 ms | 1.292 ms | +0.2% |
| Total self CUDA time | 73.060 ms | **58.200 ms** | **-20.3%** |

The throughput A/B used the same `vp-dgx-49` H100, source revision,
checkpoint, ordered 128-request workload, concurrency 32, natural EOS,
32,768-token cap, BF16, `T_d=T_AR=0.6`, exact rejection, `logprobs=1`, routed
experts, `gpu_memory_utilization=0.75`, and `FULL_AND_PIECEWISE`. The only
configuration difference was `VLLM_TIDAR_FA_NO_SPLITS`.

To control for the different stochastic trajectories, the primary comparison
uses nearly the same cumulative generated-token tranche in each run:

| FA3 mode | Generated-token tranche | Output tok/s | Mean accept, +1 | Request-blocks/s |
|---|---:|---:|---:|---:|
| No split | 209,541 to 823,257 | 3,062.65 | 7.370 | 415.58 |
| Split-KV | 206,815 to 829,531 | **3,884.63** | **7.838** | **495.69** |
| Change | matched context | **+26.8%** | +6.3% | **+19.3%** |

The acceptance-independent request-block rate confirms that this is a kernel
speedup, not only a favorable sampled trajectory. The best near-full
180-second split-KV window reaches **3,895.88 tok/s**, mean acceptance 7.845,
and 496.67 request-blocks/s at mean occupancy 31.84. Against the matched
1,665.47 tok/s AR reference, that is **2.339x**.

Split-KV changes BF16 attention reduction order, so output tokens are not
bitwise identical. Exact rejection and sampling distributions are unchanged,
and this stochastic run improved rather than reduced acceptance. The older
argmax-draft path previously showed an acceptance penalty from split-KV and
should retain `VLLM_TIDAR_FA_NO_SPLITS=1` until it is independently
revalidated. The checked-in stochastic reproducer defaults the variable to
zero:

```bash
CUDA_VISIBLE_DEVICES=7 \
  benchmarks/tidar/run_stochastic_tp1_dspark.sh serve

# In a second terminal after /health is ready:
benchmarks/tidar/run_stochastic_tp1_dspark.sh bench
```

Raw artifacts on `vp-dgx-49`:

```text
/tmp/jinzhao_tidar_fa_nosplit/
/tmp/jinzhao_tidar_fa_split/
/tmp/jinzhao_tidar_opt_profile_ranges_nosplit/
/tmp/jinzhao_tidar_opt_profile_ranges_split/
```

### Matched 32K production-contract baseline

The 4,096-token comparison above overweights early decode, where this workload
has lower acceptance. A follow-up restored the original production contract:
128 requests from the same thinking-on `workload_16x8.jsonl`, concurrency 32,
natural EOS, a 32,768-token safety cap, `T_d=T_AR=0.6`, K16,
`gpu_memory_utilization=0.75`, `FULL_AND_PIECEWISE`, `logprobs=1`, routed
experts, TP1, and no DP on one H100.

The pre-hot-path exact-rejection TF run completed all 128 requests without an
error, OOM, or preemption:

| Phase | Duration | Output tokens | Output tok/s | Mean accept, +1 |
|---|---:|---:|---:|---:|
| Strict B32 intervals | 1,012.47 s | 2,719,969 | **2,686.47** | **7.194** |
| Refill prefix through last B32 sample | 1,072.57 s | 2,879,729 | **2,684.89** | **7.209** |
| Terminal drain | 470.69 s | 442,210 | 939.50 | 6.914 |
| Complete finite batch | 1,510.58 s | 3,329,906 | **2,204.39** | **7.161** |

Mean output length was 26,014.9 tokens; 82 of 128 requests reached the 32,768
cap. The complete-run draft-token acceptance rate was 38.51%.

For the historical steady-window comparison, apply the same selection to both
modes: the best 180-second interval for which every telemetry sample reports
32 running requests.

| Mode | 180-second B32 tok/s | Mean accept, +1 | TF/AR |
|---|---:|---:|---:|
| TiDAR TF, exact rejection | **3,076.60** | **7.783** | **1.847x** |
| AR | **1,665.47** | n/a | 1.000x |

The exact sampler therefore does exceed the previously quoted 6.73 acceptance
when the 32K context distribution and steady-window definition are restored.
The 4K whole-run result of 5.29 was not a sampler regression. It measured a
different, early-context distribution. The AR control was stopped after 521
seconds of strict B32 telemetry: its stochastic trajectory then reached 99%
KV use and moved to 27 running / 5 waiting, which is cache-admission goodput
rather than a saturated B32 comparison.

Raw artifacts are on `vp-dgx-86` under:

```text
/tmp/jinzhao_tidar_exact_32k_20260714/
/tmp/jinzhao_ar_32k_20260714/
```

### Original 32K live result

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

### Drain-controlled bsz32

A separate lockstep run isolates saturated bsz32 decode from the live
workload's KV admission and terminal drain. It uses the same checkpoint,
DSpark+Markov draft path, K16, target temperature 0.6, argmax draft, and
thinking-on prompt mode. To match the earlier drain-controlled H100 probe, it
uses `FULL_AND_PIECEWISE`, a 12K model limit, 0.65 GPU-memory utilization, no
logprob or routed-expert output, and a fixed 10,000 output tokens per request.
EOS is bypassed only for this fixed-work benchmark.

The client submits 32 independent requests with seed 0. All 32 remain resident
with no waiting, produce exactly 10,000 tokens, and finish within 0.24 seconds
of one another.

| Metric | DSpark+Markov bsz32 |
|---|---:|
| Output throughput | **1,848.8 tok/s** |
| Per-request throughput | **57.78 tok/s** |
| Mean acceptance length, including bonus | **2.787** |
| Draft-token acceptance | 11.17% |
| Mean request latency | 173.07 s |
| Successful requests | 32 / 32 |

This is 1.72x the live run's 1,074.2 tok/s refill-controlled rate. The live
workload averaged only 15.74 running sequences at 94.97% KV use and carried
much longer contexts, so its concurrency setting of 32 was not a resident
bsz32 decode batch.

The earlier `iter_0012600` drain-controlled result was 3,588.0 tok/s with mean
acceptance 5.097 and a 45.46 ms device TF iteration. The DSpark run executes
114,816 TF iterations across the batch; its 173.08-second end-to-end duration
gives a conservative 48.24 ms per iteration including prefill and HTTP
overhead. Thus iteration cost is at most about 6% higher, while acceptance is
45% lower. The throughput difference is therefore predominantly this
checkpoint/prompt acceptance difference, not a collapse in bsz32 kernel
scaling.

The rerun also exposed two startup defects that the live configuration had
masked because routed-expert output was enabled: the V2 runner accessed the
routed-expert KV-group index even when capture was disabled, and memory
profiling left a dummy `execute_model_state` available to the first async
sample. The branch now gates the route slot mapping and clears the consumed
dummy state.

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
