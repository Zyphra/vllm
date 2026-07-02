# TiDAR Recovered-Token Cache And Concurrent Decode Handoff

Date: 2026-07-01

This note explains the current working diagnosis for the TiDAR two-forward KL failure that appears under concurrent batched decode but not under sequential decode.

## Short Diagnosis

The bad KL is best explained by a post-rejection state-alignment bug, not by model weights, tokenizer, temperature, or router probabilities.

In TiDAR two-forward decoding, a block can reject inside the draft window. After the first rejection, the emitted token sequence is no longer the same as the raw draft/verifier row sequence:

```text
draft block:      d0 d1 d2 d3 ... d15
accepted prefix: d0 d1
recovered token:       r2
emitted tokens:  d0 d1 r2
rejected suffix:          d3 ... d15
```

There are two state objects that must be compacted to this emitted sequence:

1. Returned AR logprobs must be gathered for the final emitted tokens, not raw verifier rows after the rejected token.
2. CCA recurrent/cache state must be committed to the post-acceptance state for `d0 d1 r2`, not the default post-full-draft state for `d0 d1 d2 ... d15`.

Sequential decode can hide this because request order, row order, and cache slot order are usually identical. Concurrent mixed prefill/decode exposes it because rows, request indices, and state slots are no longer a trivial contiguous mapping.

## Evidence We Have

### 1. AR/single-forward mode gives good KL

AR mode does not have a speculative rejected suffix or recovered-token replacement. Its mapping is simple:

```text
gen_ids[i] <-> model row i <-> rollout_logprob[i] <-> cache advanced by gen_ids[i]
```

So AR-mode KL being fine does not rule out a TiDAR-specific post-rejection compaction/cache bug. It instead supports the conclusion that the base model, tokenizer, temperature, and trainer scoring path can match.

### 2. Post-upstream standalone comparator still gets good KL

After moving `verl_trainer` to `afb0c78a`, the standalone comparator still reached target KL:

```text
/tmp/compare_no_router_upstream_pull_sft5300_ecp4_allgather_20260630T091339Z
ppo_approx_kl = 5.49e-4
kl_divergence = 2.14e-4
n_tokens = 8192

/tmp/compare_router_upstream_pull_sft5300_ecp4_allgather_20260630T092236Z
ppo_approx_kl = 1.11e-4
kl_divergence = 4.91e-5
n_tokens = 8192
```

Router replay metadata was clean:

```text
source = legacy_shm
num_layers = [40]
missing_tokens = 0
route_prob_replay_uses = 0
fallbacks = 0
```

This means the low-level trainer scoring path and router-index replay can match when the generated-token/logprob/state stream is already coherent.

### 3. We saw the misalignment pattern around sample 3, generated position 1002

The saved comparator artifact showed a block-local rejection around sample 3 token position `1002`. With `num_speculative_tokens=16`, this falls inside a speculative block, not at a natural boundary. After a rejection inside a block, raw verifier rows after the rejection correspond to discarded draft tokens, while `gen_ids` correspond to compacted emitted tokens.

That observation led to the AR-logprob correction: do not return raw AR verifier rows directly; gather rows according to the final `sampled_token_ids` layout and filter placeholders exactly as the output token path does.

### 4. Live training has low acceptance, so the failure case is frequent

Live logs showed mean acceptance length around `1.8-2.6` and average draft acceptance around `5-10%`. With `K=16`, most blocks reject early. That means most blocks have a recovered token and a rejected suffix. If cache/logprob compaction is wrong, live training will hit the bug constantly.

### 5. Temperature matched in the live run

The live process had:

```text
VERL_ACTOR_LOGPROB_TEMPERATURE=1.0
VLLM_TIDAR_AR_TEMPERATURE=1.0
tidar_ar_temperature=1.0
tidar_diff_temperature=0.0
```

So the live bad-KL behavior is not explained by AR verifier temperature mismatch.

### 6. Sequential two-forward decode works, concurrent decode does not

That pattern points to a request/row/slot mapping bug. If weights, temperature, tokenizer, or the AR verifier distribution were wrong, sequential decode would also be bad. Concurrency specifically adds mixed prefill/decode rows and request reordering, so bugs that assume contiguous per-request row order become visible.

## Relevant Code Paths

All code references below are in:

```text
/workspace/Megatron-zyphra/vllm-smoe-amd-diffusion
```

### Rejection sampler semantics

`vllm/v1/sample/rejection_sampler.py:55-76` defines the contract:

```text
output tokens = accepted tokens + recovered tokens + bonus tokens
```

This matters because recovered tokens are emitted, while rejected draft suffix rows are not emitted.

### AR-logprob gathering

`vllm/v1/sample/rejection_sampler.py:181-190` builds `output_token_ids` through `rejection_sample(...)`.

`vllm/v1/sample/rejection_sampler.py:192-229` chooses whether to return AR verifier logprobs when `VLLM_TIDAR_RETURN_AR_LOGPROBS=1`.

`vllm/v1/sample/rejection_sampler.py:275-345` is the important gather path. The current code explicitly builds `accepted_logit_indices` from `metadata.target_logits_indices` and `metadata.bonus_logits_indices`:

```text
295-299: comment explaining that mixed prefill/TiDAR decode rows are not a simple packed layout
300-319: build accepted_logit_indices per request/slot from target and bonus row maps
320-345: gather logprobs for sampled_token_ids, then parse_output filters placeholders
```

This is the intended fix for the sample-3/token-1002 class of logprob bug. The key invariant is:

```text
returned_logprobs[j] must be gathered for final_emitted_token[j]
not raw_verifier_row[j] after the first rejection
```

`vllm/v1/sample/rejection_sampler.py:416-450` filters `PLACEHOLDER_TOKEN_ID` rows in `parse_output(...)`. Logprob filtering must use the same mask as token filtering.

### SpecDecodeMetadata row maps

`vllm/v1/spec_decode/metadata.py:10-25` defines the row maps:

```text
draft_token_ids
target_logits_indices
bonus_logits_indices
logits_indices
```

These maps are the safe way to reconstruct per-request verifier rows. Assuming raw row order is unsafe once the engine mixes prefill/decode/spec rows.

### CCA commit contract

`vllm/v1/attention/backends/cca_attn.py:31-38` documents that verify forward defaults to writing the last-position state, and `commit_spec_decode_state` overwrites it with the post-acceptance state.

`vllm/v1/spec_decode/cca.py:185-216` defines `commit_spec_decode_state(...)`:

```text
Called after rejection sampling, outside the captured cudagraph.
num_accepted_per_batch_idx[i] is the number of drafts accepted for input-batch position i.
It gathers candidate offset num_accepted+1 clamped to [0, K] from the stash.
```

`vllm/v1/spec_decode/cca.py:243-254` performs the actual state replacement:

```text
selected_conv = _spec_stash_conv[arange, idx]
selected_hs = _spec_stash_hs[arange, idx]
slots = _spec_stash_slots[:n_used]
conv_states[slots] = selected_conv
prev_hs[slots] = selected_hs
```

This is where the recovered-token cache problem becomes concrete. If `idx` or `slots` are in the wrong request order, CCA commits the wrong post-acceptance state to a request's recurrent state slot.

### Runner-side CCA commit

`vllm/v1/worker/gpu_model_runner.py:2404-2465` computes `num_accepted_per_batch_idx` and calls every CCA layer's `commit_spec_decode_state(...)`.

The accepted count is currently derived as:

```python
num_accepted_per_batch_idx = [
    max(0, len(tokens) - 1 + _commit_offset)
    for tokens in valid_sampled_token_ids
]
```

This assumes `valid_sampled_token_ids` is in exactly the same batch/request order as the CCA layer stash and `_spec_stash_slots`.

This assumption is usually true in simple sequential decode. It is the suspect assumption under concurrent mixed prefill/decode.

### Async / concurrent scheduling path

`vllm/v1/worker/gpu_model_runner.py:3317-3331` commits CCA state in the non-async path after `RejectionSampler.parse_output(...)`.

`vllm/v1/worker/gpu_model_runner.py:3332-3348` is the async path. It does not parse `valid_sampled_token_ids` immediately; it caches GPU sampled tokens and request-index maps for the next step.

`vllm/v1/worker/gpu_model_runner.py:4370-4445` later wraps the output in `AsyncGPUModelRunnerOutput`, where `parse_output(...)` happens after an async CPU copy.

This split is important: if CCA commit depends on parsed compacted output tokens but async/concurrent execution defers that parsing, then CCA state can be committed too early, skipped, or committed using a stale/simple ordering assumption.

### Next-draft rollback after rejection

`vllm/v1/spec_decode/utils.py:14-54` computes `num_rejected_tokens_gpu` from the number of valid sampled tokens. It stores both the row to sample and the number of rejected tokens.

`vllm/v1/spec_decode/tidar.py:969-983` subtracts `num_rejected_tokens_gpu` from `base_seq_lens` before building the next draft metadata:

```text
After an intra-block rejection the verifier forward has advanced metadata through the full draft block,
but the live continuation is only accepted_prefix plus the recovered token.
Mirror Eagle's rollback here so the next TiDAR draft reads prefix + accepted_prefix, then consumes next_token_ids[req].
```

This is the exact recovered-vs-rejected cache-state problem in code. If that rejected-token count is missing, misordered, or applied to the wrong request, the next draft sees a prefix that includes rejected draft suffix state rather than the actual emitted prefix.

## Exact Failure Mechanism

For a request with early rejection:

```text
K = 16
accepted drafts = 2
emitted tokens this step = 3  # d0, d1, recovered r2
rejected tokens = K + 1 - emitted_count = 14
```

The target verifier forward may have produced/stashed state for the whole `K+1` verifier window. But the live sequence length and CCA state for the next step must correspond to only the emitted continuation.

Correct post-step state:

```text
prefix + d0 + d1 + r2
```

Incorrect state if not compacted/rolled back:

```text
prefix + d0 + d1 + d2 + d3 + ... + d15
```

or a hybrid corrupted state where sequence length is rolled back but the CCA recurrent state slot is committed from the wrong request or wrong candidate offset.

Once that happens, future logits are conditioned on cache state that does not match `gen_ids`. The trainer later scores the actual `gen_ids` under a correct Megatron forward, while rollout logprobs/text came from a vLLM cache trajectory conditioned on the wrong state. KL becomes bad even though router indices and temperatures are correct.

## Why Sequential Decode Can Pass

Sequential decode usually has:

```text
one active request
no mixed prefill rows
stable request index 0
stable CCA stash slot 0
row order == request order == output order
```

Under those conditions, fragile assumptions hold accidentally:

```text
valid_sampled_token_ids[0] belongs to CCA stash row 0
num_rejected_tokens_gpu[0] belongs to request 0
target_logits_indices are effectively contiguous for the one request
```

Even if the code is not fully robust to request reordering, there is no reorder to expose it.

## Why Concurrent Mixed Prefill/Decode Fails

Concurrent execution can mix:

```text
prefill rows for new requests
decode rows for old requests
TiDAR verifier rows
TiDAR drafter rows
requests with different accepted lengths
requests that were discarded or preempted
```

Then the following mappings must all stay consistent:

```text
model row -> request id
model row -> generated position
sampled output slot -> target/bonus verifier row
num_rejected_tokens_gpu[i] -> same request i used by next drafter metadata
num_accepted_per_batch_idx[i] -> same request i used by CCA stash slots
CCA stash slot -> persistent request state slot
router indices -> second forward rows for final committed tokens
```

The current suspected bug is that at least one of these mappings is still using batch/list position as an implicit request identity. That is safe in sequential decode and unsafe in concurrent mixed prefill/decode.

## What This Means For Rollout Correctness

If the issue is only AR-logprob compaction, the rollout text can be correct but KL is bad because returned logprobs are for the wrong rows.

If the issue is recovered CCA state, the rollout itself can be semantically incorrect: future tokens are generated from cache state that is not conditioned on the emitted tokens. That is more serious than logging/bookkeeping. It means the vLLM rollout trajectory diverges from the actual `gen_ids` prefix.

The current symptoms are compatible with either, and under concurrency they may both occur.

## What To Verify Next

1. Enable `VLLM_TIDAR_DEBUG_REJECTION_DIR` on a small concurrent run. Check that each returned logprob row was gathered from the target/bonus row corresponding to the final emitted token after placeholder filtering.

2. Enable `VLLM_TIDAR_DEBUG_STASH=1` with a low max step count. For each request, compare:

```text
sampled_token_ids after parse_output
num_accepted_per_batch_idx
idx used by commit_spec_decode_state
_spec_stash_slots
selected_conv/selected_hs
```

The request order must match exactly.

3. Add request-id keyed diagnostics around `_commit_tidar_cca_state`. Do not rely only on list index. Log:

```text
req_ids_output_copy
req_id_to_index_output_copy
valid_sampled_token_ids lengths
CCA stash slots in batch order
num_rejected_tokens_gpu
```

4. Run the same prompts sequential and concurrent with identical seed/config. If sequential KL is good and concurrent KL is bad, diff the debug dumps for the first request whose accepted length is less than K.

5. Prefer request-id keyed commit plumbing if diagnostics confirm mismatch. The robust fix is to derive CCA commit indices and rejected-token rollback from request ids or explicit scheduler/input-batch indices, not from implicit list position.

## Guardrails

- Continue replaying router indices only. Do not replay route/gate probabilities.
- Keep `tidar_ar_temperature=1.0` aligned with Megatron score temperature.
- Keep `num_layers=[40]` for router replay metadata.
- Do not treat acceptance rate as a KL correctness metric. Low acceptance increases exposure to this bug, but bad KL requires a row/state mismatch.
