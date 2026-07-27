# ZAYA1 BF16 AITER FMoE rows (image-scoped promotion)

These nine exact rows are scoped to the AITER build in image
`sha256:edb4ba6c85813723fc8ef29e695566bb5e881cc2a9f4c216f9ae3ba0d403f829`.
They are inert in the vLLM tree and must not become a cross-image default.

The candidate file is
`model_configs/zaya1_bf16_tuned_fmoe_edb4ba6c.csv` (SHA-256
`4ceb5e78fc5aaee1a9e7b28ee90407629e7b3230c8012a4daa77449fcf3ae091`).
It covers tokens `1,2,4,8,16,128,256,512,1024` for the exact
MI300X/ZAYA1 BF16 E16, top-1, H=I=2048 shape.

## Activation contract

AITER, rather than vLLM, owns the runtime config. For this image its canonical
source location is `aiter/configs/model_configs/`. At image-build time, copy
the candidate there without changing its basename. Leave `AITER_CONFIG_FMOE`
unset so AITER merges the root config and every model config normally.

Do not point `AITER_CONFIG_FMOE` directly at this exact-only file. That bypasses
the image's generic and other model configs.

Before serving, require all of the following:

1. the image ID and candidate SHA-256 above match;
2. AITER resolves `/tmp/aiter_configs/tuned_fmoe.csv` to SHA-256
   `0dd5b69449d0438b58f901117847807d16cfd5e56c90cc4a7a8f947b2b4127b7`;
3. the resolved file contains exactly these nine ZAYA1 keys and no duplicate
   key for any of them;
4. startup logs report the expected kernels for every reached exact token.

The current image's native resolved config has SHA-256
`870bf1d01e85cab589bc45ebc67b560bcad755ec7d9b93475458974bfd078952`.
Resolver-equivalent validation produced 1,994 native plus nine candidate rows,
found zero exact-key collisions, and preserved the first-match result for all
1,687 existing keys.

## Evidence

Same-source ABBA job `324433` measured B1 `87.3294 -> 89.9359` TPS
(`+2.9847%`), C16 `952.364 -> 964.870` (`+1.3132%`), and C64
`2788.641 -> 2818.880` (`+1.0844%`). All four trajectories and log-probabilities
were exact, and runtime logs proved every intended row hit.

NUMA/GPU-pinned same-source reverse ABBA job `324584` then promoted the
combined native-CCA and current-image AITER profile: B1
`76.0745 -> 89.5045` TPS (`+17.65%`), C16 `850.947 -> 947.521`
(`+11.35%`), and C64 `2565.789 -> 2791.908` (`+8.81%`). Both paired
directions agreed; trajectories and token log-probabilities were exact and
finite; all nine candidate rows, native CCA, and FULL/PIECEWISE replay were
proved. The final receipt SHA-256 is
`c949d969a5ee922fff1a91d4ff41c6ef34752bcf34fe2b74a6cd92edf405afaf`.

The durable directional reducer is
`/shared/home/rob/research/zaya_profile_20260725/aiter_retuned_on_native_cca_${SLURM_JOB_ID}/summary.json`.
The promoted cumulative receipt is
`/shared/home/rob/research/zaya_profile_20260725/native_cca_aiter_cumulative_runs/run_324584/FINAL_RECEIPT.json`.
The older `85a5` overlay remains rejected on this image.
