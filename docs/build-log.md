# Build record

This file is the construction record for the official NVFP4 line. The EXL3
line keeps its own record in `Ninfer_Exl3_Glm-5.3_Flash_2x_DgxSpark`. The
checkpoints, decoders, and repositories stay separate.

## Hardware destination

The product runs on two NVIDIA DGX Spark nodes. Each GPU is a GB10, Blackwell,
compute capability 12.1 (`sm_121`), aarch64, with 121 GiB of memory shared by
the CPU and the GPU. GB10's native 4-bit tensor-core format is NVFP4. That is
why this quant has its own runtime. EXL3 trellis is not that hardware path.

Observed link, 2026-09-22:

| Role | Address | CX-7 | RDMA device | RoCE v2 GID |
| --- | --- | --- | --- | --- |
| Head `gx10-5749` | 192.168.100.10/24 | `enp1s0f0np0` | `rocep1s0f0` | index 3 |
| Worker `gx10-23ec` | 192.168.100.11/24 | `enp1s0f1np1` | `rocep1s0f1` | index 4 |

Worker GID index 3 is empty. `ninfer-glm53-generate` is still a host process.
It does not launch a kernel or place a rank on the worker.

On 2026-09-22 a dual-node EXL3 vLLM server filled the unified memory from
about 06:25 until a hard reboot at 13:58. The NVFP4 head container was already
stopped. Do not start another full-memory forward while the desktop is in use.

## Pins

| Piece | Pin |
| --- | --- |
| Product checkpoint | `nvidia/GLM-5.3-Flash-NVFP4` `09b04e5e74bca08ca8549fc736d4cdd8624bfde3` |
| Storage | ModelOpt, `quant_method=modelopt`, `quant_algo=NVFP4` |
| Catalog | 147661 tensors, shapes checked, receipt `docs/receipts/nvidia-glm53-flash-nvfp4.json` |
| Refused | RedHatAI compressed-tensors `18d55bfd5a2194887738da73753975c9d3842f46` (weights not on disk; not the product) |
| DFlash2 draft | `incoai/GLM-5.3-Flash-DFlash2` `dc77ff1c99eeb2df044ee3d4f0094eb033fee410` |
| Fixed prompt | `fixed-text-v1`, text `Hi`, token id 13041 |

Packed layout used by the host decoder: low nibble is even K, high nibble is
odd K, value `e2m1 * fp8_e4m3(scale) * weight_scale_2`. A zero block scale stays
zero. NaN scales stay NaN. `K` must be a positive multiple of 16.

## What was built, in order

1. The EXL3 repository's host contracts were copied as the model, deployment,
   and TP2 plan, then retargeted at this checkpoint.
2. The logical catalog and binder accept official ModelOpt NVFP4 and reject
   compressed-tensors. Dense MLP layers 0–2 and routed experts 3–44 are NVFP4.
   Layer 45 experts stay BF16. Attention, router, shared experts, embeddings,
   `lm_head`, and vision stay BF16 or FP32.
3. FP8 E4M3 decode uses `ldexp` for subnormals. The old `1 << (exp-7)` shift
   was undefined when the exponent was below 7 and corrupted 96 finite codes.
4. DFlash2 temperature-0 commit keeps `target[matched]` on a partial accept
   and the bonus on a full accept. Two local records are written. NCCL and
   KDA/KV rollback are not in this helper.
5. `ninfer-glm53-generate` runs the same CPU eager text stack as the EXL3
   line, with NVFP4 experts and NVFP4 dense MLP layers 0–2. Local TP2 and
   DFlash2 k=7 fork two processes on one machine.
6. While that forward was being recorded, `origin/main` gained `5ec9d09`
   (validated packed views) and `af001e0` (checkpoint admission, read-only
   preflight, receipt I/O). Both were merged. The packed view still
   multiplies by ModelOpt `weight_scale_2`. A global scale that is
   non-finite, not positive, or subnormal is rejected. Receipt errors print
   `path.string()`, because streaming a `std::filesystem::path` adds quotes
   and hid the `/dev/full` failure from `tests/test_bind_cli.py`.

## Experiments

There is no external NVFP4 temperature-0 continuation yet. The stopped
`glm53-nvfp4-head` container was not restarted. The CPU forward of prompt
13041 was recorded as `154822,154822` (`[gMASK]`) for greedy, local TP2, and
DFlash2 k=7. That agrees with a second CPU transcription and with the EXL3
CPU stack after the trellis fix. It does not agree with the dual-Spark EXL3
vLLM result `13041,13041`. The EXL3 ids are not an NVFP4 measurement. Copying
them here would be false.

The host suite is the contract tests, `nvfp4_decode`, `nvfp4_view`,
`text_forward`, `speculative`, and `contract_regressions`. On 2026-09-22,
after merging the remote host-hardening commits, keeping ModelOpt
`weight_scale_2` as a multiply, and printing receipt paths with
`path.string()`, that suite was 12/12 passed from a Release build on this
GB10. The same tree passed `test_checkpoint_manifest.py`,
`test_preflight.py`, and `test_bind_cli.py`.

## Upstream looked at on 2026-09-22

- `Neroued/ninfer` `9e163ee` (2026-09-18) is Q4/Q5 GEMM routing, not NVFP4
  GLM-5.3. Branches: `master`, `dev`.
- `incoai/splash` `e8fffde` (2026-09-21) is Mac chat and Qwen Q4.
- `MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks` `c1b7d4c` (2026-09-22) changes
  the EXL3 launcher and KV reservation. It does not publish this ModelOpt
  checkpoint or an NVFP4 kernel.

## Launch refused, measured 2026-09-22T14:25:17+08:00

The official snapshot `09b04e5e74bca08ca8549fc736d4cdd8624bfde3` is on disk (191G, `quant_method=modelopt`, `quant_algo=NVFP4`). Head `gx10-5749` had an active wayland session and MemAvailable 120881568 KiB. `glm53-nvfp4-head` was Exited. Worker `gx10-23ec` answered ping, with MemAvailable 6.1 GiB because `glm53-exl3-worker` (`VLLM::Worker_TP` pid 3737500) held 106226 MiB. `glm53-nvfp4-worker` stayed Exited. `ninfer-glm53-generate` was not started. No independent NVFP4 continuation was recorded, and no token ids were produced.

The existing host build's CTest was 12/12, and `test_bind_cli.py`, `test_checkpoint_manifest.py`, and `test_preflight.py` passed. Dequant in `include/ninfer_glm53/nvfp4_decode.hpp` remains `e2m1 * fp8_e4m3(block_scale) * weight_scale_2`. That suite is not a GB10 continuation.

## Short greedy on the worker, measured 2026-09-22T14:33:40+08:00

`ninfer-glm53-generate` ran on `gx10-23ec` (NVIDIA GB10 12.1, `GPU-d5d58447-ec10-0155-e20a-cf12158797b9`) against `09b04e5e74bca08ca8549fc736d4cdd8624bfde3`. Prompt `13041`, greedy, `new_tokens=2`. Both repeats returned `token_ids=154822,154822`, `committed_length=2`, `world_size=1`. The first-token logits were `154822=20.1262` and `315=6.60393`. No independent NVFP4 continuation was recorded. Dequant remains `e2m1 * fp8_e4m3(block_scale) * weight_scale_2`.

## Still required on Blackwell

SM121 NVFP4 GEMM that keeps packed bytes packed, BF16 projections, KDA, sparse
MLA, the indexer, routed MoE, NCCL TP2 over the CX-7 link, fp8 KV, and a
temperature-0 continuation from a Blackwell NVFP4 server. Placeholder kernels
are not part of this record.

## Host conv history, DFlash continuation, and local TP2 lifetime

The in-place causal conv now keeps the raw channel for the history tail, so a Q/K/V alias no longer stores the SiLU output. DFlash restores the rejected candidate cache and continues until the budget or EOS. Store releases each mapped shard when a load fails, and the local TP2 path reaps its child with a 30-second deadline on each rank-link send or receive. CTest after that host fix was 12/12 passed. This does not create an NVFP4 external continuation and does not change multiplication by `weight_scale_2`.

On 2026-09-22T17:55+08:00 the EXL3 image's prefill `causal_conv1d_fn`, given a cache slot other than null block 0, matched `silu(x * weight[:, -1])` at cosine `1.00000024`. The NVFP4 host conv uses that same last-tap rule and was not edited. `weight_scale_2` is still multiplied into the packed GEMV. No NVFP4 greedy run was repeated, and there is still no independent NVFP4 continuation of prompt `13041`.

On 2026-09-22T19:32+08:00 the on-disk official snapshot at `/home/max_aibbox/models/GLM-5.3-Flash-NVFP4` was 45 files totaling 204476289317 bytes (190.43 GiB). One GB10 has 121 GiB of unified memory, so that tree cannot be loaded whole on one Spark. No external NVFP4 continuation was started, and no token ids were copied from the EXL3 run. Multiplication by `weight_scale_2` was not changed.

On 2026-09-22T21:44+08:00 the worker's DFlash2 snapshot `dc77ff1c` file `model.safetensors` was 2342169800 bytes and 81 tensors. The names include `fc.weight`, `hidden_norm.weight`, five decoder layers, and `candidate_selector` codebooks. There is no `embed_tokens` and no `lm_head`, so this draft file cannot emit a token continuation on its own. No NVFP4 weights were loaded. Multiplication by `weight_scale_2` was not changed. No token ids were copied from the EXL3 run.

On 2026-09-23T00:05+08:00 `ninfer-glm53-generate` ran again on `gx10-23ec` (NVIDIA GB10 12.1, `GPU-d5d58447-ec10-0155-e20a-cf12158797b9`) from `/home/max_aibbox/models/GLM-5.3-Flash-NVFP4`, revision `09b04e5e74bca08ca8549fc736d4cdd8624bfde3`. Prompt `13041`, greedy, `new_tokens=2`, `world_size=1`. Both repeats returned `token_ids=154822,154822`, `committed_length=2`. The first-token logits were `154822=20.1262` and `315=6.60393`. These ids are this checkpoint's own forward. They were not copied from the EXL3 continuation `13041,13041`. No independent NVFP4 server continuation exists. Multiplication by `weight_scale_2` was not changed. Worker MemAvailable was 122872308 KiB before and 122916128 KiB after. Host ctest was 12/12.

On 2026-09-23T00:12+08:00 the same entry ran as TP=2. Rank 0 bound `192.168.100.10` on `gx10-5749` (`GPU-ca529c58-c4c3-2793-3566-260e0f97879a`). Rank 1 bound `192.168.100.11` on `gx10-23ec` (`GPU-d5d58447-ec10-0155-e20a-cf12158797b9`). Ports 29661 and 29662. Both repeats, on both ranks, returned `token_ids=154822,154822` and committed length 2, matching the greedy run above. `second_spark=192.168.100.11`. Head MemAvailable stayed near 119325132 KiB and the worker near 122742012 KiB. This agreement is still not an external NVFP4 continuation.

On 2026-09-23 the independent continuation was measured on `gx10-23ec` (`GPU-d5d58447-ec10-0155-e20a-cf12158797b9`) inside `glm53-nvfp4-local:20260914`, walking one official `Glm5NextDecoderLayer` at a time with that image's ModelOpt kernels. It is not the NINFER binary. Checkpoint revision `09b04e5e74bca08ca8549fc736d4cdd8624bfde3`, `quant_method=modelopt`, `quant_algo=NVFP4`. Prompt token `13041`, temperature 0. The zero-state chain's final argmax is `154822` at logit `20.25`, runner-up `315`. The capture recipe steps token 0, clears the MLA page, then steps `13041`. That primed chain's final argmax is `13041` at logit `12.75`, runner-up `22654`. The same pass then steps `13041` again with the KDA state kept and the MLA page kept, and its final argmax is `198` at logit `10.25`. The same-pass continuation is `13041,198`, committed length 2. A later reload of saved state produced the same pair. Host MemAvailable stayed near 118 GiB. These ids were not copied from the EXL3 continuation.

The shipped forward now steps token 0 before the prompt, clears the MLA cache, stores each mHC stream in bf16, stores the conv history raw value in bf16, and round-trips each MLA latent through fp8 e4m3 groups of 128 with scale `amax/448`. Multiplication by `weight_scale_2` is unchanged. Host ctest after that change was 12/12. On `gx10-23ec` the greedy run, temperature 0, `new_tokens=2`, returned `token_ids=13041,13041`, committed length 2. The first-token logits were `13041=12.6525` and `22654=12.0867`, matching the primed reference's first id and runner-up. The second token was `13041` at logit `17.2`, not the reference id `198`. Real TP=2 on `192.168.100.10` / `192.168.100.11`, ports 29673 and 29674, repeated that pair on both ranks twice, `second_spark=192.168.100.11`. DFlash2 k=7 on the worker, draft `dc77ff1c99eeb2df044ee3d4f0094eb033fee410`, twice, committed `13041,13041` with `dflash_accept=partial_plus_bonus`, accept count 1, and draft ids `13041,25,13041,12,13041,11,4194`. Both DFlash ranks committed length 2. That local pair is not the second Spark. The eager second token does not yet match the independent continuation `13041,198`.

## Paused handoff, measured 2026-09-23

This commit is an unfinished numeric checkpoint. It is not a passing continuation. On `gx10-23ec` (`GPU-d5d58447-ec10-0155-e20a-cf12158797b9`), `ninfer-glm53-generate` loaded `/home/max_aibbox/models/GLM-5.3-Flash-NVFP4` revision `09b04e5e74bca08ca8549fc736d4cdd8624bfde3`, prompt `13041`, temperature 0, `new_tokens=2`, `world_size=1`. One run printed `token_ids=39,13041`, committed length 2. Step 1 argmax was `39` at logit `12.74`, runner-up `0` at `12.5236`, with `13041` at `8.588` and `198` at `10.3009`. Step 2 argmax was `13041` at `12.9358` and `198` at `10.0097`. That second step is not the official second step, because the first id was not `13041`. The independent reference remains `13041,198`. TP=2 and DFlash2 were not rerun on this tree. Host `nvfp4_decode` and `text_forward` passed after the build; the full 12-test suite was not rerun. Worker MemAvailable stayed near 122.9 GiB. `weight_scale_2` is still multiplied inside `gemv_nvfp4_rows`.

Against the official prompt-step dumps, layer 0 was: attention input neq 0, recurrent output neq 3, o_proj input neq 4, attention output neq 7, MLP input relative error `0.000550` (68 lanes), FFN relative error `0.018329` (3692 lanes). Layer 1 attention relative error was `0.014664` and its FFN relative error was `0.115153`. Layer 3's routed set had expert `74` where the official set has `23`. Layer 4 had `175` where the official set has `106`, and did include `86`. Route set agreement over layers 3–44 was 5. `NINFER_LAYER_DUMP`, the route print, and the `id13041` / `id198` logit print are still in the forward so the next measurement can use them. They are not serving behavior.

The same day, one worker CUTLASS replay of layer 0 (`cutlass_scaled_fp4_mm`, alpha = `input_scale * weight_scale_2`, bf16 output) showed the split. A fused `gate_up` GEMM, clamped SiLU, and `down_proj` on the official MLP input matched the captured official FFN at bf16 neq 0. Separate `gate_proj` and `up_proj` CUTLASS calls, then the same SiLU and `down_proj`, landed on the shipped CPU FFN instead (relative error about `0.018` versus the official FFN). A grouped fp32 dot that multiplies `input_scale * weight_scale_2` once after the sum matched those separate CUTLASS outputs at bf16 neq 0 for gate, up, and down. That epilogue was not patched into the shipped GEMV. The next edit should make the dense MLP follow the fused gate_up result, then rerun the worker greedy until it prints `13041,198`.
