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

## Experiments

There is no external NVFP4 temperature-0 continuation yet. The stopped
`glm53-nvfp4-head` container was not restarted. The CPU forward of prompt
13041 was recorded as `154822,154822` (`[gMASK]`) for greedy, local TP2, and
DFlash2 k=7. That agrees with a second CPU transcription and with the EXL3
CPU stack after the trellis fix. It does not agree with the dual-Spark EXL3
vLLM result `13041,13041`. The EXL3 ids are not an NVFP4 measurement. Copying
them here would be false.

The host suite for this tree is `model_spec`, `deployment_contract`,
`execution_plan`, `checkpoint_binding`, `nvfp4_decode`, `tiny_layer`,
`speculative`, and `text_forward`. On 2026-09-22 that suite was 8/8 passed
from a Release build on this GB10.

## Upstream looked at on 2026-09-22

- `Neroued/ninfer` `9e163ee` (2026-09-18) is Q4/Q5 GEMM routing, not NVFP4
  GLM-5.3. Branches: `master`, `dev`.
- `incoai/splash` `e8fffde` (2026-09-21) is Mac chat and Qwen Q4.
- `MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks` `c1b7d4c` (2026-09-22) changes
  the EXL3 launcher and KV reservation. It does not publish this ModelOpt
  checkpoint or an NVFP4 kernel.

## Still required on Blackwell

SM121 NVFP4 GEMM that keeps packed bytes packed, BF16 projections, KDA, sparse
MLA, the indexer, routed MoE, NCCL TP2 over the CX-7 link, fp8 KV, and a
temperature-0 continuation from a Blackwell NVFP4 server. Placeholder kernels
are not part of this record.
