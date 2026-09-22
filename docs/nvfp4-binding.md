# NVFP4 checkpoint binding

The product checkpoint is official `nvidia/GLM-5.3-Flash-NVFP4` revision
`09b04e5e74bca08ca8549fc736d4cdd8624bfde3`. `quant_method` is `modelopt` and
`quant_algo` is `NVFP4` (ModelOpt `0.47.0.dev393`). Compressed-tensors exports
are rejected.

## Measured storage

A logical linear of shape `[N, K]` (out, in) is stored as:

| Region | Tensors | Dtype and shape |
|---|---|---|
| Dense MLP layers 0–2 and routed experts in layers 3–44 | `.weight`, `.weight_scale`, `.weight_scale_2`, `.input_scale` | U8 `[N, K/2]`, F8_E4M3 `[N, K/16]`, F32 `[]`, F32 `[]` |
| Layer 45 routed experts | `.weight` only | BF16 `[N, K]` |
| Attention, router, shared experts, embeddings, `lm_head`, vision | native `.weight` and auxiliaries | BF16 or FP32 |

Gate and up use `N=2048`, `K=4096` on experts and `N=12288`, `K=4096` on the dense MLP. Down swaps those dimensions. The low nibble of each packed byte is the even K element. Reconstruction is `e2m1 * fp8_e4m3(weight_scale) * weight_scale_2`. `input_scale` is the activation scale and is not applied to the weight.

The catalog has 147,661 tensors: 2,473 native (213 of them FP32) and 145,188 NVFP4 storage tensors. There is no FP8-block expert group. On this export, KDA `q`/`k`/`v` convolutions are FP32 and every mHC tensor, including base and scale, is BF16. Logical catalog sha256 is `9df33d5e4a73bcd41237a3d9c2c69ec9c1b2e78087f2bcbb4ef2fb5660136d30`. The local snapshot is `/home/max_aibbox/models/GLM-5.3-Flash-NVFP4`, the same files as Hugging Face snapshot `09b04e5e74bca08ca8549fc736d4cdd8624bfde3`. The shapes-checked receipt is [`receipts/nvidia-glm53-flash-nvfp4.json`](receipts/nvidia-glm53-flash-nvfp4.json).

DFlash2 stays the separate BF16 draft `incoai/GLM-5.3-Flash-DFlash2` @ `dc77ff1c99eeb2df044ee3d4f0094eb033fee410`, 81 tensors, catalog sha256 `ccad9b633dc4090c50c6d1667778402cc634eb08bd7d87ac22e2abe209626b44`.

## What this does not prove

A shapes-checked bind does not show that a full 45-layer forward matches a BF16 reference. Packed decode is tested on a CPU fixture against the ModelOpt formula above.
