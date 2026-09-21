# NVFP4 checkpoint binding

The product checkpoint is `RedHatAI/GLM-5.3-Flash-NVFP4` revision
`18d55bfd5a2194887738da73753975c9d3842f46`. `quant_method` is
`compressed-tensors`. Routed experts in layers 3–44 use
`nvfp4-pack-quantized`. Layer 45 routed experts use FP8 block quantization.
The saved tensor names are the released GLM namespace (`hc_attn_*`,
`self_attn.f_a_proj`, `self_attn.A_log`), including tensors stored in
`model_mtp.safetensors`.

## Measured storage

Shapes below were read from the safetensors headers of
`model-00001-of-00010.safetensors` and `model_mtp.safetensors`. Inside shard 1,
every normalized tensor family has one dtype and shape except `o_proj`, which
is `[4096, 8192]` on KDA layers and `[4096, 16384]` on sparse MLA layers.

A logical linear of shape `[N, K]` (out, in) is stored as:

| Region | Tensors | Dtype and shape |
|---|---|---|
| Layers 3–44 routed `gate`/`up` | `weight_packed`, `weight_scale`, `weight_global_scale`, `input_global_scale` | U8 `[N, K/2]`, F8_E4M3 `[N, K/16]`, F32 `[1]`, F32 `[1]` |
| Layers 3–44 routed `down` | same four suffixes | `N=4096`, `K=2048` |
| Layer 45 routed experts | `weight`, `weight_scale` | F8_E4M3 `[N, K]`, BF16 `[N/128, K/128]` |
| Everything else | native `.weight` / auxiliary tensors | BF16 or FP32, same geometry as the EXL3 catalog |

Gate and up use `N=2048`, `K=4096`. Example from layer 3, expert 0:
`weight_packed` is U8 `[2048, 2048]` and `weight_scale` is F8_E4M3 `[2048, 256]`.
Layer 45 expert 0 gate is F8_E4M3 `[2048, 4096]` with BF16 scale `[16, 32]`.

The index contains 148,498 tensors: 1,618 native, 145,152 NVFP4 storage
tensors, and 1,728 FP8-block tensors. A names-only bind of revision
`18d55bfd5a2194887738da73753975c9d3842f46` matched every name:
config sha256 `29c9f4171196910e99b9c069d6b76c56e3cdcd0f436dc1bacbc9513c9a7529ac`,
index sha256 `015faae91e8189c7553f1d48ec3d0694b8c02b282d7f58af2d7b4064a81ce4c0`.
The logical catalog sha256 is
`674ff493a2911b72d94ddda1eaedc742b6895a0b1235d39656b765516e1971e8`.
The receipt is [`receipts/redhat-nvfp4-names.json`](receipts/redhat-nvfp4-names.json).
It records `shapes_checked: false` because the weight shards were not opened.

DFlash2 stays the separate BF16 draft
`incoai/GLM-5.3-Flash-DFlash2` @ `dc77ff1c99eeb2df044ee3d4f0094eb033fee410`,
81 tensors, unchanged from the EXL3 line.

## What this does not prove

Header shapes do not decode FP4 values and do not show that a GEMM matches
BF16. The full weight shards were not loaded. ModelOpt exports
(`weight`, `weight_scale`, `weight_scale_2`, `input_scale`, with scalar
scales) are a different container and are rejected by the binder.
