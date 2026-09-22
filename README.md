# NInfer × GLM-5.3-Flash NVFP4 × 2 DGX Spark

A purpose-built inference runtime for **GLM-5.3-Flash official ModelOpt NVFP4**
on exactly **2 × NVIDIA DGX Spark / GB10**, using tensor parallelism over the
ConnectX-7 200 Gb/s RoCE link.

The model contract, TP2 execution plan and DFlash2 protocol come from
[Ninfer_Exl3_Glm-5.3_Flash_2x_DgxSpark](https://github.com/alan70435/Ninfer_Exl3_Glm-5.3_Flash_2x_DgxSpark).
That repository keeps the EXL3 trellis line. This repository is the NVFP4
line: GB10 Blackwell's 4-bit tensor-core path is NVFP4, so those kernels
will be written against this checkpoint. The construction record is
[`docs/build-log.md`](docs/build-log.md).

## Status

Host contracts, the ModelOpt storage catalog, and a CPU eager text forward are in place.
`ninfer-glm53-generate` reads the local checkpoint and emits greedy token ids.
CUDA kernels, NCCL and the serving API are not.

The product checkpoint is `nvidia/GLM-5.3-Flash-NVFP4` at
`09b04e5e74bca08ca8549fc736d4cdd8624bfde3`:

- dense MLP layers 0–2 and routed experts in layers 3–44: ModelOpt NVFP4
  (`.weight`, `.weight_scale`, `.weight_scale_2`, `.input_scale`);
- layer 45 routed experts: BF16 `.weight` only;
- attention, router, shared experts, embeddings, `lm_head` and vision: native BF16/FP32.

The logical catalog is 147,661 tensors. See
[`docs/nvfp4-binding.md`](docs/nvfp4-binding.md). Compressed-tensors exports are refused.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure

./build/ninfer-glm53-plan configs/dgxspark_tp2.env.example
./build/ninfer-glm53-bind /path/to/GLM-5.3-Flash-NVFP4 -o receipt.json
```

`--names-only` checks the index without opening shards. A complete receipt
requires the shard headers.

## Verification gates

Architecture verification for this runtime is:

1. one rank executes every GLM operator correctly;
2. TP=2 logits and persistent state match that rank;
3. DFlash2 at k=7 and temperature 0 matches target-only generation, including partial acceptance.

Those gates are still open. The order is in
[`docs/implementation-roadmap.md`](docs/implementation-roadmap.md). What has
already been built, and what the CPU forward did not match, is in
[`docs/build-log.md`](docs/build-log.md).
