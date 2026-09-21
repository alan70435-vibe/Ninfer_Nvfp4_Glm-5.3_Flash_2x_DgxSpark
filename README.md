# NInfer × GLM-5.3-Flash NVFP4 × 2 DGX Spark

A purpose-built inference runtime for **GLM-5.3-Flash compressed-tensors NVFP4**
on exactly **2 × NVIDIA DGX Spark / GB10**, using tensor parallelism over the
ConnectX-7 200 Gb/s RoCE link.

The model contract, TP2 execution plan and DFlash2 protocol come from
[Ninfer_Exl3_Glm-5.3_Flash_2x_DgxSpark](https://github.com/alan70435/Ninfer_Exl3_Glm-5.3_Flash_2x_DgxSpark).
That repository stops at EXL3 checkpoint binding. This repository is the
runtime line: GB10's 4-bit path is NVFP4, so the kernels will be written
against this checkpoint instead of the EXL3 trellis.

## Status

Host contracts and the NVFP4 storage catalog are in place. CUDA kernels, NCCL
and the serving API are not.

The product checkpoint is `RedHatAI/GLM-5.3-Flash-NVFP4` at
`18d55bfd5a2194887738da73753975c9d3842f46`:

- layers 3–44 routed experts: NVFP4 packed weights, FP8 block scales, FP32 global scales;
- layer 45 routed experts: FP8 E4M3 weights with BF16 128×128 block scales;
- attention, dense FFN, router, shared experts, embeddings, `lm_head` and vision: native BF16/FP32.

Name coverage of that index is 148,498 / 148,498. Storage shapes were checked
against the shard-1 and `model_mtp.safetensors` headers. The weight shards
themselves are not in this repository. See
[`docs/nvfp4-binding.md`](docs/nvfp4-binding.md).

ModelOpt NVFP4 checkpoints are refused. They use different tensor suffixes and
have emitted corrupted token ids under vLLM.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure

./build/ninfer-glm53-plan configs/dgxspark_tp2.env.example
./build/ninfer-glm53-bind /path/to/RedHatAI-GLM-5.3-Flash-NVFP4 -o receipt.json
```

`--names-only` checks the index without opening shards. A complete receipt
requires the shard headers.

## Verification gates

Architecture verification for this runtime is:

1. one rank executes every GLM operator correctly;
2. TP=2 logits and persistent state match that rank;
3. DFlash2 at k=7 and temperature 0 matches target-only generation, including partial acceptance.

Those gates are still open. The order is in
[`docs/implementation-roadmap.md`](docs/implementation-roadmap.md).
