# Engineering contract for agents

## Product target

This repository serves GLM-5.3-Flash official ModelOpt NVFP4 on exactly two
NVIDIA DGX Spark / GB10 nodes with TP=2. The product checkpoint is
`nvidia/GLM-5.3-Flash-NVFP4` @ `09b04e5e74bca08ca8549fc736d4cdd8624bfde3`.
Do not retarget the hot path to EXL3 or to a compressed-tensors export.

The mathematical model contract is shared with
`Ninfer_Exl3_Glm-5.3_Flash_2x_DgxSpark`. Quantized storage is not shared.

## Correctness rules

- `include/ninfer_glm53/model_spec.hpp` defines the logical model topology.
- `parameter_schema` is the storage catalog for this checkpoint. Shapes there
  are safetensors shapes, not decoded GEMMs.
- `DeploymentContract` is deployment intent. Hardware preflight is observed fact.
- KDA state, sparse-MLA KV/index state, draft state and host sampler state stay
  separate. A speculative step commits on both ranks together.
- Never silently fall back to another quantization, attention backend, or world size.

## Checkpoint rules

A binder fails when a required tensor is missing, duplicated, or has the wrong
dtype or shape. `quant_method` must be `modelopt` with `quant_algo` NVFP4.
Dense MLP layers 0–2 and routed experts in layers 3–44 are NVFP4. Layer 45
routed experts are BF16 weights with no scale tensor.

## GPU implementation rules

Before a public operator is optimized, provide a shape contract, an oracle,
eager coverage, CUDA graph replay when the decode path uses it, boundary
shapes, compute-sanitizer coverage for new synchronization, and a benchmark of
the operator the model actually calls.

## Scope

Do not add vLLM as a runtime dependency. Do not add placeholder CUDA kernels to
advance the roadmap. Host contracts and tests come first.
