# Engineering contract for agents

## Product target

This repository serves GLM-5.3-Flash compressed-tensors NVFP4 on exactly two
NVIDIA DGX Spark / GB10 nodes with TP=2. The product checkpoint is
`RedHatAI/GLM-5.3-Flash-NVFP4`. Do not retarget the hot path to EXL3 or to a
ModelOpt NVFP4 export.

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
dtype or shape. `quant_method: modelopt` is a hard error. Layer 45 routed
experts are FP8 block storage, not NVFP4.

## GPU implementation rules

Before a public operator is optimized, provide a shape contract, an oracle,
eager coverage, CUDA graph replay when the decode path uses it, boundary
shapes, compute-sanitizer coverage for new synchronization, and a benchmark of
the operator the model actually calls.

## Scope

Do not add vLLM as a runtime dependency. Do not add placeholder CUDA kernels to
advance the roadmap. Host contracts and tests come first.
