# Implementation roadmap

The model math, TP2 boundaries and DFlash2 commit rules are already specified.
This roadmap starts at NVFP4 storage. EXL3 kernel work stays in
`Ninfer_Exl3_Glm-5.3_Flash_2x_DgxSpark`.

## P0 — host contracts

**Implemented**, carried over from the EXL3 line and retargeted:

- 45-layer GLM-5.3-Flash `ModelSpec`;
- two-node deployment contract pinned to official ModelOpt NVFP4;
- TP2 execution plan with mHC all-reduce boundaries;
- KDA versus sparse-MLA persistent state;
- DFlash2 k=7 plan fields;
- read-only DGX Spark preflight and host CI.

## P1 — NVFP4 catalog

**Implemented**

- logical catalog for native tensors, ModelOpt NVFP4 on dense MLP layers 0–2
  and routed experts in layers 3–44, BF16 MTP experts, the vision tower, and
  the DFlash2 draft;
- binder that requires `quant_method: modelopt` and `quant_algo: NVFP4`;
- shapes checked against the local official shards.

A CPU eager forward now emits token ids from this catalog. It does not yet
match an external Blackwell continuation. See `docs/build-log.md`.

## P2 — packed views and memory plan

1. Define the byte layout of packed `.weight` and the FP8 block scales, with a
   slice round-trip against the ModelOpt decoder.
2. Keep packed bytes packed. Do not expand the resident expert weights to BF16.
3. Account for two 121 GiB nodes: target, draft, graphs, KDA state, and the
   1M-token KV budget at fp8.

**Exit:** both ranks can place the checkpoint into rank-local views and report
the byte budget, without running the model.

## P3 — one-rank operators

Order: norms and embeddings, NVFP4 linear for the measured `[N, K]`, dense FFN,
KDA, sparse MLA and indexer, FP8 layer-45 experts, routed MoE top-8, then
graph-safe decode. Each public operator needs an oracle, eager and graph
tests, and compute-sanitizer coverage before tuning.

**Exit:** one GB10 rank matches a reference on synthetic and checkpoint fixtures.

## P4 — TP2 on the CX-7 link

One process per Spark, the execution-plan collectives, and identical sequence
state on both ranks.

**Exit:** TP2 logits match the one-rank reference within the NVFP4 tolerance.

## P5 — long context state

KDA recurrent state, paged sparse-MLA fp8 KV, and indexer metadata commit and
roll back together. Admission uses allocated bytes.

## P6 — DFlash2 k=7

Bind the pinned draft, verify blocks on the target, and commit accepted tokens
on both ranks. Temperature-0 output matches target-only execution.

## P7 — API

Single-inflight scheduler, rank-0 sampler state, and an OpenAI-compatible
stream. This is the first serving surface.

## P8 — parity and tuning

Compare token quality and speed with the current vLLM NVFP4 baseline on the
same checkpoint and prompts. Tuning follows those measurements.
