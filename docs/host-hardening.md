# Host correctness and packed-view increment

This increment addresses the CPU-side defects in #1, #2, #10 and #11, improves
#5's test validity, and starts the packed-view part of #6. It does **not**
implement CUDA execution, a full checkpoint loader, NCCL, long-context
allocation, distributed speculative transactions, or serving.

## Numerical oracle and buffer API

`fp8_e4m3_to_f32` is an E4M3FN format decoder: it uses a floating-point exponent
operation (no negative integer shifts), preserves signed zero, and returns NaN
for 0x7f and 0xff. Format decoding and weight-scale validation are separate.

`Nvfp4MatrixView` consumes the existing compressed-tensors storage convention:
low nibble first, one E4M3 scale per 16 K elements, and
`E2M1 * block_scale / weight_global_scale`. It rejects nonpositive dimensions,
misaligned K, insufficient buffers, size overflow, nonfinite/negative block
scales, and nonfinite/nonpositive global scales. Zero block scales are allowed.
Scales that would overflow the float oracle are rejected, not silently changed.

The view does not own its buffers. They must remain alive and immutable until
all views and slices are destroyed. This is a CPU metadata/view contract, not a
claim that arbitrary pointers or forged span lengths can be made memory-safe.
All checks complete before dequantization starts writing output.

The pointer-only NVFP4 APIs have been replaced with length-bearing spans.
Arrays and vectors pass directly:

```cpp
Nvfp4MatrixView weights(packed_bytes, block_scales, global_scale, n, k);
std::vector<float> y(n);
nvfp4_packed_gemv(weights, x, y);
// Small/reference fixtures only:
std::vector<float> expanded(n * k);
nvfp4_dequantize(weights, expanded);
```

Callers previously passing `.data()` must now pass a vector, array, or an
explicit span with its actual length. Repository NVFP4 and tiny-layer tests
have been migrated. Invalid inputs throw; no zero-global-scale fallback exists.

## Packed slices, not a completed TP2 runtime

`slice(row_offset, row_count, col_offset, col_count)` retains the parent's
packed/scales row strides and requires group-aligned K offsets and widths.
Nested, noncontiguous row-major submatrices are supported without copying or
expanding resident weights. `nvfp4_packed_gemv` uses only O(N) output scratch.

The synthetic tests verify N-split output concatenation and K-split partial
output summation against the unsplit CPU result. They also exercise routed
expert shapes [2048,4096] and [4096,2048]. They do not use real checkpoint bytes,
activation quantization, GPU tensor cores, or communication collectives.
A rank-local allocation/admission plan and all of #6's 1M-context accounting
remain open. Do not replace packed resident weights with the expanded oracle.

## Validated model and execution inputs

All eleven geometry fields reported in #10 are now checked against the fixed
model contract; tests mutate each to zero and to an incorrect nonzero value.
The plan builder rejects invalid model/execution inputs before dividing heads.
Execution validation compares local heads with the checked global/TP quotient,
not a wrapping 32-bit multiplication.

Only explicit `none` and `dflash` modes are accepted. Enabled plans preserve and
validate draft identity, revision, proposal count, and TP against deployment.
A 40-hex revision is required; this is syntactic pinning, not provenance or a
check that a remote model exists. The existing host helper's 1..8 proposal
range remains available for explicit experiments; the product baseline is k=7.
Disabled plans are canonical: draft identity and geometry are cleared even if
the input environment retains stale draft settings. #4's checkpoint pin and
provenance admission is a separate, still-open task.

## Always-on tests and sanitizers

The three contract tests now use `CHECK`, which evaluates its expression once
regardless of `NDEBUG`. `test_support_rejects_false` intentionally exits nonzero;
CTest's `WILL_FAIL` means it fails the suite if the false check ever passes.
Other tests continue to use their existing always-on `expect` checks.

Full repository commands (also configured in host CI):

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --parallel 2
ctest --test-dir build-release --output-on-failure
cmake -S . -B build-sanitizers -DCMAKE_BUILD_TYPE=Debug \
  -DNINFER_GLM53_ENABLE_SANITIZERS=ON
cmake --build build-sanitizers --parallel 2
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  ctest --test-dir build-sanitizers --output-on-failure
```

CI now has Debug, Release, and ASan/UBSan configurations and prints the
available CTest log even on failure. Python/shell syntax checks remain labelled
as syntax checks; they are not checkpoint or hardware behavior validation.
The previous hosted Actions no-steps/no-logs failure has not been diagnosed by
this code change. A green hosted run must be observed separately.

## Evidence from this editing environment

Base: `5069f43ea3ffb4c896d7196649ab0f1064f766c8`. Materialized original source
files were checked against their Git blob hashes before editing. In the local
x86_64/GCC 14.2.0 environment, eight selected test executables passed in Debug,
Release (`-O3 -DNDEBUG`), and ASan+UBSan configurations:

`model_spec`, `execution_plan`, `contract_regressions`, `nvfp4_decode`,
`nvfp4_view`, `tiny_layer`, `speculative`, and `test_support`.

The deliberately false CHECK exited with status 1 in all three configurations.
The migrated deployment-contract test compiled in all three but was not linked
or executed here. Sanitizer executables used `-fno-pie -no-pie` in this container.
This was selected-source compilation, **not a full repository CMake/CTest run**.
No full checkpoint, GB10, NCCL, distributed DFlash2, 1M context, or performance
validation was performed. #3/#4's checkpoint admission and #7/#8's execution
and state management are unchanged.

Format reference: NVIDIA CUDA Math API, `__nv_fp8_e4m3`:
https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__fp8__e4m3.html
