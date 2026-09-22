#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::glm53 {

// Pure format conversion. E4M3FN preserves signed zero and both NaN encodings.
[[nodiscard]] float e2m1(std::uint8_t nibble);
[[nodiscard]] float fp8_e4m3_to_f32(std::uint8_t bits);

// Non-owning ModelOpt NVFP4 view. Backing storage must remain alive and
// immutable for the lifetime of this view and its slices. Construction checks
// positive dimensions, K % 16, byte lengths, overflow, finite nonnegative
// block scales, and a finite positive weight_scale_2. Dequant is
// e2m1 * fp8_e4m3(block_scale) * weight_scale_2.
class Nvfp4MatrixView {
public:
    Nvfp4MatrixView(std::span<const std::uint8_t> packed,
                    std::span<const std::uint8_t> scales, float global_scale,
                    std::size_t rows, std::size_t cols);

    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
    [[nodiscard]] float at(std::size_t row, std::size_t col) const;
    // Slices retain the parent's row strides; K offsets and widths must be
    // group-aligned. No packed weights or scales are copied or expanded.
    [[nodiscard]] Nvfp4MatrixView slice(std::size_t row, std::size_t rows,
                                       std::size_t col, std::size_t cols) const;

private:
    struct Validated {};
    Nvfp4MatrixView(Validated, std::span<const std::uint8_t> packed,
                    std::span<const std::uint8_t> scales, float global_scale,
                    std::size_t rows, std::size_t cols,
                    std::size_t packed_stride, std::size_t scale_stride);
    std::span<const std::uint8_t> packed_;
    std::span<const std::uint8_t> scales_;
    float global_scale_;
    std::size_t rows_, cols_, packed_stride_, scale_stride_;
    friend void nvfp4_dequantize(const Nvfp4MatrixView&, std::span<float>);
    friend void nvfp4_packed_gemv(const Nvfp4MatrixView&, std::span<const float>, std::span<float>);
};

// All public buffer APIs carry lengths. Invalid metadata is rejected before
// output writes (invalid_argument / overflow_error); at/slice use out_of_range
// for invalid ranges. Expanded weights are CPU oracle scratch, not residency.
void nvfp4_dequantize(const Nvfp4MatrixView& view, std::span<float> weight);
void nvfp4_dequantize(std::span<const std::uint8_t> packed,
                      std::span<const std::uint8_t> scales, float global_scale,
                      std::size_t rows, std::size_t cols, std::span<float> weight);
void nvfp4_gemv(std::span<const float> x, std::span<const float> weight,
                std::size_t rows, std::size_t cols, std::span<float> y);
// CPU packed oracle: O(rows) output scratch, never an expanded [rows, cols]
// weight allocation. This is NOT a CUDA kernel or a device residency planner.
void nvfp4_packed_gemv(const Nvfp4MatrixView& view, std::span<const float> x,
                       std::span<float> y);

}  // namespace ninfer::glm53
