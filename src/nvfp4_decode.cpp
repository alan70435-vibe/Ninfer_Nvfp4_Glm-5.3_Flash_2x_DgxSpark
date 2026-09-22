#include "ninfer_glm53/nvfp4_decode.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ninfer::glm53 {
namespace {
constexpr float kE2M1[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};

std::size_t checked_mul(std::size_t a, std::size_t b) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::overflow_error("NVFP4 size overflow");
    return a * b;
}
std::size_t elements(std::size_t rows, std::size_t cols) {
    if (rows == 0 || cols == 0) throw std::invalid_argument("matrix dimensions must be positive");
    return checked_mul(rows, cols);
}
template <class T>
void require_size(std::span<T> buffer, std::size_t n) {
    if (buffer.data() == nullptr || buffer.size() < n)
        throw std::invalid_argument("matrix buffer is too short");
    (void)checked_mul(n, sizeof(T));
}
bool overlaps(std::span<const std::byte> a, std::span<const std::byte> b) {
    const auto pa = reinterpret_cast<std::uintptr_t>(a.data());
    const auto pb = reinterpret_cast<std::uintptr_t>(b.data());
    if (a.empty() || b.empty()) return false;
    // Subtract addresses instead of adding lengths (no address wraparound).
    return pa <= pb ? pb - pa < a.size() : pa - pb < b.size();
}
}  // namespace

float e2m1(std::uint8_t nibble) {
    const float magnitude = kE2M1[nibble & 0x7u];
    return (nibble & 0x8u) != 0u ? -magnitude : magnitude;
}

float fp8_e4m3_to_f32(std::uint8_t bits) {
    const int exponent = (bits >> 3) & 0xf;
    const int mantissa = bits & 0x7;
    float magnitude;
    if (exponent == 0) {
        magnitude = std::ldexp(static_cast<float>(mantissa), -9);
    } else if (exponent == 15 && mantissa == 7) {
        magnitude = std::numeric_limits<float>::quiet_NaN();
    } else {
        magnitude = std::ldexp(1.f + static_cast<float>(mantissa) / 8.f, exponent - 7);
    }
    return std::copysign(magnitude, (bits & 0x80u) != 0u ? -1.f : 1.f);
}

Nvfp4MatrixView::Nvfp4MatrixView(std::span<const std::uint8_t> packed,
                                 std::span<const std::uint8_t> scales,
                                 float global_scale, std::size_t rows, std::size_t cols)
    : packed_(packed), scales_(scales), global_scale_(global_scale),
      rows_(rows), cols_(cols), packed_stride_(cols / 2), scale_stride_(cols / 16) {
    (void)elements(rows, cols);
    if (cols % 16 != 0) throw std::invalid_argument("NVFP4 K must be a multiple of 16");
    const auto packed_size = checked_mul(rows, packed_stride_);
    const auto scale_size = checked_mul(rows, scale_stride_);
    require_size(packed, packed_size);
    require_size(scales, scale_size);
    if (!std::isfinite(global_scale) || global_scale <= 0.f)
        throw std::invalid_argument("NVFP4 global scale must be finite and positive");
    packed_ = packed.first(packed_size);
    scales_ = scales.first(scale_size);
    for (const auto bits : scales_) {
        const float scale = fp8_e4m3_to_f32(bits);
        if (!std::isfinite(scale) || scale < 0.f)
            throw std::invalid_argument("NVFP4 block scales must be finite and nonnegative");
        if (6.0 * static_cast<double>(scale) / static_cast<double>(global_scale) >
            static_cast<double>(std::numeric_limits<float>::max()))
            throw std::invalid_argument("NVFP4 dequantized scale overflows float");
    }
}

Nvfp4MatrixView::Nvfp4MatrixView(Validated, std::span<const std::uint8_t> packed,
                                 std::span<const std::uint8_t> scales, float global_scale,
                                 std::size_t rows, std::size_t cols,
                                 std::size_t packed_stride, std::size_t scale_stride)
    : packed_(packed), scales_(scales), global_scale_(global_scale),
      rows_(rows), cols_(cols), packed_stride_(packed_stride), scale_stride_(scale_stride) {}

float Nvfp4MatrixView::at(std::size_t row, std::size_t col) const {
    if (row >= rows_ || col >= cols_) throw std::out_of_range("NVFP4 element outside view");
    const auto byte = packed_[row * packed_stride_ + col / 2];
    const auto nibble = static_cast<std::uint8_t>(col % 2 == 0 ? byte & 0xfu : byte >> 4);
    const auto scale = fp8_e4m3_to_f32(scales_[row * scale_stride_ + col / 16]);
    // Double intermediate avoids a spurious overflow before the division.
    return static_cast<float>(static_cast<double>(e2m1(nibble)) *
                              static_cast<double>(scale) / static_cast<double>(global_scale_));
}

Nvfp4MatrixView Nvfp4MatrixView::slice(std::size_t row, std::size_t rows,
                                      std::size_t col, std::size_t cols) const {
    if (rows == 0 || cols == 0 || row >= rows_ || col >= cols_ ||
        rows > rows_ - row || cols > cols_ - col)
        throw std::out_of_range("NVFP4 slice outside view");
    if (col % 16 != 0 || cols % 16 != 0)
        throw std::invalid_argument("NVFP4 K slices must preserve groups of 16");
    const auto packed_offset = row * packed_stride_ + col / 2;
    const auto scale_offset = row * scale_stride_ + col / 16;
    // Every range is bounded by the validated parent, including nested slices.
    const auto packed_length = (rows - 1) * packed_stride_ + cols / 2;
    const auto scale_length = (rows - 1) * scale_stride_ + cols / 16;
    return Nvfp4MatrixView(Validated{}, packed_.subspan(packed_offset, packed_length),
                           scales_.subspan(scale_offset, scale_length), global_scale_,
                           rows, cols, packed_stride_, scale_stride_);
}

void nvfp4_dequantize(const Nvfp4MatrixView& view, std::span<float> weight) {
    const auto count = elements(view.rows(), view.cols());
    require_size(weight, count);
    const auto output = std::as_bytes(weight.first(count));
    if (overlaps(output, std::as_bytes(view.packed_)) ||
        overlaps(output, std::as_bytes(view.scales_)))
        throw std::invalid_argument("dequantization output overlaps packed storage");
    for (std::size_t row = 0; row < view.rows(); ++row)
        for (std::size_t col = 0; col < view.cols(); ++col)
            weight[row * view.cols() + col] = view.at(row, col);
}

void nvfp4_dequantize(std::span<const std::uint8_t> packed,
                      std::span<const std::uint8_t> scales, float global_scale,
                      std::size_t rows, std::size_t cols, std::span<float> weight) {
    nvfp4_dequantize(Nvfp4MatrixView(packed, scales, global_scale, rows, cols), weight);
}

void nvfp4_gemv(std::span<const float> x, std::span<const float> weight,
                std::size_t rows, std::size_t cols, std::span<float> y) {
    require_size(weight, elements(rows, cols));
    require_size(x, cols);
    require_size(y, rows);
    std::vector<float> result(rows);
    for (std::size_t row = 0; row < rows; ++row) {
        double sum = 0.0;
        for (std::size_t col = 0; col < cols; ++col)
            sum += static_cast<double>(weight[row * cols + col]) * static_cast<double>(x[col]);
        result[row] = static_cast<float>(sum);
    }
    std::copy(result.begin(), result.end(), y.begin());
}

void nvfp4_packed_gemv(const Nvfp4MatrixView& view, std::span<const float> x, std::span<float> y) {
    require_size(x, view.cols());
    require_size(y, view.rows());
    const auto output = std::as_bytes(y.first(view.rows()));
    if (overlaps(output, std::as_bytes(view.packed_)) || overlaps(output, std::as_bytes(view.scales_)))
        throw std::invalid_argument("GEMV output overlaps packed storage");
    std::vector<float> result(view.rows());
    for (std::size_t row = 0; row < view.rows(); ++row) {
        double sum = 0.0;
        for (std::size_t col = 0; col < view.cols(); ++col)
            sum += static_cast<double>(view.at(row, col)) * static_cast<double>(x[col]);
        result[row] = static_cast<float>(sum);
    }
    std::copy(result.begin(), result.end(), y.begin());
}

}  // namespace ninfer::glm53
