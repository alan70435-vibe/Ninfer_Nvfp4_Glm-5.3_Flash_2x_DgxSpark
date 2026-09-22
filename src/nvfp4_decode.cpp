#include "ninfer_glm53/nvfp4_decode.hpp"

#include <cmath>
#include <limits>

namespace ninfer::glm53 {
namespace {

constexpr float kE2M1[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};

}  // namespace

float e2m1(std::uint8_t nibble) {
    const float mag = kE2M1[nibble & 0x7u];
    return (nibble & 0x8u) != 0u ? -mag : mag;
}

float fp8_e4m3_to_f32(std::uint8_t bits) {
    const int sign = (bits >> 7) & 1;
    const int exp = (bits >> 3) & 0xf;
    const int mant = bits & 0x7;
    float magnitude = 0.f;
    if (exp == 0) {
        magnitude = std::ldexp(static_cast<float>(mant), -9);
    } else if (exp == 15 && mant == 7) {
        magnitude = std::numeric_limits<float>::quiet_NaN();
    } else {
        magnitude = std::ldexp(1.f + static_cast<float>(mant) / 8.f, exp - 7);
    }
    return std::copysign(magnitude, sign != 0 ? -1.f : 1.f);
}

void nvfp4_dequantize(const std::uint8_t* packed, const std::uint8_t* scales, float global_scale, int rows,
                      int cols, float* weight) {
    if (packed == nullptr || scales == nullptr || weight == nullptr || rows <= 0 || cols <= 0 || cols % 16 != 0) {
        return;
    }
    const int packed_cols = cols / 2;
    const int scale_cols = cols / 16;
    for (int row = 0; row < rows; ++row) {
        const std::uint8_t* packed_row = packed + row * packed_cols;
        const std::uint8_t* scale_row = scales + row * scale_cols;
        float* out = weight + row * cols;
        for (int col = 0; col < cols; ++col) {
            const std::uint8_t byte = packed_row[col / 2];
            const std::uint8_t nibble = (col % 2 == 0) ? static_cast<std::uint8_t>(byte & 0x0fu)
                                                       : static_cast<std::uint8_t>(byte >> 4);
            const float scale = fp8_e4m3_to_f32(scale_row[col / 16]) * global_scale;
            out[col] = e2m1(nibble) * scale;
        }
    }
}

void nvfp4_gemv(const float* x, const float* weight, int rows, int cols, float* y) {
    if (x == nullptr || weight == nullptr || y == nullptr || rows <= 0 || cols <= 0) return;
    for (int row = 0; row < rows; ++row) {
        float sum = 0.f;
        const float* w = weight + row * cols;
        for (int col = 0; col < cols; ++col) sum += w[col] * x[col];
        y[row] = sum;
    }
}

}  // namespace ninfer::glm53
