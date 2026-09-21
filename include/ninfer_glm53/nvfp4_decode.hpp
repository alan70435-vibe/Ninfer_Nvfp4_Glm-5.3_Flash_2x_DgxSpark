#pragma once

#include <cstdint>

namespace ninfer::glm53 {

// E2M1 magnitude table, sign in bit 3. Low nibble of each byte is the even K element.
[[nodiscard]] float e2m1(std::uint8_t nibble);
[[nodiscard]] float fp8_e4m3_to_f32(std::uint8_t bits);

// weight is row-major [rows, cols]. packed is [rows, cols/2], scales are [rows, cols/16] FP8 E4M3.
// Dequant matches compressed-tensors: e2m1 * fp8_scale / global_scale.
void nvfp4_dequantize(const std::uint8_t* packed, const std::uint8_t* scales, float global_scale, int rows,
                      int cols, float* weight);

// y[row] = sum_col weight[row, col] * x[col]
void nvfp4_gemv(const float* x, const float* weight, int rows, int cols, float* y);

}  // namespace ninfer::glm53
