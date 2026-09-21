#pragma once

namespace ninfer::glm53 {

// y = x + W * RMSNorm(x). W is row-major [width, width].
void dense_residual_proj(const float* x, const float* norm_weight, const float* projection, float eps,
                         int width, float* y);

void rmsnorm(const float* x, const float* weight, float eps, int width, float* y);

}  // namespace ninfer::glm53
