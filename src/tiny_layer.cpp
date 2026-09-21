#include "ninfer_glm53/tiny_layer.hpp"

#include <cmath>

namespace ninfer::glm53 {

void rmsnorm(const float* x, const float* weight, float eps, int width, float* y) {
    float square = 0.f;
    for (int i = 0; i < width; ++i) square += x[i] * x[i];
    const float scale = 1.f / std::sqrt(square / static_cast<float>(width) + eps);
    for (int i = 0; i < width; ++i) y[i] = x[i] * scale * (weight != nullptr ? weight[i] : 1.f);
}

void dense_residual_proj(const float* x, const float* norm_weight, const float* projection, float eps,
                         int width, float* y) {
    float normed[64];
    if (width > 64) return;
    rmsnorm(x, norm_weight, eps, width, normed);
    for (int row = 0; row < width; ++row) {
        float sum = 0.f;
        const float* w = projection + row * width;
        for (int col = 0; col < width; ++col) sum += w[col] * normed[col];
        y[row] = x[row] + sum;
    }
}

}  // namespace ninfer::glm53
