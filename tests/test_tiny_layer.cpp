#include "ninfer_glm53/nvfp4_decode.hpp"
#include "ninfer_glm53/tiny_layer.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

}  // namespace

int main() {
    using namespace ninfer::glm53;
    constexpr int width = 16;
    std::vector<std::uint8_t> packed(width * width / 2, 0);
    packed[0] = 0x21;
    std::vector<std::uint8_t> scales(width, 0);
    scales[0] = 0x38;
    std::vector<float> weight(width * width);
    nvfp4_dequantize(packed.data(), scales.data(), 1.f, width, width, weight.data());

    float x[width];
    float norm_w[width];
    for (int i = 0; i < width; ++i) {
        x[i] = 0.25f * static_cast<float>(i + 1);
        norm_w[i] = 1.f;
    }
    float y[width];
    dense_residual_proj(x, norm_w, weight.data(), 1e-5f, width, y);

    float normed[width];
    rmsnorm(x, norm_w, 1e-5f, width, normed);
    const float expected0 = x[0] + weight[0] * normed[0] + weight[1] * normed[1];
    expect(std::fabs(y[0] - expected0) < 1e-5f, "nvfp4 residual projection");
    expect(std::fabs(weight[0] - 0.5f) < 1e-5f && std::fabs(weight[1] - 1.f) < 1e-5f, "decoded first pair");
    return 0;
}
