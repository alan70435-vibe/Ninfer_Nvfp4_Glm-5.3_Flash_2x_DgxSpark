#include "ninfer_glm53/nvfp4_decode.hpp"

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

bool near(float actual, float expected) { return std::fabs(actual - expected) < 1e-5f; }

}  // namespace

int main() {
    using namespace ninfer::glm53;
    expect(near(e2m1(0x1), 0.5f) && near(e2m1(0x2), 1.f) && near(e2m1(0x9), -0.5f), "e2m1 table");
    expect(near(fp8_e4m3_to_f32(0x38), 1.f) && near(fp8_e4m3_to_f32(0x40), 2.f), "fp8 e4m3 anchors");

    const std::uint8_t packed[] = {0x21, 0, 0, 0, 0, 0, 0, 0};
    const std::uint8_t scales[] = {0x38};
    float weight[16];
    nvfp4_dequantize(packed, scales, 1.f, 1, 16, weight);
    expect(near(weight[0], 0.5f) && near(weight[1], 1.f), "low nibble then high nibble");
    for (int i = 2; i < 16; ++i) expect(weight[i] == 0.f, "remaining nibbles are zero");

    float x[16] = {};
    x[0] = 1.f;
    x[1] = 1.f;
    float y[1];
    nvfp4_gemv(x, weight, 1, 16, y);
    expect(near(y[0], 1.5f), "nvfp4 gemv");

    nvfp4_dequantize(packed, scales, 2.f, 1, 16, weight);
    nvfp4_gemv(x, weight, 1, 16, y);
    expect(near(y[0], 0.75f), "global scale divides the block scale");

    std::vector<std::uint8_t> packed32(16, 0x22);
    const std::uint8_t scales2[] = {0x38, 0x40};
    std::vector<float> wide(32);
    nvfp4_dequantize(packed32.data(), scales2, 1.f, 1, 32, wide.data());
    expect(near(wide[0], 1.f) && near(wide[16], 2.f), "group of 16 selects the scale");
    std::vector<float> x32(32, 1.f);
    nvfp4_gemv(x32.data(), wide.data(), 1, 32, y);
    expect(near(y[0], 48.f), "two-group gemv");
    return 0;
}
