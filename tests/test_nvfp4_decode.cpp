#include "ninfer_glm53/nvfp4_decode.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
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
    expect(near(fp8_e4m3_to_f32(0x08), 0.015625f), "fp8 smallest normal");
    expect(near(fp8_e4m3_to_f32(0x30), 0.5f), "fp8 half");
    expect(std::isnan(fp8_e4m3_to_f32(0x7f)) && std::isnan(fp8_e4m3_to_f32(0xff)), "fp8 nan stays nan");
    expect(fp8_e4m3_to_f32(0x80) == 0.f && std::signbit(fp8_e4m3_to_f32(0x80)), "fp8 negative zero");
    for (int code = 0; code < 256; ++code) {
        const auto bits = static_cast<std::uint8_t>(code);
        const int exp = (bits >> 3) & 0xf;
        const int mant = bits & 0x7;
        const float decoded = fp8_e4m3_to_f32(bits);
        if (exp == 15 && mant == 7) {
            expect(std::isnan(decoded), "every nan encoding");
            continue;
        }
        const float magnitude = exp == 0 ? std::ldexp(static_cast<float>(mant), -9)
                                         : std::ldexp(1.f + static_cast<float>(mant) / 8.f, exp - 7);
        const float expected = (bits & 0x80u) != 0u ? -magnitude : magnitude;
        expect(decoded == expected || (decoded == 0.f && expected == 0.f), "fp8 code");
    }
    float untouched = 7.f;
    bool rejected = false;
    try {
        nvfp4_dequantize(std::span<const std::uint8_t>{}, std::span<const std::uint8_t>{}, 1.f, 1, 17,
                         std::span<float>(&untouched, 1));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    expect(rejected && untouched == 7.f, "illegal width does not write");

    const std::uint8_t packed[] = {0x21, 0, 0, 0, 0, 0, 0, 0};
    const std::uint8_t scales[] = {0x38};
    float weight[16];
    nvfp4_dequantize(std::span<const std::uint8_t>(packed, 8), std::span<const std::uint8_t>(scales, 1), 1.f, 1, 16,
                     std::span<float>(weight, 16));
    expect(near(weight[0], 0.5f) && near(weight[1], 1.f), "low nibble then high nibble");
    for (int i = 2; i < 16; ++i) expect(weight[i] == 0.f, "remaining nibbles are zero");

    float x[16] = {};
    x[0] = 1.f;
    x[1] = 1.f;
    float y[1];
    nvfp4_gemv(std::span<const float>(x, 16), std::span<const float>(weight, 16), 1, 16, std::span<float>(y, 1));
    expect(near(y[0], 1.5f), "nvfp4 gemv");

    nvfp4_dequantize(std::span<const std::uint8_t>(packed, 8), std::span<const std::uint8_t>(scales, 1), 2.f, 1, 16,
                     std::span<float>(weight, 16));
    nvfp4_gemv(std::span<const float>(x, 16), std::span<const float>(weight, 16), 1, 16, std::span<float>(y, 1));
    expect(near(y[0], 3.f), "weight_scale_2 multiplies the block scale");

    std::vector<std::uint8_t> packed32(16, 0x22);
    const std::uint8_t scales2[] = {0x38, 0x40};
    std::vector<float> wide(32);
    nvfp4_dequantize(std::span<const std::uint8_t>(packed32), std::span<const std::uint8_t>(scales2, 2), 1.f, 1, 32,
                     std::span<float>(wide));
    expect(near(wide[0], 1.f) && near(wide[16], 2.f), "group of 16 selects the scale");
    std::vector<float> x32(32, 1.f);
    nvfp4_gemv(std::span<const float>(x32), std::span<const float>(wide), 1, 32, std::span<float>(y, 1));
    expect(near(y[0], 48.f), "two-group gemv");
    return 0;
}
