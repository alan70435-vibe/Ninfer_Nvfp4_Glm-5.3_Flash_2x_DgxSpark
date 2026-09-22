#include "ninfer_glm53/nvfp4_decode.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
// Independent IEEE-754 bit construction, deliberately not the decoder's ldexp.
float reference_fp8(std::uint8_t bits) {
    const auto exponent = static_cast<std::uint32_t>((bits >> 3) & 15);
    const auto mantissa = static_cast<std::uint32_t>(bits & 7);
    if (exponent == 15 && mantissa == 7) return std::numeric_limits<float>::quiet_NaN();
    const float magnitude = exponent == 0 ? static_cast<float>(mantissa) / 512.f
        : std::bit_cast<float>(((exponent + 120U) << 23) | (mantissa << 20));
    return (bits & 128U) != 0 ? -magnitude : magnitude;
}
}  // namespace

int main() {
    using namespace ninfer::glm53;
    using ninfer::test::throws;
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
    for (unsigned bits = 0; bits < 256; ++bits) {
        const auto input = static_cast<std::uint8_t>(bits);
        const float actual = fp8_e4m3_to_f32(input), expected = reference_fp8(input);
        CHECK(std::isnan(expected) ? std::isnan(actual) : actual == expected);
        if (actual == 0.f) CHECK(std::signbit(actual) == std::signbit(expected));
    }
    constexpr std::array<float, 8> magnitudes{0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    for (unsigned code = 0; code < 16; ++code) {
        const float expected = code < 8 ? magnitudes[code] : -magnitudes[code - 8];
        const float actual = e2m1(static_cast<std::uint8_t>(code));
        CHECK(actual == expected && std::signbit(actual) == std::signbit(expected));
    }

    constexpr std::size_t rows = 4, cols = 64;
    std::vector<std::uint8_t> packed(rows * cols / 2), scales(rows * cols / 16);
    for (std::size_t i = 0; i < packed.size(); ++i) packed[i] = static_cast<std::uint8_t>(i * 37U);
    constexpr std::array<std::uint8_t, 4> scale_codes{0x08, 0x30, 0x38, 0x40};
    for (std::size_t i = 0; i < scales.size(); ++i) scales[i] = scale_codes[i % 4];
    const Nvfp4MatrixView view(packed, scales, 2.f, rows, cols);
    std::vector<float> expanded(rows * cols), x(cols, 1.f), dense(rows), direct(rows);
    nvfp4_dequantize(view, expanded);
    nvfp4_gemv(x, expanded, rows, cols, dense);
    nvfp4_packed_gemv(view, x, direct);
    CHECK(dense == direct);
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            const auto byte = packed[r * cols / 2 + c / 2];
            const auto code = static_cast<unsigned>(c % 2 == 0 ? byte & 15U : byte >> 4);
            const float fp4 = code < 8 ? magnitudes[code] : -magnitudes[code - 8];
            CHECK(expanded[r * cols + c] == fp4 * reference_fp8(scales[r * cols / 16 + c / 16]) / 2.f);
        }
    }
    // TP2 column split: rank-local outputs concatenate to the unsplit output.
    std::vector<float> n0(2), n1(2);
    nvfp4_packed_gemv(view.slice(0, 2, 0, cols), x, n0);
    nvfp4_packed_gemv(view.slice(2, 2, 0, cols), x, n1);
    CHECK(std::equal(n0.begin(), n0.end(), dense.begin()));
    CHECK(std::equal(n1.begin(), n1.end(), dense.begin() + 2));
    // TP2 row split: keep strided rows and sum partial K contributions.
    std::vector<float> k0(rows), k1(rows);
    nvfp4_packed_gemv(view.slice(0, rows, 0, cols / 2), std::span<const float>(x).first(cols / 2), k0);
    nvfp4_packed_gemv(view.slice(0, rows, cols / 2, cols / 2), std::span<const float>(x).subspan(cols / 2), k1);
    for (std::size_t i = 0; i < rows; ++i) CHECK(k0[i] + k1[i] == dense[i]);
    const auto nested = view.slice(1, 3, 16, 48).slice(1, 2, 16, 32);
    for (std::size_t r = 0; r < 2; ++r)
        for (std::size_t c = 0; c < 32; ++c) CHECK(nested.at(r, c) == view.at(r + 2, c + 32));

    std::array<std::uint8_t, 8> p16{};
    std::array<std::uint8_t, 1> s16{0x30};
    std::array<float, 16> output;
    output.fill(123.f);
    auto reject = [&](float global, std::size_t r, std::size_t c) {
        CHECK(throws<std::invalid_argument>([&] { nvfp4_dequantize(p16, s16, global, r, c, output); }));
        CHECK(std::all_of(output.begin(), output.end(), [](float v) { return v == 123.f; }));
    };
    for (const auto c : {0U, 8U, 17U, 18U}) reject(1.f, 1, c);
    reject(1.f, 0, 16);
    for (const auto global : {0.f, -0.f, -1.f, std::numeric_limits<float>::infinity(),
                               std::numeric_limits<float>::quiet_NaN()}) reject(global, 1, 16);
    CHECK(throws<std::overflow_error>([&] {
        (void)Nvfp4MatrixView(p16, s16, 1.f, std::numeric_limits<std::size_t>::max(), 16);
    }));
    CHECK(throws<std::invalid_argument>([&] {
        nvfp4_dequantize(std::span<const std::uint8_t>(p16).first(7), s16, 1.f, 1, 16, output);
    }));
    CHECK(throws<std::invalid_argument>([&] { nvfp4_dequantize(p16, {}, 1.f, 1, 16, output); }));
    CHECK(throws<std::invalid_argument>([&] {
        nvfp4_dequantize(p16, s16, 1.f, 1, 16, std::span<float>(output).first(15));
    }));
    for (const auto bad_scale : {0x7fU, 0xffU, 0xb8U}) {
        s16[0] = static_cast<std::uint8_t>(bad_scale);
        reject(1.f, 1, 16);
    }
    s16[0] = 0x38;
    reject(std::numeric_limits<float>::denorm_min(), 1, 16);
    s16[0] = 0;
    nvfp4_dequantize(p16, s16, 1.f, 1, 16, output);  // Zero block scales are valid.
    CHECK(std::all_of(output.begin(), output.end(), [](float v) { return v == 0.f; }));

    CHECK(throws<std::out_of_range>([&] { (void)view.at(rows, 0); }));
    CHECK(throws<std::out_of_range>([&] { (void)view.slice(0, rows + 1, 0, 16); }));
    CHECK(throws<std::out_of_range>([&] { (void)view.slice(0, 1, 0, std::numeric_limits<std::size_t>::max()); }));
    CHECK(throws<std::invalid_argument>([&] { (void)view.slice(0, 1, 1, 16); }));
    CHECK(throws<std::invalid_argument>([&] { (void)view.slice(0, 1, 0, 17); }));
    CHECK(throws<std::invalid_argument>([&] { nvfp4_packed_gemv(view, {}, direct); }));
    CHECK(throws<std::invalid_argument>([&] { nvfp4_packed_gemv(view, x, {}); }));
    CHECK(throws<std::invalid_argument>([&] { nvfp4_gemv(x, {}, rows, cols, direct); }));
    CHECK(throws<std::overflow_error>([&] {
        nvfp4_gemv(x, expanded, std::numeric_limits<std::size_t>::max(), 2, direct);
    }));
    // Reject byte aliasing between expanded output and compressed source.
    std::array<float, 16> shared{};
    const auto bytes = std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(shared.data()), 8);
    CHECK(throws<std::invalid_argument>([&] { nvfp4_dequantize(bytes, s16, 1.f, 1, 16, shared); }));

    const Nvfp4MatrixView shared_view(bytes, s16, 1.f, 1, 16);
    CHECK(throws<std::invalid_argument>([&] {
        nvfp4_packed_gemv(shared_view, output, std::span<float>(shared).first(1));
    }));

    // Actual routed-expert [N,K] geometries, synthetic bytes only (no checkpoint).
    for (const auto shape : {std::array<std::size_t, 2>{2048, 4096}, {4096, 2048}}) {
        const auto n = shape[0], k = shape[1];
        const std::vector<std::uint8_t> p(n * k / 2, 0x22), s(n * k / 16, 0x30);
        const Nvfp4MatrixView actual_shape(p, s, 2.f, n, k);
        const std::vector<float> ones(k, 1.f);
        std::vector<float> y(n);
        nvfp4_packed_gemv(actual_shape, ones, y);
        CHECK(std::all_of(y.begin(), y.end(), [k](float v) { return v == static_cast<float>(k) / 4.f; }));
    }
    return 0;
}
