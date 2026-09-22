#include "ninfer_glm53/text_forward.hpp"

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

bool near(float actual, float expected) { return std::fabs(actual - expected) < 1e-4f; }

float reference_silu(float x) {
    if (x >= 0.f) {
        const float z = std::exp(-x);
        return x / (1.f + z);
    }
    const float z = std::exp(x);
    return x * z / (1.f + z);
}

void reference_conv(const float* x, const float* weight, float* mem, int channels, int kernel, float* y) {
    const int history = kernel - 1;
    for (int channel = 0; channel < channels; ++channel) {
        const float* taps = weight + channel * kernel;
        const float raw = x[channel];
        float acc = 0.f;
        if (history > 0) {
            const float* state = mem + channel * history;
            for (int tap = 0; tap < history; ++tap) acc += taps[tap] * state[tap];
        }
        acc += taps[history] * raw;
        y[channel] = reference_silu(acc);
        if (history <= 0) continue;
        float* state = mem + channel * history;
        for (int tap = 0; tap < history - 1; ++tap) state[tap] = state[tap + 1];
        state[history - 1] = raw;
    }
}

bool same_span(const float* left, const float* right, int count) {
    for (int index = 0; index < count; ++index) {
        if (!near(left[index], right[index])) return false;
    }
    return true;
}

int count_open_fds() {
    DIR* directory = ::opendir("/proc/self/fd");
    if (directory == nullptr) {
        std::cerr << "cannot scan fds\n";
        std::exit(1);
    }
    const int scanner = ::dirfd(directory);
    int count = 0;
    while (const dirent* entry = ::readdir(directory)) {
        if (entry->d_name[0] == '.') continue;
        char* end = nullptr;
        const long value = std::strtol(entry->d_name, &end, 10);
        if (end == entry->d_name || *end != '\0') continue;
        if (value == scanner) continue;
        ++count;
    }
    if (::closedir(directory) != 0) {
        std::cerr << "cannot close fd scan\n";
        std::exit(1);
    }
    return count;
}

bool maps_contain(const std::filesystem::path& needle) {
    std::ifstream maps("/proc/self/maps");
    const std::string text = needle.string();
    std::string line;
    while (std::getline(maps, line)) {
        if (line.find(text) != std::string::npos) return true;
    }
    return false;
}

std::filesystem::path make_temp_dir(const char* prefix) {
    const auto pattern = (std::filesystem::temp_directory_path() / (std::string(prefix) + "XXXXXX")).string();
    std::vector<char> storage(pattern.begin(), pattern.end());
    storage.push_back('\0');
    if (::mkdtemp(storage.data()) == nullptr) {
        std::cerr << "mkdtemp failed\n";
        std::exit(1);
    }
    return std::filesystem::path(storage.data());
}

void test_causal_conv_alias() {
    using ninfer::glm53::causal_conv_silu;
    const float taps[] = {0.f, 0.f, 1.f, 2.f};
    const float inputs[] = {3.f, 1.f, -2.f, 4.f};
    float mem_in[3] = {};
    float mem_out[3] = {};
    for (int step = 0; step < 4; ++step) {
        float out = 0.f;
        float inplace = inputs[step];
        causal_conv_silu(&inputs[step], taps, mem_out, 1, 4, &out);
        causal_conv_silu(&inplace, taps, mem_in, 1, 4, &inplace);
        expect(near(out, inplace), "in-place conv matches out-of-place");
        expect(same_span(mem_in, mem_out, 3), "in-place history matches");
        if (step == 0) {
            expect(near(inplace, 5.985164f), "first in-place output is silu(6)");
            expect(mem_in[2] == 3.f, "history tail stores 3, not SiLU");
        }
        if (step == 1) {
            expect(near(inplace, 4.966536f), "second in-place output");
            expect(std::fabs(inplace - 7.982f) > 0.5f, "broken in-place history must not return");
        }
    }

    const float pair_taps[] = {0.f, 0.f, 1.f, 1.f};
    const float pair_inputs[] = {2.f, 0.f, -1.f, 3.f, 0.f};
    float pair_in[3] = {};
    float pair_out[3] = {};
    for (int step = 0; step < 5; ++step) {
        float out = 0.f;
        float inplace = pair_inputs[step];
        causal_conv_silu(&pair_inputs[step], pair_taps, pair_out, 1, 4, &out);
        causal_conv_silu(&inplace, pair_taps, pair_in, 1, 4, &inplace);
        expect(near(out, inplace) && same_span(pair_in, pair_out, 3), "x[t]+x[t-1] in-place matches");
        if (step == 0) {
            expect(near(inplace, 1.761594f), "first SiLU(x[t]+x[t-1])");
            expect(pair_in[2] == 2.f, "history tail stores 2");
        }
        if (step == 1) expect(near(inplace, 1.761594f), "second SiLU(x[t]+x[t-1])");
    }

    float kernel1_mem[4] = {9.f, 8.f, 7.f, 6.f};
    const float kernel1_saved[4] = {9.f, 8.f, 7.f, 6.f};
    const float kernel1_weight[] = {2.f, -3.f};
    const float kernel1_raw[] = {1.5f, -2.f};
    float kernel1_out[2] = {};
    float kernel1_in[2] = {1.5f, -2.f};
    float kernel1_in_mem[4] = {9.f, 8.f, 7.f, 6.f};
    causal_conv_silu(kernel1_raw, kernel1_weight, kernel1_mem, 2, 1, kernel1_out);
    causal_conv_silu(kernel1_in, kernel1_weight, kernel1_in_mem, 2, 1, kernel1_in);
    expect(same_span(kernel1_mem, kernel1_saved, 4), "kernel 1 writes no history");
    expect(same_span(kernel1_in_mem, kernel1_saved, 4), "in-place kernel 1 writes no history");
    expect(near(kernel1_out[0], reference_silu(3.f)) && near(kernel1_out[1], reference_silu(6.f)), "kernel 1 values");
    expect(same_span(kernel1_in, kernel1_out, 2), "kernel 1 in-place matches");

    const float wide_weight[] = {0.5f, -1.f, -2.f, 0.5f};
    const float wide_inputs[8][2] = {
        {2.f, -4.f}, {-1.f, 0.5f}, {3.f, -2.f}, {0.f, 1.f}, {-3.f, -0.5f}, {4.f, 2.f}, {-2.f, 3.f}, {1.f, -1.f},
    };
    float mem_ref[2] = {4.f, -3.f};
    float mem_ship[2] = {4.f, -3.f};
    float mem_alias[2] = {4.f, -3.f};
    for (int step = 0; step < 8; ++step) {
        const float raw[2] = {wide_inputs[step][0], wide_inputs[step][1]};
        float y_ref[2] = {};
        float y_ship[2] = {};
        float y_alias[2] = {raw[0], raw[1]};
        reference_conv(raw, wide_weight, mem_ref, 2, 2, y_ref);
        causal_conv_silu(raw, wide_weight, mem_ship, 2, 2, y_ship);
        causal_conv_silu(y_alias, wide_weight, mem_alias, 2, 2, y_alias);
        expect(same_span(y_ship, y_ref, 2) && same_span(y_alias, y_ref, 2), "kernel 2 signed steps");
        expect(same_span(mem_ship, mem_ref, 2) && same_span(mem_alias, mem_ref, 2), "kernel 2 history");
    }

    float once_raw[2] = {-1.5f, 2.5f};
    float once_copy[2] = {-1.5f, 2.5f};
    float once_out[2] = {};
    float once_mem_a[2] = {0.25f, -4.f};
    float once_mem_b[2] = {0.25f, -4.f};
    const float once_weight[] = {-0.5f, 1.5f, 2.f, -1.f};
    causal_conv_silu(once_raw, once_weight, once_mem_a, 2, 2, once_raw);
    causal_conv_silu(once_copy, once_weight, once_mem_b, 2, 2, once_out);
    expect(same_span(once_raw, once_out, 2) && same_span(once_mem_a, once_mem_b, 2), "one multi-channel in-place call");

    float overlap[6] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    float overlap_saved[6];
    std::memcpy(overlap_saved, overlap, sizeof overlap);
    float overlap_mem[3] = {7.f, 8.f, 9.f};
    float overlap_mem_saved[3];
    std::memcpy(overlap_mem_saved, overlap_mem, sizeof overlap_mem);
    const float overlap_weight[6] = {};
    bool overlap_threw = false;
    try {
        causal_conv_silu(overlap, overlap_weight, overlap_mem, 3, 2, overlap + 1);
    } catch (const std::runtime_error& error) {
        overlap_threw = std::string(error.what()) == "causal conv partial overlap";
    }
    expect(overlap_threw, "partial overlap throws");
    expect(std::memcmp(overlap, overlap_saved, sizeof overlap) == 0, "partial overlap leaves x unchanged");
    expect(std::memcmp(overlap_mem, overlap_mem_saved, sizeof overlap_mem) == 0, "partial overlap leaves mem unchanged");

    float reverse[6] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    float reverse_saved[6];
    std::memcpy(reverse_saved, reverse, sizeof reverse);
    float reverse_mem[3] = {1.f, 1.f, 1.f};
    float reverse_mem_saved[3] = {1.f, 1.f, 1.f};
    bool reverse_threw = false;
    try {
        causal_conv_silu(reverse + 1, overlap_weight, reverse_mem, 3, 2, reverse);
    } catch (const std::runtime_error& error) {
        reverse_threw = std::string(error.what()) == "causal conv partial overlap";
    }
    expect(reverse_threw, "reverse partial overlap throws");
    expect(std::memcmp(reverse, reverse_saved, sizeof reverse) == 0, "reverse overlap leaves x unchanged");
    expect(std::memcmp(reverse_mem, reverse_mem_saved, sizeof reverse_mem) == 0, "reverse overlap leaves mem unchanged");

    float replay_mem[3] = {0.5f, -1.f, 2.f};
    float replay_copy[3];
    std::memcpy(replay_copy, replay_mem, sizeof replay_mem);
    const float replay_taps[] = {0.f, 1.f, -1.f, 0.5f};
    const float replay_steps[] = {1.f, -2.f, 0.25f};
    float replay_out[3] = {};
    for (int step = 0; step < 3; ++step) {
        causal_conv_silu(&replay_steps[step], replay_taps, replay_mem, 1, 4, &replay_out[step]);
    }
    float replay_hist[3];
    std::memcpy(replay_hist, replay_mem, sizeof replay_mem);
    std::memcpy(replay_mem, replay_copy, sizeof replay_mem);
    float replay_again[3] = {};
    for (int step = 0; step < 3; ++step) {
        float inplace = replay_steps[step];
        causal_conv_silu(&inplace, replay_taps, replay_mem, 1, 4, &inplace);
        replay_again[step] = inplace;
    }
    expect(same_span(replay_again, replay_out, 3), "restored history replays outputs");
    expect(same_span(replay_mem, replay_hist, 3), "restored history replays state");

    float bad_mem[2] = {1.f, 2.f};
    float bad_x = 3.f;
    float bad_y = 4.f;
    bool kernel_threw = false;
    try {
        causal_conv_silu(&bad_x, &bad_x, bad_mem, 1, 0, &bad_y);
    } catch (const std::runtime_error&) {
        kernel_threw = true;
    }
    expect(kernel_threw, "kernel 0 throws");
    expect(bad_mem[0] == 1.f && bad_mem[1] == 2.f && bad_x == 3.f && bad_y == 4.f, "kernel 0 writes nothing");
    kernel_threw = false;
    try {
        causal_conv_silu(&bad_x, &bad_x, bad_mem, 1, -2, &bad_y);
    } catch (const std::runtime_error&) {
        kernel_threw = true;
    }
    expect(kernel_threw, "negative kernel throws");
}

void test_store_and_tp2_release() {
    using ninfer::glm53::generate_text;
    const auto store_dir = make_temp_dir("ninfer-nvfp4-store-");
    {
        std::ofstream shard(store_dir / "model.safetensors", std::ios::binary);
        const char zeros[8] = {};
        shard.write(zeros, sizeof zeros);
        expect(static_cast<bool>(shard), "bad shard was written");
    }
    const int store_before = count_open_fds();
    int store_throws = 0;
    for (int attempt = 0; attempt < 32; ++attempt) {
        try {
            (void)generate_text(store_dir, "fixed-text-v1", "greedy", 1);
        } catch (const std::runtime_error&) {
            ++store_throws;
        }
    }
    const int store_after = count_open_fds();
    const bool store_mapped = maps_contain(store_dir);
    std::filesystem::remove_all(store_dir);
    expect(store_throws == 32, "bad shard throws 32 times");
    expect(store_after == store_before, "store fd count is unchanged");
    expect(!store_mapped, "temp shard is absent from maps");

    const auto tp2_dir = make_temp_dir("ninfer-nvfp4-tp2-");
    const int tp2_before = count_open_fds();
    bool tp2_threw = false;
    bool tp2_ok = true;
    try {
        tp2_ok = generate_text(tp2_dir, "fixed-text-v1", "tp2", 1).ok;
    } catch (const std::exception&) {
        tp2_threw = true;
    }
    const int tp2_after = count_open_fds();
    int status = 0;
    const pid_t leftover = ::waitpid(-1, &status, WNOHANG);
    std::filesystem::remove_all(tp2_dir);
    expect(tp2_threw || !tp2_ok, "empty tp2 directory fails");
    expect(tp2_after == tp2_before, "tp2 fd count is unchanged");
    expect(leftover == -1, "tp2 child was reaped");
}

}  // namespace

int main() {
    using namespace ninfer::glm53;
    const float gate[] = {0.f, 100.f, -2.f};
    const float up[] = {1.f, 100.f, -20.f};
    float hidden[3];
    swiglu_clamp(gate, up, 3, 10.f, hidden);
    expect(near(hidden[0], 0.f) && near(hidden[1], 99.995460213f) && near(hidden[2], 2.384058440f), "clamped swiglu");

    const float logits[] = {0.f, 10.f, 3.f, 10.f, 1.f};
    const float bias[] = {0.f, 0.f, 0.f, 0.f, 0.f};
    int indices[2];
    float weights[2];
    router_select(logits, bias, 5, 2, 2.5f, true, indices, weights);
    expect(indices[0] == 1 && indices[1] == 3, "router keeps the lower index on a tie");
    expect(near(weights[0], 1.25f) && near(weights[1], 1.25f), "router renormalizes");

    float forget = 0.f;
    const float proj[] = {0.f};
    const float dt[] = {0.f};
    const float a_log[] = {0.f};
    kda_forget_gate(proj, dt, a_log, -5.f, 1, 1, &forget);
    expect(near(forget, -2.5f), "forget gate lower bound");

    float mem[3] = {0.f, 0.f, 0.f};
    const float taps[] = {0.f, 0.f, 0.f, 2.f};
    const float input[] = {3.f};
    float conv = 0.f;
    causal_conv_silu(input, taps, mem, 1, 4, &conv);
    expect(near(conv, 5.985164f), "first causal tap is silu");
    expect(near(mem[2], 3.f), "conv state stores the current input");

    float state[4] = {};
    const float q[] = {3.f, 4.f};
    const float k[] = {0.f, 2.f};
    const float v[] = {1.f, 0.f};
    const float g[] = {0.f, 0.f};
    const float beta[] = {1.f};
    float out[2];
    kda_recurrent_heads(state, q, k, v, g, beta, 1, 2, out);
    expect(near(out[0], 0.565685343f) && near(out[1], 0.f), "kda recurrence");

    const float streams[] = {0.5f, 1.5f};
    float fn[16] = {};
    fn[0] = 1.f;
    fn[3] = 1.f;
    float base[8] = {};
    const float scale[] = {1.f, 1.f, 1.f};
    float post[2];
    float comb[4];
    float collapsed[1];
    mhc_project(streams, 2, 1, fn, base, scale, 1e-5f, 1e-6f, 2, post, comb, collapsed);
    expect(near(collapsed[0], 1.494128191f), "mhc collapse");
    expect(near(post[0], 1.f) && near(post[1], 1.f), "mhc post");
    expect(near(comb[0], 0.4999995f) && near(comb[3], 0.4999995f), "mhc sinkhorn");

    const float residual[] = {0.5f, 1.5f};
    const float branch[] = {2.f};
    const float mix_post[] = {1.f, 1.f};
    const float mix_comb[] = {0.5f, 0.5f, 0.5f, 0.5f};
    float mixed[2];
    mhc_combine(residual, branch, mix_post, mix_comb, 2, 1, mixed);
    expect(near(mixed[0], 3.f) && near(mixed[1], 3.f), "mhc combine");

    std::string text;
    std::vector<std::int32_t> tokens;
    expect(fixed_prompt_tokens("fixed-text-v1", text, tokens) && text == "Hi" && tokens.size() == 1U && tokens[0] == 13041,
           "fixed prompt");
    expect(!fixed_prompt_tokens("other", text, tokens), "unknown prompt");
    test_causal_conv_alias();
    test_store_and_tp2_release();
    return 0;
}
