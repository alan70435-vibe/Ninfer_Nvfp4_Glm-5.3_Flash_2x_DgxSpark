#include "ninfer_glm53/text_forward.hpp"

#include <cmath>
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

bool near(float actual, float expected) { return std::fabs(actual - expected) < 1e-4f; }

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
    return 0;
}
