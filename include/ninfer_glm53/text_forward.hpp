#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace ninfer::glm53 {

// Clamped SwiGLU used by every text MLP. gate is capped at +limit, up at ±limit.
void swiglu_clamp(const float* gate, const float* up, int n, float limit, float* hidden);

// Sigmoid router with bias used only for selection. Ties keep the lower expert index.
// weights are renormalized and multiplied by scaling when normalize is true.
void router_select(const float* logits, const float* bias, int experts, int top_k, float scaling, bool normalize,
                   int* indices, float* weights);

// Forget-gate output that the KDA recurrence exponentiates. lower_bound is the checkpoint's negative cap.
void kda_forget_gate(const float* proj, const float* dt_bias, const float* a_log, float lower_bound, int heads,
                     int dim, float* g);

// Depthwise causal conv. weight is [channels, kernel], mem is [channels, kernel - 1] oldest first.
// Exact alias x == y is supported. Partial overlap of the channel ranges is rejected.
void causal_conv_silu(const float* x, const float* weight, float* mem, int channels, int kernel, float* y);

// One KDA step. q/k/v/g are [heads, dim], beta is [heads], state is [heads, dim, dim] and is updated.
// q and k are L2-normalized with eps inside the square root, then q is scaled by 1/sqrt(dim).
void kda_recurrent_heads(float* state, const float* q, const float* k, const float* v, const float* g,
                         const float* beta, int heads, int dim, float* out);

// mHC weight projection. streams and the returned mix are row-major [hc, hidden].
// fn is [((hc + 2) * hc), hc * hidden]. scale is pre, post, comb.
void mhc_project(const float* streams, int hc, int hidden, const float* fn, const float* base, const float* scale,
                 float rms_eps, float hc_eps, int sinkhorn_iters, float* post, float* comb, float* collapsed);

// out[n, d] = post[n] * branch[d] + sum_i comb[i, n] * residual[i, d]
void mhc_combine(const float* residual, const float* branch, const float* post, const float* comb, int hc, int hidden,
                 float* out);

struct GenerateResult {
    bool ok = false;
    std::string reason;
    std::vector<std::int32_t> token_ids;
    std::uint32_t committed = 0;
    std::uint32_t rank0_committed = 0;
    std::uint32_t rank1_committed = 0;
    std::vector<std::int32_t> rank0_tokens;
    std::vector<std::int32_t> rank1_tokens;
    int world_size = 1;
    std::string prompt_text;
    std::vector<std::int32_t> prompt_tokens;
    std::string dflash_accept;
    std::vector<std::int32_t> draft_tokens;
    int accept_count = -1;
    float logit_margin = 0.f;
};

// Greedy text tokens. draft_checkpoint is required for mode dflash.
// world_size 2 forks a second process; the ranks exchange GEMV row shards over a socket.

// fixed-text-v1 is the single token for the text "Hi" (vocab id 13041).
[[nodiscard]] bool fixed_prompt_tokens(std::string_view prompt_id, std::string& text, std::vector<std::int32_t>& tokens);

// Greedy text tokens from the official ModelOpt checkpoint. world_size 2 shards every GEMV by output rows
// and concatenates the halves in process. Sequences longer than the indexer top-k are refused.
[[nodiscard]] GenerateResult generate_text(const std::filesystem::path& checkpoint, std::string_view prompt_id,
                                           std::string_view mode, int new_tokens,
                                           const std::filesystem::path& draft_checkpoint = {});

// One rank of TP=2 on two machines. Rank 0 binds bind_host:port. Rank 1 binds
// bind_host and connects to peer_host:port. GEMV row halves use that socket.
[[nodiscard]] GenerateResult generate_text_peer(const std::filesystem::path& checkpoint, std::string_view prompt_id,
                                                int new_tokens, int rank, std::string_view bind_host,
                                                std::string_view peer_host, int port);

}  // namespace ninfer::glm53
