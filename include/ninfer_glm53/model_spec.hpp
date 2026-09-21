#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::glm53 {

enum class MixerKind : std::uint8_t {
    kKda,
    kSparseMla,
};

enum class FfnKind : std::uint8_t {
    kDense,
    kSparseMoe,
};

struct LayerSpec {
    std::uint32_t index{};
    MixerKind mixer{MixerKind::kKda};
    FfnKind ffn{FfnKind::kSparseMoe};
};

struct ModelSpec {
    static constexpr std::size_t kLayerCount = 45;

    std::string_view architecture;
    std::string_view checkpoint;

    std::uint32_t hidden_size{};
    std::uint32_t intermediate_size{};
    std::uint32_t vocab_size{};
    std::uint32_t max_position_embeddings{};

    std::uint32_t attention_heads{};
    std::uint32_t kv_heads{};
    std::uint32_t q_lora_rank{};
    std::uint32_t kv_lora_rank{};
    std::uint32_t qk_head_dim{};
    std::uint32_t qk_nope_head_dim{};
    std::uint32_t qk_rope_head_dim{};
    std::uint32_t v_head_dim{};

    std::uint32_t index_heads{};
    std::uint32_t index_head_dim{};
    std::uint32_t index_topk{};
    std::uint32_t index_kpool{};

    std::uint32_t linear_attention_heads{};
    std::uint32_t linear_attention_head_dim{};
    std::uint32_t short_conv_kernel_size{};
    double kda_gate_lower_bound{};

    bool manifold_hyper_connections{};
    std::uint32_t hyper_connection_streams{};
    double hyper_connection_epsilon{};
    std::uint32_t hyper_connection_sinkhorn_iterations{};

    std::uint32_t routed_experts{};
    std::uint32_t shared_experts{};
    std::uint32_t experts_per_token{};
    std::uint32_t moe_intermediate_size{};
    std::uint32_t dense_prefix_layers{};
    double routed_scaling_factor{};
    double swiglu_limit{};
    std::string_view router_dtype;
    std::string_view router_scoring_function;
    std::string_view router_topk_method;

    bool mla_use_nope{};
    bool index_kpool_compress{};
    bool index_kpool_always_select_tail{};
    bool index_share_for_mtp_iteration{};
    bool indexer_rope_interleave{};
    std::uint32_t nextn_predict_layers{};

    // Precision islands are part of the semantic contract. These parameters
    // must not be silently folded into low-bit weight storage.
    bool mhc_parameters_fp32{};
    bool kda_decay_parameters_fp32{};

    std::array<LayerSpec, kLayerCount> layers{};
};

[[nodiscard]] const ModelSpec& glm53_flash_spec();
[[nodiscard]] std::vector<std::string> validate_model_spec(const ModelSpec& spec);
[[nodiscard]] std::size_t count_mixer(const ModelSpec& spec, MixerKind kind) noexcept;
[[nodiscard]] std::size_t count_ffn(const ModelSpec& spec, FfnKind kind) noexcept;
[[nodiscard]] std::string_view to_string(MixerKind kind) noexcept;
[[nodiscard]] std::string_view to_string(FfnKind kind) noexcept;

}  // namespace ninfer::glm53
