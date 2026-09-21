#include "ninfer_glm53/model_spec.hpp"

#include <algorithm>
#include <array>
#include <sstream>

namespace ninfer::glm53 {
namespace {

constexpr bool is_sparse_mla_layer(std::uint32_t layer) noexcept {
    // Official GLM-5.3-Flash config: sparse/full-attention layer every four
    // layers, beginning at layer 3.
    return layer >= 3U && ((layer - 3U) % 4U) == 0U;
}

constexpr std::array<LayerSpec, ModelSpec::kLayerCount> make_layers() noexcept {
    std::array<LayerSpec, ModelSpec::kLayerCount> layers{};
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const auto index = static_cast<std::uint32_t>(i);
        layers[i] = LayerSpec{
            .index = index,
            .mixer = is_sparse_mla_layer(index) ? MixerKind::kSparseMla : MixerKind::kKda,
            .ffn = index < 3U ? FfnKind::kDense : FfnKind::kSparseMoe,
        };
    }
    return layers;
}

constexpr auto kLayers = make_layers();

const ModelSpec kSpec{
    .architecture = "Glm5NextForConditionalGeneration",
    .checkpoint = "zai-org/GLM-5.3-Flash",
    .hidden_size = 4096,
    .intermediate_size = 12288,
    .vocab_size = 154880,
    .max_position_embeddings = 1048576,
    .attention_heads = 64,
    .kv_heads = 64,
    .q_lora_rank = 1536,
    .kv_lora_rank = 512,
    .qk_head_dim = 256,
    .qk_nope_head_dim = 256,
    .qk_rope_head_dim = 0,
    .v_head_dim = 256,
    .index_heads = 32,
    .index_head_dim = 128,
    .index_topk = 2048,
    .index_kpool = 4,
    .linear_attention_heads = 64,
    .linear_attention_head_dim = 128,
    .short_conv_kernel_size = 4,
    .kda_gate_lower_bound = -5.0,
    .manifold_hyper_connections = true,
    .hyper_connection_streams = 4,
    .hyper_connection_epsilon = 1e-6,
    .hyper_connection_sinkhorn_iterations = 20,
    .routed_experts = 288,
    .shared_experts = 1,
    .experts_per_token = 8,
    .moe_intermediate_size = 2048,
    .dense_prefix_layers = 3,
    .routed_scaling_factor = 2.5,
    .swiglu_limit = 10.0,
    .router_dtype = "float32",
    .router_scoring_function = "sigmoid",
    .router_topk_method = "noaux_tc",
    .mla_use_nope = true,
    .index_kpool_compress = true,
    .index_kpool_always_select_tail = true,
    .index_share_for_mtp_iteration = true,
    .indexer_rope_interleave = true,
    .nextn_predict_layers = 1,
    .mhc_parameters_fp32 = true,
    .kda_decay_parameters_fp32 = true,
    .layers = kLayers,
};

}  // namespace

const ModelSpec& glm53_flash_spec() { return kSpec; }

std::size_t count_mixer(const ModelSpec& spec, MixerKind kind) noexcept {
    return static_cast<std::size_t>(std::count_if(
        spec.layers.begin(), spec.layers.end(),
        [kind](const LayerSpec& layer) { return layer.mixer == kind; }));
}

std::size_t count_ffn(const ModelSpec& spec, FfnKind kind) noexcept {
    return static_cast<std::size_t>(std::count_if(
        spec.layers.begin(), spec.layers.end(),
        [kind](const LayerSpec& layer) { return layer.ffn == kind; }));
}

std::vector<std::string> validate_model_spec(const ModelSpec& spec) {
    std::vector<std::string> errors;

    if (spec.architecture != "Glm5NextForConditionalGeneration") {
        errors.emplace_back("architecture must be Glm5NextForConditionalGeneration");
    }
    if (spec.hidden_size != 4096U) errors.emplace_back("hidden_size must be 4096");
    if (spec.layers.size() != ModelSpec::kLayerCount) errors.emplace_back("layer count must be 45");
    if (spec.max_position_embeddings != 1048576U) {
        errors.emplace_back("max_position_embeddings must be 1048576");
    }
    if (spec.attention_heads != 64U || spec.kv_heads != 64U) {
        errors.emplace_back("attention and KV head counts must both be 64");
    }
    if (spec.q_lora_rank != 1536U || spec.kv_lora_rank != 512U) {
        errors.emplace_back("MLA LoRA ranks do not match GLM-5.3-Flash");
    }
    if (spec.qk_rope_head_dim != 0U) {
        errors.emplace_back("GLM-5.3-Flash contract expects qk_rope_head_dim=0");
    }
    if (spec.index_heads != 32U || spec.index_head_dim != 128U || spec.index_topk != 2048U) {
        errors.emplace_back("sparse indexer geometry does not match GLM-5.3-Flash");
    }
    if (spec.routed_experts != 288U || spec.experts_per_token != 8U) {
        errors.emplace_back("MoE geometry must be 288 routed experts with top-8 routing");
    }
    if (spec.dense_prefix_layers != 3U) {
        errors.emplace_back("first three layers must use dense FFN");
    }
    if (!spec.manifold_hyper_connections || spec.hyper_connection_streams != 4U ||
        spec.hyper_connection_epsilon != 1e-6 || spec.hyper_connection_sinkhorn_iterations != 20U) {
        errors.emplace_back("mHC contract must use four streams, eps=1e-6 and 20 Sinkhorn iterations");
    }
    if (spec.kda_gate_lower_bound != -5.0) {
        errors.emplace_back("KDA gate lower bound must be -5.0");
    }
    if (spec.routed_scaling_factor != 2.5 || spec.swiglu_limit != 10.0) {
        errors.emplace_back("MoE routed scaling / SwiGLU clamp do not match GLM-5.3-Flash");
    }
    if (spec.router_dtype != "float32" || spec.router_scoring_function != "sigmoid" ||
        spec.router_topk_method != "noaux_tc") {
        errors.emplace_back("MoE router contract does not match GLM-5.3-Flash");
    }
    if (!spec.mla_use_nope || !spec.index_kpool_compress || !spec.index_kpool_always_select_tail ||
        !spec.index_share_for_mtp_iteration || !spec.indexer_rope_interleave) {
        errors.emplace_back("sparse MLA/indexer feature flags do not match GLM-5.3-Flash");
    }
    if (spec.nextn_predict_layers != 1U) {
        errors.emplace_back("official config declares one next-token-prediction layer");
    }
    if (!spec.mhc_parameters_fp32 || !spec.kda_decay_parameters_fp32) {
        errors.emplace_back("mHC and KDA decay precision islands must remain fp32");
    }

    for (std::size_t i = 0; i < spec.layers.size(); ++i) {
        const auto& layer = spec.layers[i];
        if (layer.index != i) {
            std::ostringstream out;
            out << "layer index mismatch at slot " << i << ": got " << layer.index;
            errors.push_back(out.str());
        }
        const auto expected_mixer = is_sparse_mla_layer(static_cast<std::uint32_t>(i))
                                        ? MixerKind::kSparseMla
                                        : MixerKind::kKda;
        if (layer.mixer != expected_mixer) {
            std::ostringstream out;
            out << "unexpected mixer kind at layer " << i;
            errors.push_back(out.str());
        }
        const auto expected_ffn = i < 3U ? FfnKind::kDense : FfnKind::kSparseMoe;
        if (layer.ffn != expected_ffn) {
            std::ostringstream out;
            out << "unexpected FFN kind at layer " << i;
            errors.push_back(out.str());
        }
    }

    if (count_mixer(spec, MixerKind::kKda) != 34U) errors.emplace_back("expected 34 KDA layers");
    if (count_mixer(spec, MixerKind::kSparseMla) != 11U) {
        errors.emplace_back("expected 11 sparse-MLA layers");
    }
    if (count_ffn(spec, FfnKind::kDense) != 3U) errors.emplace_back("expected 3 dense FFN layers");
    if (count_ffn(spec, FfnKind::kSparseMoe) != 42U) {
        errors.emplace_back("expected 42 sparse-MoE layers");
    }

    return errors;
}

std::string_view to_string(MixerKind kind) noexcept {
    switch (kind) {
        case MixerKind::kKda:
            return "kda";
        case MixerKind::kSparseMla:
            return "sparse_mla";
    }
    return "unknown";
}

std::string_view to_string(FfnKind kind) noexcept {
    switch (kind) {
        case FfnKind::kDense:
            return "dense";
        case FfnKind::kSparseMoe:
            return "sparse_moe";
    }
    return "unknown";
}

}  // namespace ninfer::glm53
