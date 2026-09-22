#include "ninfer_glm53/parameter_schema.hpp"

#include "sha256.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace ninfer::glm53 {
namespace {

std::int64_t as_dim(std::uint32_t value) { return static_cast<std::int64_t>(value); }

std::int64_t mul_dim(std::uint32_t lhs, std::uint32_t rhs) {
    return static_cast<std::int64_t>(lhs) * static_cast<std::int64_t>(rhs);
}

void add(std::vector<ExpectedTensor>& out,
         std::string logical_id,
         std::string source_name,
         DType dtype,
         StorageClass storage,
         std::initializer_list<std::int64_t> shape,
         bool mcg_payload = false) {
    ExpectedTensor tensor;
    tensor.logical_id = std::move(logical_id);
    tensor.source_name = std::move(source_name);
    tensor.dtype = dtype;
    tensor.storage = storage;
    tensor.shape.assign(shape.begin(), shape.end());
    tensor.mcg_payload = mcg_payload;
    out.push_back(std::move(tensor));
}

void add_bf16(std::vector<ExpectedTensor>& out,
              const std::string& logical_id,
              const std::string& source_name,
              std::initializer_list<std::int64_t> shape) {
    add(out, logical_id, source_name, DType::kBf16, StorageClass::kNativeBf16, shape, false);
}

void add_f32(std::vector<ExpectedTensor>& out,
             const std::string& logical_id,
             const std::string& source_name,
             std::initializer_list<std::int64_t> shape) {
    add(out, logical_id, source_name, DType::kF32, StorageClass::kNativeF32, shape, false);
}

// Official ModelOpt NVFP4 linear. Logical [N, K] is stored as packed U8 [N, K/2]
// (low nibble is the even K element), FP8 E4M3 block scales [N, K/16], and two
// rank-0 FP32 scales. Reconstruction is e2m1 * fp8_scale * weight_scale_2.
void add_nvfp4(std::vector<ExpectedTensor>& out,
               const std::string& logical_prefix,
               const std::string& source_prefix,
               std::int64_t out_features,
               std::int64_t in_features) {
    if (out_features <= 0 || in_features <= 0 || in_features % static_cast<std::int64_t>(kNvfp4GroupSize) != 0) {
        throw std::logic_error("NVFP4 input features must be a positive multiple of the group size");
    }
    add(out, logical_prefix + ".weight", source_prefix + ".weight", DType::kU8, StorageClass::kNvfp4PackedU8,
        {out_features, in_features / 2}, false);
    add(out, logical_prefix + ".weight_scale", source_prefix + ".weight_scale", DType::kF8E4M3,
        StorageClass::kNvfp4BlockScaleF8, {out_features, in_features / static_cast<std::int64_t>(kNvfp4GroupSize)},
        false);
    add(out, logical_prefix + ".weight_scale_2", source_prefix + ".weight_scale_2", DType::kF32,
        StorageClass::kNvfp4GlobalScaleF32, {}, false);
    add(out, logical_prefix + ".input_scale", source_prefix + ".input_scale", DType::kF32,
        StorageClass::kNvfp4GlobalScaleF32, {}, false);
}

void add_moe(std::vector<ExpectedTensor>& out, const ModelSpec& model, const std::string& logical,
             const std::string& source, bool nextn) {
    const auto hidden = as_dim(model.hidden_size);
    const auto moe = as_dim(model.moe_intermediate_size);
    add_bf16(out, logical + "moe.router.weight", source + "mlp.gate.weight",
             {as_dim(model.routed_experts), hidden});
    add_f32(out, logical + "moe.router.score_correction", source + "mlp.gate.e_score_correction_bias",
            {as_dim(model.routed_experts)});
    add_bf16(out, logical + "moe.shared.gate_proj", source + "mlp.shared_experts.gate_proj.weight", {moe, hidden});
    add_bf16(out, logical + "moe.shared.up_proj", source + "mlp.shared_experts.up_proj.weight", {moe, hidden});
    add_bf16(out, logical + "moe.shared.down_proj", source + "mlp.shared_experts.down_proj.weight", {hidden, moe});

    for (std::uint32_t expert = 0; expert < model.routed_experts; ++expert) {
        const std::string expert_source = source + "mlp.experts." + std::to_string(expert) + ".";
        const std::string expert_logical = logical + "moe.expert." + std::to_string(expert) + ".";
        if (nextn) {
            // Official ModelOpt leaves the MTP experts in BF16. There is no scale tensor.
            add_bf16(out, expert_logical + "gate", expert_source + "gate_proj.weight", {moe, hidden});
            add_bf16(out, expert_logical + "up", expert_source + "up_proj.weight", {moe, hidden});
            add_bf16(out, expert_logical + "down", expert_source + "down_proj.weight", {hidden, moe});
        } else {
            add_nvfp4(out, expert_logical + "gate", expert_source + "gate_proj", moe, hidden);
            add_nvfp4(out, expert_logical + "up", expert_source + "up_proj", moe, hidden);
            add_nvfp4(out, expert_logical + "down", expert_source + "down_proj", hidden, moe);
        }
    }
}

void add_kda(std::vector<ExpectedTensor>& out, const ModelSpec& model, const std::string& logical,
             const std::string& source) {
    const auto hidden = as_dim(model.hidden_size);
    const auto heads = as_dim(model.linear_attention_heads);
    const auto head_dim = as_dim(model.linear_attention_head_dim);
    const auto width = mul_dim(model.linear_attention_heads, model.linear_attention_head_dim);
    const auto kernel = as_dim(model.short_conv_kernel_size);
    add_f32(out, logical + "kda.a_log", source + "self_attn.A_log", {heads});
    add_f32(out, logical + "kda.dt_bias", source + "self_attn.dt_bias", {width});
    add_bf16(out, logical + "kda.b_proj", source + "self_attn.b_proj.weight", {heads, hidden});
    add_bf16(out, logical + "kda.f_a_proj", source + "self_attn.f_a_proj.weight", {head_dim, hidden});
    add_bf16(out, logical + "kda.f_b_proj", source + "self_attn.f_b_proj.weight", {width, head_dim});
    add_bf16(out, logical + "kda.g_a_proj", source + "self_attn.g_a_proj.weight", {head_dim, hidden});
    add_bf16(out, logical + "kda.g_b_proj", source + "self_attn.g_b_proj.weight", {width, head_dim});
    add_bf16(out, logical + "kda.q_proj", source + "self_attn.q_proj.weight", {width, hidden});
    add_bf16(out, logical + "kda.k_proj", source + "self_attn.k_proj.weight", {width, hidden});
    add_bf16(out, logical + "kda.v_proj", source + "self_attn.v_proj.weight", {width, hidden});
    // Official ModelOpt export stores the short convolutions as FP32.
    add_f32(out, logical + "kda.q_conv", source + "self_attn.q_conv1d.weight", {width, 1, kernel});
    add_f32(out, logical + "kda.k_conv", source + "self_attn.k_conv1d.weight", {width, 1, kernel});
    add_f32(out, logical + "kda.v_conv", source + "self_attn.v_conv1d.weight", {width, 1, kernel});
    add_bf16(out, logical + "kda.o_norm", source + "self_attn.o_norm.weight", {head_dim});
}

void add_sparse_mla(std::vector<ExpectedTensor>& out, const ModelSpec& model, const std::string& logical,
                    const std::string& source) {
    const auto hidden = as_dim(model.hidden_size);
    const auto q_lora = as_dim(model.q_lora_rank);
    const auto kv_lora = as_dim(model.kv_lora_rank);
    const auto nope = as_dim(model.qk_nope_head_dim);
    const auto v_head = as_dim(model.v_head_dim);
    const auto q_out = mul_dim(model.attention_heads, model.qk_head_dim);
    const auto kv_b_out = static_cast<std::int64_t>(model.attention_heads) * (nope + v_head);
    const auto index_heads = as_dim(model.index_heads);
    const auto index_dim = as_dim(model.index_head_dim);
    const auto index_q_out = mul_dim(model.index_heads, model.index_head_dim);
    add_bf16(out, logical + "mla.q_a_proj", source + "self_attn.q_a_proj.weight", {q_lora, hidden});
    add_bf16(out, logical + "mla.q_a_norm", source + "self_attn.q_a_layernorm.weight", {q_lora});
    add_bf16(out, logical + "mla.q_b_proj", source + "self_attn.q_b_proj.weight", {q_out, q_lora});
    add_bf16(out, logical + "mla.kv_a_proj", source + "self_attn.kv_a_proj_with_mqa.weight", {kv_lora, hidden});
    add_bf16(out, logical + "mla.kv_a_norm", source + "self_attn.kv_a_layernorm.weight", {kv_lora});
    add_bf16(out, logical + "mla.kv_b_proj", source + "self_attn.kv_b_proj.weight", {kv_b_out, kv_lora});
    add_bf16(out, logical + "indexer.wk", source + "self_attn.indexer.wk.weight", {index_dim, hidden});
    add_bf16(out, logical + "indexer.wq_b", source + "self_attn.indexer.wq_b.weight", {index_q_out, q_lora});
    add_bf16(out, logical + "indexer.weights_proj", source + "self_attn.indexer.weights_proj.weight",
             {index_heads, hidden});
    add_bf16(out, logical + "indexer.k_norm", source + "self_attn.indexer.k_norm.weight", {index_dim});
    add_bf16(out, logical + "indexer.k_norm_bias", source + "self_attn.indexer.k_norm.bias", {index_dim});
    add_bf16(out, logical + "indexer.kpool_ape", source + "self_attn.indexer.index_kpool_compress_ape",
             {as_dim(model.index_kpool), index_dim});
    add_bf16(out, logical + "indexer.kpool_gate", source + "self_attn.indexer.index_kpool_compress_gate",
             {index_dim, hidden});
}

void add_mhc(std::vector<ExpectedTensor>& out, const ModelSpec& model, const std::string& logical,
             const std::string& source) {
    // Official ModelOpt export stores every mHC tensor, including base and scale, as BF16.
    // base has streams*(streams+2) coefficients, scale has streams-1, and fn maps
    // the concatenated streams (hidden*streams) into that width.
    const auto streams = model.hyper_connection_streams;
    const auto coeff = as_dim(streams * (streams + 2U));
    const auto scale = as_dim(streams - 1U);
    const auto fn_in = mul_dim(model.hidden_size, streams);
    add_bf16(out, logical + "mhc.attn.base", source + "hc_attn_base", {coeff});
    add_bf16(out, logical + "mhc.attn.fn", source + "hc_attn_fn", {coeff, fn_in});
    add_bf16(out, logical + "mhc.attn.scale", source + "hc_attn_scale", {scale});
    add_bf16(out, logical + "mhc.ffn.base", source + "hc_ffn_base", {coeff});
    add_bf16(out, logical + "mhc.ffn.fn", source + "hc_ffn_fn", {coeff, fn_in});
    add_bf16(out, logical + "mhc.ffn.scale", source + "hc_ffn_scale", {scale});
}

void add_language_layer(std::vector<ExpectedTensor>& out, const ModelSpec& model, std::uint32_t index,
                        bool nextn, MixerKind mixer, FfnKind ffn) {
    const std::string source = "model.language_model.layers." + std::to_string(index) + ".";
    const std::string logical =
        nextn ? std::string("language.nextn.") : "language.layer." + std::to_string(index) + ".";
    const auto hidden = as_dim(model.hidden_size);
    add_bf16(out, logical + "input_norm", source + "input_layernorm.weight", {hidden});
    add_bf16(out, logical + "post_attn_norm", source + "post_attention_layernorm.weight", {hidden});
    if (!nextn) add_mhc(out, model, logical, source);

    const auto o_proj_k = mixer == MixerKind::kKda
                              ? mul_dim(model.linear_attention_heads, model.linear_attention_head_dim)
                              : mul_dim(model.attention_heads, model.v_head_dim);
    add_bf16(out, logical + "mixer.o_proj", source + "self_attn.o_proj.weight", {hidden, o_proj_k});
    if (mixer == MixerKind::kKda) add_kda(out, model, logical, source);
    else add_sparse_mla(out, model, logical, source);

    if (ffn == FfnKind::kDense) {
        const auto intermediate = as_dim(model.intermediate_size);
        add_nvfp4(out, logical + "ffn.gate_proj", source + "mlp.gate_proj", intermediate, hidden);
        add_nvfp4(out, logical + "ffn.up_proj", source + "mlp.up_proj", intermediate, hidden);
        add_nvfp4(out, logical + "ffn.down_proj", source + "mlp.down_proj", hidden, intermediate);
    } else {
        add_moe(out, model, logical, source, nextn);
    }

    if (nextn) {
        add_bf16(out, logical + "eh_proj", source + "eh_proj.weight", {hidden, hidden * 2});
        add_bf16(out, logical + "enorm", source + "enorm.weight", {hidden});
        add_bf16(out, logical + "hnorm", source + "hnorm.weight", {hidden});
        add_bf16(out, logical + "shared_head_norm", source + "shared_head.norm.weight", {hidden});
    }
}

void add_vision(std::vector<ExpectedTensor>& out, const VisionSpec& vision) {
    const auto hidden = as_dim(vision.hidden_size);
    const auto heads = vision.num_heads;
    if (heads == 0U || vision.hidden_size % heads != 0U) {
        throw std::logic_error("vision hidden size must divide evenly by head count");
    }
    const auto head_dim = as_dim(vision.hidden_size / heads);
    const auto intermediate = as_dim(vision.intermediate_size);
    const auto out_hidden = as_dim(vision.out_hidden_size);
    const auto qkv = hidden * 3;
    for (std::uint32_t block = 0; block < vision.depth; ++block) {
        const std::string source = "model.visual.blocks." + std::to_string(block) + ".";
        const std::string logical = "vision.block." + std::to_string(block) + ".";
        add_bf16(out, logical + "norm1", source + "norm1.weight", {hidden});
        add_bf16(out, logical + "norm2", source + "norm2.weight", {hidden});
        add_bf16(out, logical + "attn.q_norm", source + "attn.q_norm.weight", {head_dim});
        add_bf16(out, logical + "attn.k_norm", source + "attn.k_norm.weight", {head_dim});
        add_bf16(out, logical + "attn.qkv", source + "attn.qkv.weight", {qkv, hidden});
        add_bf16(out, logical + "attn.qkv_bias", source + "attn.qkv.bias", {qkv});
        add_bf16(out, logical + "attn.proj", source + "attn.proj.weight", {hidden, hidden});
        add_bf16(out, logical + "attn.proj_bias", source + "attn.proj.bias", {hidden});
        add_bf16(out, logical + "mlp.gate_proj", source + "mlp.gate_proj.weight", {intermediate, hidden});
        add_bf16(out, logical + "mlp.gate_bias", source + "mlp.gate_proj.bias", {intermediate});
        add_bf16(out, logical + "mlp.up_proj", source + "mlp.up_proj.weight", {intermediate, hidden});
        add_bf16(out, logical + "mlp.up_bias", source + "mlp.up_proj.bias", {intermediate});
        add_bf16(out, logical + "mlp.down_proj", source + "mlp.down_proj.weight", {hidden, intermediate});
        add_bf16(out, logical + "mlp.down_bias", source + "mlp.down_proj.bias", {hidden});
    }

    const auto proj_mid = as_dim(vision.projection_intermediate_size);
    const auto merge = as_dim(vision.spatial_merge_size);
    add_bf16(out, "vision.patch_embed", "model.visual.patch_embed.proj.weight",
             {hidden, as_dim(vision.in_channels), as_dim(vision.temporal_patch_size), as_dim(vision.patch_size),
              as_dim(vision.patch_size)});
    add_bf16(out, "vision.patch_embed_bias", "model.visual.patch_embed.proj.bias", {hidden});
    add_bf16(out, "vision.downsample", "model.visual.downsample.weight", {out_hidden, hidden, merge, merge});
    add_bf16(out, "vision.downsample_bias", "model.visual.downsample.bias", {out_hidden});
    add_bf16(out, "vision.post_layernorm", "model.visual.post_layernorm.weight", {hidden});
    add_bf16(out, "vision.merger.gate_proj", "model.visual.merger.gate_proj.weight", {proj_mid, out_hidden});
    add_bf16(out, "vision.merger.up_proj", "model.visual.merger.up_proj.weight", {proj_mid, out_hidden});
    add_bf16(out, "vision.merger.down_proj", "model.visual.merger.down_proj.weight", {out_hidden, proj_mid});
    add_bf16(out, "vision.merger.proj", "model.visual.merger.proj.weight", {out_hidden, out_hidden});
    add_bf16(out, "vision.merger.post_norm", "model.visual.merger.post_projection_norm.weight", {out_hidden});
    add_bf16(out, "vision.merger.post_norm_bias", "model.visual.merger.post_projection_norm.bias", {out_hidden});
}

const VisionSpec kVision{
    .depth = 24,
    .hidden_size = 1024,
    .num_heads = 16,
    .intermediate_size = 4096,
    .out_hidden_size = 4096,
    .in_channels = 3,
    .patch_size = 14,
    .temporal_patch_size = 2,
    .spatial_merge_size = 2,
    .projection_intermediate_size = 10240,
};

const DFlash2Spec kDflash{
    .layers = 5,
    .hidden_size = 4096,
    .intermediate_size = 12288,
    .vocab_size = 154880,
    .num_heads = 32,
    .num_kv_heads = 8,
    .head_dim = 128,
    .selector_rank = 256,
    .conv_kernel_size = 2,
    .conv_group_size = 16,
    .block_size = 8,
    .num_target_layers = 45,
    .target_layer_ids = {5U, 14U, 24U, 33U, 42U},
};

}  // namespace

const VisionSpec& glm53_flash_vision_spec() { return kVision; }

const DFlash2Spec& glm53_flash_dflash2_spec() { return kDflash; }

std::vector<std::string> validate_vision_spec(const VisionSpec& spec) {
    std::vector<std::string> errors;
    if (spec.depth != 24U) errors.emplace_back("vision depth must be 24");
    if (spec.hidden_size != 1024U || spec.num_heads != 16U) {
        errors.emplace_back("vision hidden size / head count must be 1024 / 16");
    }
    if (spec.num_heads == 0U || spec.hidden_size % spec.num_heads != 0U) {
        errors.emplace_back("vision head count must divide hidden size");
    }
    if (spec.intermediate_size != 4096U || spec.out_hidden_size != 4096U) {
        errors.emplace_back("vision intermediate and merger width must be 4096");
    }
    if (spec.in_channels != 3U || spec.patch_size != 14U || spec.temporal_patch_size != 2U ||
        spec.spatial_merge_size != 2U) {
        errors.emplace_back("vision patch geometry does not match GLM-5.3-Flash");
    }
    if (spec.projection_intermediate_size != 10240U) {
        errors.emplace_back("vision merger intermediate size must be 10240");
    }
    return errors;
}

std::vector<std::string> validate_dflash2_spec(const DFlash2Spec& spec) {
    std::vector<std::string> errors;
    if (spec.layers != 5U) errors.emplace_back("DFlash2 draft must have 5 layers");
    if (spec.hidden_size != 4096U || spec.intermediate_size != 12288U || spec.vocab_size != 154880U) {
        errors.emplace_back("DFlash2 hidden/intermediate/vocab geometry does not match the pinned draft");
    }
    if (spec.num_heads != 32U || spec.num_kv_heads != 8U || spec.head_dim != 128U) {
        errors.emplace_back("DFlash2 attention geometry must be 32 heads, 8 KV heads, head dim 128");
    }
    if (spec.num_heads * spec.head_dim != spec.hidden_size) {
        errors.emplace_back("DFlash2 query width must equal hidden size");
    }
    if (spec.selector_rank != 256U || spec.conv_kernel_size != 2U || spec.conv_group_size != 16U ||
        spec.block_size != 8U) {
        errors.emplace_back("DFlash2 selector/conv/block contract does not match the pinned draft");
    }
    if (spec.conv_group_size == 0U || spec.hidden_size % spec.conv_group_size != 0U) {
        errors.emplace_back("DFlash2 conv group size must divide hidden size");
    }
    if (spec.num_target_layers != 45U) errors.emplace_back("DFlash2 num_target_layers must be 45");
    const std::array<std::uint32_t, 5> expected_ids{5U, 14U, 24U, 33U, 42U};
    if (spec.target_layer_ids != expected_ids) {
        errors.emplace_back("DFlash2 target layer ids must be 5,14,24,33,42");
    }
    return errors;
}

std::vector<ExpectedTensor> expected_glm53_flash_tensors(const ModelSpec& model, const VisionSpec& vision) {
    const auto model_errors = validate_model_spec(model);
    const auto vision_errors = validate_vision_spec(vision);
    if (!model_errors.empty() || !vision_errors.empty()) {
        throw std::logic_error("GLM-5.3-Flash parameter schema requires the pinned model and vision contracts");
    }
    if (model.nextn_predict_layers != 1U) {
        throw std::logic_error("parameter schema expects exactly one next-token layer");
    }
    if (model.hyper_connection_streams < 2U) {
        throw std::logic_error("mHC scale width is streams-1");
    }

    std::vector<ExpectedTensor> out;
    out.reserve(160000U);
    add_bf16(out, "language.embed_tokens", "model.language_model.embed_tokens.weight",
             {as_dim(model.vocab_size), as_dim(model.hidden_size)});
    add_bf16(out, "language.final_norm", "model.language_model.norm.weight", {as_dim(model.hidden_size)});
    add_bf16(out, "language.lm_head", "lm_head.weight", {as_dim(model.vocab_size), as_dim(model.hidden_size)});

    for (const auto& layer : model.layers) {
        add_language_layer(out, model, layer.index, false, layer.mixer, layer.ffn);
    }
    const auto nextn_index = static_cast<std::uint32_t>(model.layers.size());
    add_language_layer(out, model, nextn_index, true, MixerKind::kSparseMla, FfnKind::kSparseMoe);
    add_vision(out, vision);
    return out;
}

std::vector<ExpectedTensor> expected_dflash2_tensors(const DFlash2Spec& spec) {
    if (!validate_dflash2_spec(spec).empty()) {
        throw std::logic_error("DFlash2 parameter schema requires the pinned draft contract");
    }
    std::vector<ExpectedTensor> out;
    out.reserve(96U);
    const auto hidden = as_dim(spec.hidden_size);
    const auto intermediate = as_dim(spec.intermediate_size);
    const auto vocab = as_dim(spec.vocab_size);
    const auto q_width = mul_dim(spec.num_heads, spec.head_dim);
    const auto kv_width = mul_dim(spec.num_kv_heads, spec.head_dim);
    const auto head_dim = as_dim(spec.head_dim);
    const auto kernel = as_dim(spec.conv_kernel_size);
    const auto conv_out = kernel * kernel * (hidden / as_dim(spec.conv_group_size));
    const auto targets = static_cast<std::int64_t>(spec.target_layer_ids.size());

    add_bf16(out, "draft.fc", "fc.weight", {hidden, targets * hidden});
    add_bf16(out, "draft.hidden_norm", "hidden_norm.weight", {hidden});
    add_bf16(out, "draft.final_norm", "norm.weight", {hidden});
    add_bf16(out, "draft.selector.hidden_proj", "candidate_selector.hidden_projection.weight",
             {as_dim(spec.selector_rank), hidden});
    add_bf16(out, "draft.selector.predecessor_codebook", "candidate_selector.predecessor_codebook",
             {vocab, as_dim(spec.selector_rank)});
    add_bf16(out, "draft.selector.successor_codebook", "candidate_selector.successor_codebook",
             {vocab, as_dim(spec.selector_rank)});

    for (std::uint32_t layer = 0; layer < spec.layers; ++layer) {
        const std::string source = "layers." + std::to_string(layer) + ".";
        const std::string logical = "draft.layer." + std::to_string(layer) + ".";
        add_bf16(out, logical + "input_norm", source + "input_layernorm.weight", {hidden});
        add_bf16(out, logical + "post_attn_norm", source + "post_attention_layernorm.weight", {hidden});
        add_bf16(out, logical + "attn.q_proj", source + "self_attn.q_proj.weight", {q_width, hidden});
        add_bf16(out, logical + "attn.k_proj", source + "self_attn.k_proj.weight", {kv_width, hidden});
        add_bf16(out, logical + "attn.v_proj", source + "self_attn.v_proj.weight", {kv_width, hidden});
        add_bf16(out, logical + "attn.o_proj", source + "self_attn.o_proj.weight", {hidden, q_width});
        add_bf16(out, logical + "attn.q_norm", source + "self_attn.q_norm.weight", {head_dim});
        add_bf16(out, logical + "attn.k_norm", source + "self_attn.k_norm.weight", {head_dim});
        add_bf16(out, logical + "attn_conv.base", source + "attention_conv.base_kernel", {kernel, kernel, hidden});
        add_bf16(out, logical + "attn_conv.kernel_proj", source + "attention_conv.kernel_projection.weight",
                 {conv_out, hidden});
        add_bf16(out, logical + "mlp_conv.base", source + "mlp_conv.base_kernel", {kernel, kernel, hidden});
        add_bf16(out, logical + "mlp_conv.kernel_proj", source + "mlp_conv.kernel_projection.weight",
                 {conv_out, hidden});
        add_bf16(out, logical + "ffn.gate_proj", source + "mlp.gate_proj.weight", {intermediate, hidden});
        add_bf16(out, logical + "ffn.up_proj", source + "mlp.up_proj.weight", {intermediate, hidden});
        add_bf16(out, logical + "ffn.down_proj", source + "mlp.down_proj.weight", {hidden, intermediate});
    }
    return out;
}

std::vector<ExpectedTensor> expected_glm53_flash_tensors() {
    return expected_glm53_flash_tensors(glm53_flash_spec(), glm53_flash_vision_spec());
}

std::vector<ExpectedTensor> expected_dflash2_tensors() {
    return expected_dflash2_tensors(glm53_flash_dflash2_spec());
}

std::vector<std::string> validate_parameter_catalog(const std::vector<ExpectedTensor>& catalog) {
    std::vector<std::string> errors;
    std::unordered_set<std::string> logical_ids;
    std::unordered_set<std::string> source_names;
    logical_ids.reserve(catalog.size());
    source_names.reserve(catalog.size());
    for (const auto& tensor : catalog) {
        if (tensor.logical_id.empty() || tensor.source_name.empty()) {
            errors.emplace_back("parameter catalog entry is missing a logical id or source name");
            continue;
        }
        if (!logical_ids.insert(tensor.logical_id).second) {
            errors.push_back("duplicate logical id " + tensor.logical_id);
        }
        if (!source_names.insert(tensor.source_name).second) {
            errors.push_back("duplicate source name " + tensor.source_name);
        }
        const bool per_tensor_scale = tensor.storage == StorageClass::kNvfp4GlobalScaleF32;
        if (!per_tensor_scale && tensor.shape.empty()) {
            errors.push_back("missing shape for " + tensor.logical_id);
        }
        for (const auto dim : tensor.shape) {
            if (dim <= 0) errors.push_back("non-positive shape for " + tensor.logical_id);
        }
        if (tensor.mcg_payload) {
            errors.push_back("NVFP4 catalog does not use an MCG payload for " + tensor.logical_id);
        }
        if (tensor.storage == StorageClass::kNativeBf16 && tensor.dtype != DType::kBf16) {
            errors.push_back("native bf16 storage has the wrong dtype for " + tensor.logical_id);
        }
        if (tensor.storage == StorageClass::kNativeF32 && tensor.dtype != DType::kF32) {
            errors.push_back("native fp32 storage has the wrong dtype for " + tensor.logical_id);
        }
        if (tensor.storage == StorageClass::kNvfp4PackedU8 &&
            (tensor.dtype != DType::kU8 || tensor.shape.size() != 2U)) {
            errors.push_back("NVFP4 packed storage must be U8[N, K/2] for " + tensor.logical_id);
        }
        if (tensor.storage == StorageClass::kNvfp4BlockScaleF8 &&
            (tensor.dtype != DType::kF8E4M3 || tensor.shape.size() != 2U)) {
            errors.push_back("NVFP4 block scale must be F8_E4M3[N, K/16] for " + tensor.logical_id);
        }
        if (tensor.storage == StorageClass::kNvfp4GlobalScaleF32 &&
            (tensor.dtype != DType::kF32 || !tensor.shape.empty())) {
            errors.push_back("NVFP4 per-tensor scale must be rank-0 F32 for " + tensor.logical_id);
        }
        if (tensor.storage == StorageClass::kFp8BlockWeight &&
            (tensor.dtype != DType::kF8E4M3 || tensor.shape.size() != 2U)) {
            errors.push_back("FP8 block weight must be F8_E4M3[N, K] for " + tensor.logical_id);
        }
        if (tensor.storage == StorageClass::kFp8BlockScaleBf16 &&
            (tensor.dtype != DType::kBf16 || tensor.shape.size() != 2U)) {
            errors.push_back("FP8 block scale must be BF16[N/128, K/128] for " + tensor.logical_id);
        }
    }
    return errors;
}

std::string logical_catalog_sha256(const std::vector<ExpectedTensor>& catalog) {
    std::vector<const ExpectedTensor*> ordered;
    ordered.reserve(catalog.size());
    for (const auto& tensor : catalog) ordered.push_back(&tensor);
    std::sort(ordered.begin(), ordered.end(), [](const ExpectedTensor* lhs, const ExpectedTensor* rhs) {
        return lhs->logical_id < rhs->logical_id;
    });
    detail::Sha256 hasher;
    for (const auto* tensor : ordered) {
        hasher.update(tensor->logical_id);
        hasher.update("\t");
        hasher.update(tensor->source_name);
        hasher.update("\t");
        hasher.update(to_string(tensor->dtype));
        hasher.update("\t");
        hasher.update(shape_to_string(tensor->shape));
        hasher.update("\n");
    }
    return hasher.hex_digest();
}

std::string_view to_string(DType dtype) noexcept {
    switch (dtype) {
        case DType::kBf16:
            return "BF16";
        case DType::kF16:
            return "F16";
        case DType::kF32:
            return "F32";
        case DType::kF8E4M3:
            return "F8_E4M3";
        case DType::kU8:
            return "U8";
        case DType::kI16:
            return "I16";
        case DType::kI32:
            return "I32";
    }
    return "UNKNOWN";
}

std::string_view to_string(StorageClass storage) noexcept {
    switch (storage) {
        case StorageClass::kNativeBf16:
            return "native_bf16";
        case StorageClass::kNativeF32:
            return "native_f32";
        case StorageClass::kNvfp4PackedU8:
            return "nvfp4_packed_u8";
        case StorageClass::kNvfp4BlockScaleF8:
            return "nvfp4_block_scale_f8";
        case StorageClass::kNvfp4GlobalScaleF32:
            return "nvfp4_global_scale_f32";
        case StorageClass::kFp8BlockWeight:
            return "fp8_block_weight";
        case StorageClass::kFp8BlockScaleBf16:
            return "fp8_block_scale_bf16";
    }
    return "unknown";
}

std::string shape_to_string(const std::vector<std::int64_t>& shape) {
    std::ostringstream out;
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i != 0U) out << ',';
        out << shape[i];
    }
    return out.str();
}

}  // namespace ninfer::glm53
