#include "ninfer_glm53/checkpoint_binding.hpp"
#include "ninfer_glm53/model_spec.hpp"
#include "ninfer_glm53/parameter_schema.hpp"

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

void expect_eq(std::size_t actual, std::size_t expected, const char* label) {
    if (actual != expected) {
        std::cerr << label << " actual=" << actual << " expected=" << expected << '\n';
        std::exit(1);
    }
}

template <class Fn>
void expect_throws(Fn&& fn, const char* label) {
    try {
        fn();
    } catch (const std::exception&) {
        return;
    }
    std::cerr << label << " did not throw\n";
    std::exit(1);
}

const ninfer::glm53::ExpectedTensor* find_logical(const std::vector<ninfer::glm53::ExpectedTensor>& catalog,
                                                  std::string_view logical_id) {
    for (const auto& tensor : catalog) {
        if (tensor.logical_id == logical_id) return &tensor;
    }
    return nullptr;
}

void write_text(const std::filesystem::path& path, std::string_view text) {
    std::ofstream output(path, std::ios::binary);
    expect(static_cast<bool>(output), "cannot write " + path.string());
    output << text;
}

void write_safetensors(const std::filesystem::path& path, std::string header, std::string_view data) {
    while ((8U + header.size()) % 8U != 0U) header.push_back(' ');
    std::ofstream output(path, std::ios::binary);
    expect(static_cast<bool>(output), "cannot write " + path.string());
    const auto length = static_cast<std::uint64_t>(header.size());
    unsigned char encoded[8]{};
    for (int byte = 0; byte < 8; ++byte) {
        encoded[byte] = static_cast<unsigned char>((length >> static_cast<unsigned>(byte * 8)) & 0xFFU);
    }
    output.write(reinterpret_cast<const char*>(encoded), 8);
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    output.write(data.data(), static_cast<std::streamsize>(data.size()));
}

ninfer::glm53::TensorCatalog catalog_from(const std::vector<ninfer::glm53::ExpectedTensor>& expected) {
    ninfer::glm53::TensorCatalog catalog;
    catalog.tensors.reserve(expected.size());
    for (const auto& tensor : expected) {
        ninfer::glm53::ObservedTensor observed;
        observed.dtype = std::string(ninfer::glm53::to_string(tensor.dtype));
        observed.shape = tensor.shape;
        observed.has_metadata = true;
        (void)tensor.mcg_payload;
        catalog.tensors.emplace(tensor.source_name, std::move(observed));
    }
    return catalog;
}

}  // namespace

int main() {
    using namespace ninfer::glm53;
    expect(sha256_hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "empty sha256");
    expect(sha256_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "abc sha256");

    expect(validate_model_spec(glm53_flash_spec()).empty(), "model spec");
    expect(validate_vision_spec(glm53_flash_vision_spec()).empty(), "vision spec");
    expect(validate_dflash2_spec(glm53_flash_dflash2_spec()).empty(), "dflash spec");

    const auto target = expected_glm53_flash_tensors();
    const auto draft = expected_dflash2_tensors();
    expect(validate_parameter_catalog(target).empty(), "target catalog");
    expect(validate_parameter_catalog(draft).empty(), "draft catalog");
    expect_eq(target.size(), 148498U, "target tensors");
    expect_eq(draft.size(), 81U, "draft tensors");

    std::size_t native = 0;
    std::size_t nvfp4 = 0;
    std::size_t fp8_block = 0;
    std::size_t f32 = 0;
    for (const auto& tensor : target) {
        if (tensor.storage == StorageClass::kNativeBf16 || tensor.storage == StorageClass::kNativeF32) ++native;
        else if (tensor.storage == StorageClass::kFp8BlockWeight || tensor.storage == StorageClass::kFp8BlockScaleBf16) {
            ++fp8_block;
        } else {
            ++nvfp4;
        }
        if (tensor.storage == StorageClass::kNativeF32) ++f32;
    }
    expect_eq(native, 1618U, "native tensors");
    expect_eq(nvfp4, 145152U, "nvfp4 tensors");
    expect_eq(fp8_block, 1728U, "fp8 block tensors");
    expect_eq(f32, 291U, "fp32 tensors");

    const auto* kda_q = find_logical(target, "language.layer.0.kda.q_proj");
    const auto* kda_log = find_logical(target, "language.layer.0.kda.a_log");
    const auto* no_mla = find_logical(target, "language.layer.0.mla.q_a_proj");
    expect(kda_q != nullptr && kda_q->source_name == "model.language_model.layers.0.self_attn.q_proj.weight",
           "kda q source");
    expect(kda_q->dtype == DType::kBf16 && kda_q->shape == std::vector<std::int64_t>({8192, 4096}), "kda q shape");
    expect(kda_log != nullptr && kda_log->dtype == DType::kF32 && kda_log->shape == std::vector<std::int64_t>({64}),
           "kda a_log");
    expect(no_mla == nullptr, "layer 0 is not sparse mla");

    const auto* mla_q = find_logical(target, "language.layer.3.mla.q_b_proj");
    const auto* packed = find_logical(target, "language.layer.3.moe.expert.0.gate.packed");
    const auto* scale = find_logical(target, "language.layer.3.moe.expert.0.gate.weight_scale");
    const auto* global_scale = find_logical(target, "language.layer.3.moe.expert.0.gate.weight_global_scale");
    const auto* dense = find_logical(target, "language.layer.0.ffn.gate_proj");
    const auto* kda_on_mla = find_logical(target, "language.layer.3.kda.a_log");
    const auto* o_mla = find_logical(target, "language.layer.3.mixer.o_proj");
    const auto* o_kda = find_logical(target, "language.layer.44.mixer.o_proj");
    expect(mla_q != nullptr && mla_q->shape == std::vector<std::int64_t>({16384, 1536}), "mla q_b");
    expect(packed != nullptr && packed->dtype == DType::kU8 && packed->shape == std::vector<std::int64_t>({2048, 2048}) &&
               packed->source_name == "model.language_model.layers.3.mlp.experts.0.gate_proj.weight_packed",
           "nvfp4 packed");
    expect(scale != nullptr && scale->dtype == DType::kF8E4M3 && scale->shape == std::vector<std::int64_t>({2048, 256}),
           "nvfp4 scale");
    expect(global_scale != nullptr && global_scale->dtype == DType::kF32 &&
               global_scale->shape == std::vector<std::int64_t>({1}),
           "nvfp4 global scale");
    expect(dense != nullptr && dense->dtype == DType::kBf16 && dense->shape == std::vector<std::int64_t>({12288, 4096}),
           "dense ffn stays bf16");
    expect(kda_on_mla == nullptr, "sparse layer has no kda parameters");
    expect(o_mla != nullptr && o_mla->shape == std::vector<std::int64_t>({4096, 16384}), "mla o_proj");
    expect(o_kda != nullptr && o_kda->shape == std::vector<std::int64_t>({4096, 8192}), "kda o_proj");

    const auto* nextn = find_logical(target, "language.nextn.eh_proj");
    const auto* nextn_mhc = find_logical(target, "language.nextn.mhc.attn.base");
    const auto* nextn_expert = find_logical(target, "language.nextn.moe.expert.0.gate.weight");
    const auto* nextn_scale = find_logical(target, "language.nextn.moe.expert.0.gate.weight_scale");
    const auto* nextn_packed = find_logical(target, "language.nextn.moe.expert.0.gate.packed");
    expect(nextn != nullptr && nextn->source_name == "model.language_model.layers.45.eh_proj.weight" &&
               nextn->shape == std::vector<std::int64_t>({4096, 8192}),
           "nextn eh_proj");
    expect(nextn_mhc == nullptr, "nextn has no mhc");
    expect(nextn_expert != nullptr && nextn_expert->dtype == DType::kF8E4M3 &&
               nextn_expert->shape == std::vector<std::int64_t>({2048, 4096}),
           "nextn expert fp8");
    expect(nextn_scale != nullptr && nextn_scale->dtype == DType::kBf16 &&
               nextn_scale->shape == std::vector<std::int64_t>({16, 32}),
           "nextn expert fp8 scale");
    expect(nextn_packed == nullptr, "nextn experts are not nvfp4 packed");

    const auto* qkv = find_logical(target, "vision.block.0.attn.qkv");
    expect(qkv != nullptr && qkv->shape == std::vector<std::int64_t>({3072, 1024}) &&
               qkv->source_name == "model.visual.blocks.0.attn.qkv.weight",
           "vision qkv");

    const auto* fc = find_logical(draft, "draft.fc");
    const auto* draft_q = find_logical(draft, "draft.layer.0.attn.q_proj");
    const auto* absent = find_logical(draft, "draft.layer.5.attn.q_proj");
    expect(fc != nullptr && fc->source_name == "fc.weight" && fc->shape == std::vector<std::int64_t>({4096, 20480}),
           "draft fc");
    expect(draft_q != nullptr && draft_q->shape == std::vector<std::int64_t>({4096, 4096}), "draft q");
    expect(absent == nullptr, "draft has 5 layers");

    const std::string target_hash = logical_catalog_sha256(target);
    const std::string draft_hash = logical_catalog_sha256(draft);
    constexpr std::string_view kTargetHash = "674ff493a2911b72d94ddda1eaedc742b6895a0b1235d39656b765516e1971e8";
    constexpr std::string_view kDraftHash = "ccad9b633dc4090c50c6d1667778402cc634eb08bd7d87ac22e2abe209626b44";
    if (target_hash != kTargetHash || draft_hash != kDraftHash) {
        std::cerr << "target catalog " << target_hash << '\n' << "draft catalog " << draft_hash << '\n';
        return 1;
    }

    const auto round_trip = bind_catalog(CheckpointKind::kGlm53Nvfp4Target, target, catalog_from(target), true);
    expect(round_trip.complete, "target round trip");
    expect(round_trip.bound == target.size() && round_trip.missing == 0 && round_trip.unexpected == 0, "round counts");

    auto missing = catalog_from(draft);
    missing.tensors.erase("norm.weight");
    const auto missing_report = bind_catalog(CheckpointKind::kDflash2Draft, draft, missing, true);
    expect(!missing_report.complete && missing_report.missing == 1, "draft missing norm");

    auto extra = catalog_from(draft);
    extra.tensors.emplace("not.a.parameter", ObservedTensor{});
    const auto extra_report = bind_catalog(CheckpointKind::kDflash2Draft, draft, extra, true);
    expect(!extra_report.complete && extra_report.unexpected == 1, "draft extra tensor");

    ExpectedTensor duplicated = draft.front();
    auto duplicated_catalog = std::vector<ExpectedTensor>{draft.front(), duplicated};
    expect(!validate_parameter_catalog(duplicated_catalog).empty(), "duplicate source rejected");

    const auto temp = std::filesystem::temp_directory_path() / "ninfer-glm53-binding-test";
    std::filesystem::remove_all(temp);
    std::filesystem::create_directories(temp);
    const char data[] = {'\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\xed', '\x1f', '\xac', '\xcb'};
    write_safetensors(temp / "tiny.safetensors",
                      R"({"tiny.weight":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"tiny.mcg":{"dtype":"I32","shape":[1],"data_offsets":[8,12]}})",
                      std::string_view(data, sizeof(data)));
    const auto loaded = read_safetensors_file(temp / "tiny.safetensors", true);
    expect_eq(loaded.tensors.size(), 2U, "tiny tensors");
    expect(loaded.tensors.at("tiny.weight").dtype == "BF16" &&
               loaded.tensors.at("tiny.weight").shape == std::vector<std::int64_t>({2}),
           "tiny weight");
    expect(loaded.tensors.at("tiny.mcg").has_mcg_value, "safetensors payload offset is readable");

    write_safetensors(temp / "truncated.safetensors",
                      R"({"w":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}})", "");
    expect_throws([&] { (void)read_safetensors_file(temp / "truncated.safetensors", false); },
                  "truncated payload");

    write_safetensors(temp / "wrong-size.safetensors",
                      R"({"w":{"dtype":"BF16","shape":[2],"data_offsets":[0,2]}})", std::string_view(data, 2));
    expect_throws([&] { (void)read_safetensors_file(temp / "wrong-size.safetensors", false); },
                  "dtype/shape payload size mismatch");

    const char overlap_data[] = {'a', 'b', 'c'};
    write_safetensors(temp / "overlap.safetensors",
                      R"({"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"b":{"dtype":"U8","shape":[2],"data_offsets":[1,3]}})",
                      std::string_view(overlap_data, sizeof(overlap_data)));
    expect_throws([&] { (void)read_safetensors_file(temp / "overlap.safetensors", false); },
                  "overlapping payloads");

    std::string nested = R"({"__metadata__":{"audit":)";
    nested.append(80, '[');
    nested += "0";
    nested.append(80, ']');
    nested += R"(},"w":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}})";
    write_safetensors(temp / "deep.safetensors", nested, std::string_view(data, 2));
    expect_throws([&] { (void)read_safetensors_file(temp / "deep.safetensors", false); },
                  "invalid/deep metadata rejected");

    std::string deep_unknown = R"({"w":{"dtype":"BF16","shape":[1],"data_offsets":[0,2],"unknown":)";
    deep_unknown.append(80, '[');
    deep_unknown += "0";
    deep_unknown.append(80, ']');
    deep_unknown += "}}";
    write_safetensors(temp / "deep-unknown.safetensors", deep_unknown, std::string_view(data, 2));
    expect_throws([&] { (void)read_safetensors_file(temp / "deep-unknown.safetensors", false); },
                  "excessive JSON nesting in unknown tensor field");

    const auto name_dir = temp / "names";
    std::filesystem::create_directories(name_dir);
    write_text(name_dir / "config.json", R"({
  "architectures": ["Glm5NextForConditionalGeneration"],
  "quantization_config": {"quant_method": "compressed-tensors", "config_groups": {"group_0": {"format": "nvfp4-pack-quantized"}}},
  "text_config": {"num_hidden_layers": 45, "n_routed_experts": 288, "num_nextn_predict_layers": 1},
  "vision_config": {"depth": 24}
})");
    write_text(name_dir / "model.safetensors.index.json",
               R"({"metadata":{"total_size":1},"weight_map":{"lm_head.weight":"model.safetensors"}})");
    write_text(name_dir / "model.safetensors", "");
    const auto names = bind_checkpoint_directory(name_dir, true);
    expect(names.kind == CheckpointKind::kGlm53Nvfp4Target, "names kind");
    expect_eq(names.observed_tensors, 1U, "names observed");
    expect_eq(names.bound, 1U, "names bound");
    expect_eq(names.missing, target.size() - 1U, "names missing");
    expect(!names.complete && !names.shapes_checked && names.io_errors == 0, "names incomplete without io errors");

    write_text(name_dir / "model.safetensors.index.json",
               R"({"weight_map":{"lm_head.weight":"a.safetensors","lm_head.weight":"b.safetensors"}})");
    const auto duplicate_index = bind_checkpoint_directory(name_dir, true);
    expect(!duplicate_index.complete && duplicate_index.io_errors > 0, "duplicate weight_map key rejected");

    write_text(name_dir / "model.safetensors.index.json",
               R"({"weight_map":{"lm_head.weight":"../outside.safetensors"}})");
    const auto unsafe_path = bind_checkpoint_directory(name_dir, true);
    expect(!unsafe_path.complete && unsafe_path.io_errors > 0, "unsafe shard path rejected");

    std::filesystem::remove_all(temp);
    return 0;
}
