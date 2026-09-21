#include "ninfer_glm53/checkpoint_binding.hpp"

#include "json_scan.hpp"
#include "sha256.hpp"

#include <algorithm>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>

namespace ninfer::glm53 {
namespace {

constexpr std::size_t kMaxListedErrors = 24;
constexpr std::uint64_t kMaxSafetensorsHeader = 256ULL * 1024ULL * 1024ULL;

void add_error(BindingReport& report, std::string message) {
    if (report.errors.size() < kMaxListedErrors) report.errors.push_back(std::move(message));
}

std::string read_text(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read " + path.string());
    std::ostringstream buffer;
    buffer << input.rdbuf();
    if (input.bad()) throw std::runtime_error("failed while reading " + path.string());
    return buffer.str();
}

std::string json_escape(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (unsigned char ch : value) {
        switch (ch) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (ch < 0x20U) {
                    constexpr char kDigits[] = "0123456789abcdef";
                    out += "\\u00";
                    out.push_back(kDigits[ch >> 4U]);
                    out.push_back(kDigits[ch & 0x0FU]);
                } else {
                    out.push_back(static_cast<char>(ch));
                }
        }
    }
    return out;
}

std::string snapshot_revision(const std::filesystem::path& directory) {
    std::error_code error;
    auto cursor = std::filesystem::weakly_canonical(directory, error);
    if (error) cursor = directory;
    while (!cursor.empty()) {
        const auto name = cursor.filename().string();
        if (detail::is_hex40(name)) return name;
        const auto parent = cursor.parent_path();
        if (parent == cursor) break;
        cursor = parent;
    }
    return {};
}

struct ParsedTensor {
    std::string name;
    ObservedTensor observed;
    std::int64_t data_begin{-1};
    std::int64_t data_end{-1};
};

void parse_tensor_object(json_scan::Cursor& cursor, ParsedTensor& parsed) {
    cursor.expect('{');
    if (cursor.peek() == '}') {
        cursor.expect('}');
        return;
    }
    while (true) {
        const auto key = cursor.parse_string();
        cursor.expect(':');
        if (key == "dtype") {
            parsed.observed.dtype = cursor.parse_string();
        } else if (key == "shape") {
            cursor.expect('[');
            if (cursor.peek() != ']') {
                while (true) {
                    parsed.observed.shape.push_back(cursor.parse_int());
                    if (cursor.peek() == ',') {
                        cursor.expect(',');
                        continue;
                    }
                    break;
                }
            }
            cursor.expect(']');
        } else if (key == "data_offsets") {
            cursor.expect('[');
            parsed.data_begin = cursor.parse_int();
            cursor.expect(',');
            parsed.data_end = cursor.parse_int();
            cursor.expect(']');
        } else {
            cursor.skip_value();
        }
        const char next = cursor.peek();
        if (next == ',') {
            cursor.expect(',');
            continue;
        }
        cursor.expect('}');
        return;
    }
}

std::vector<ParsedTensor> parse_safetensors_header(std::string_view header) {
    json_scan::Cursor cursor(header);
    cursor.expect('{');
    std::vector<ParsedTensor> tensors;
    if (cursor.peek() == '}') return tensors;
    while (true) {
        const auto key = cursor.parse_string();
        cursor.expect(':');
        if (key == "__metadata__") {
            cursor.skip_value();
        } else {
            ParsedTensor parsed;
            parsed.name = key;
            parse_tensor_object(cursor, parsed);
            parsed.observed.has_metadata = true;
            tensors.push_back(std::move(parsed));
        }
        const char next = cursor.peek();
        if (next == ',') {
            cursor.expect(',');
            continue;
        }
        cursor.expect('}');
        break;
    }
    return tensors;
}

std::unordered_map<std::string, std::string> parse_weight_map(std::string_view json) {
    json_scan::Cursor cursor(json);
    cursor.expect('{');
    std::unordered_map<std::string, std::string> weight_map;
    if (cursor.peek() == '}') return weight_map;
    while (true) {
        const auto key = cursor.parse_string();
        cursor.expect(':');
        if (key == "weight_map") {
            cursor.expect('{');
            if (cursor.peek() != '}') {
                while (true) {
                    auto name = cursor.parse_string();
                    cursor.expect(':');
                    auto shard = cursor.parse_string();
                    weight_map.emplace(std::move(name), std::move(shard));
                    const char next = cursor.peek();
                    if (next == ',') {
                        cursor.expect(',');
                        continue;
                    }
                    cursor.expect('}');
                    break;
                }
            } else {
                cursor.expect('}');
            }
        } else {
            cursor.skip_value();
        }
        const char next = cursor.peek();
        if (next == ',') {
            cursor.expect(',');
            continue;
        }
        cursor.expect('}');
        break;
    }
    return weight_map;
}

bool require_string(BindingReport& report, std::string_view json, std::string_view key, std::string_view expected) {
    const auto actual = json_scan::find_string(json, key);
    if (!actual || *actual != expected) {
        ++report.io_errors;
        add_error(report, "config field " + std::string(key) + " must be " + std::string(expected));
        return false;
    }
    return true;
}

bool require_int(BindingReport& report, std::string_view json, std::string_view key, std::int64_t expected) {
    const auto actual = json_scan::find_int(json, key);
    if (!actual || *actual != expected) {
        ++report.io_errors;
        add_error(report, "config field " + std::string(key) + " must be " + std::to_string(expected));
        return false;
    }
    return true;
}

void check_config(BindingReport& report, CheckpointKind kind, std::string_view json) {
    if (kind == CheckpointKind::kGlm53Nvfp4Target) {
        if (json.find("\"Glm5NextForConditionalGeneration\"") == std::string_view::npos) {
            ++report.io_errors;
            add_error(report, "config architecture must be Glm5NextForConditionalGeneration");
        }
        const auto method = json_scan::find_string(json, "quant_method");
        if (method && *method == "modelopt") {
            ++report.io_errors;
            add_error(report, "ModelOpt NVFP4 is not the product checkpoint; use compressed-tensors RedHatAI/GLM-5.3-Flash-NVFP4");
        }
        require_string(report, json, "quant_method", "compressed-tensors");
        if (json.find("nvfp4-pack-quantized") == std::string_view::npos) {
            ++report.io_errors;
            add_error(report, "config must declare nvfp4-pack-quantized expert weights");
        }
        require_int(report, json, "num_hidden_layers", 45);
        require_int(report, json, "n_routed_experts", 288);
        require_int(report, json, "num_nextn_predict_layers", 1);
        require_int(report, json, "depth", 24);
        return;
    }

    if (json.find("\"DFlash2DraftModel\"") == std::string_view::npos) {
        ++report.io_errors;
        add_error(report, "config architecture must be DFlash2DraftModel");
    }
    require_int(report, json, "num_hidden_layers", 5);
    require_int(report, json, "hidden_size", 4096);
    require_int(report, json, "vocab_size", 154880);
    require_int(report, json, "block_size", 8);
    require_int(report, json, "conv_kernel_size", 2);
    require_int(report, json, "conv_group_size", 16);
    require_int(report, json, "selector_rank", 256);
    const auto ids = json_scan::find_int_array(json, "target_layer_ids");
    const std::vector<std::int64_t> expected{5, 14, 24, 33, 42};
    if (!ids || *ids != expected) {
        ++report.io_errors;
        add_error(report, "DFlash2 target_layer_ids must be [5, 14, 24, 33, 42]");
    }
}

CheckpointKind detect_kind(std::string_view config, BindingReport& report) {
    const bool target = config.find("\"Glm5NextForConditionalGeneration\"") != std::string_view::npos;
    const bool draft = config.find("\"DFlash2DraftModel\"") != std::string_view::npos;
    if (target == draft) {
        ++report.io_errors;
        add_error(report, "config.json must declare exactly one of Glm5NextForConditionalGeneration or DFlash2DraftModel");
        return CheckpointKind::kGlm53Nvfp4Target;
    }
    return target ? CheckpointKind::kGlm53Nvfp4Target : CheckpointKind::kDflash2Draft;
}

void read_mcg_payload(std::ifstream& input, std::uint64_t data_start, ParsedTensor& parsed) {
    if (parsed.observed.dtype != "I32" || parsed.observed.shape.size() != 1U || parsed.observed.shape[0] != 1) {
        return;
    }
    if (parsed.data_begin < 0 || parsed.data_end < parsed.data_begin) return;
    const auto nbytes = static_cast<std::uint64_t>(parsed.data_end - parsed.data_begin);
    if (nbytes != 4U) return;
    const auto position = data_start + static_cast<std::uint64_t>(parsed.data_begin);
    input.seekg(static_cast<std::streamoff>(position));
    unsigned char bytes[4]{};
    input.read(reinterpret_cast<char*>(bytes), 4);
    if (input.gcount() != 4) {
        input.clear();
        return;
    }
    parsed.observed.has_mcg_value = true;
    parsed.observed.mcg_value = static_cast<std::uint32_t>(bytes[0]) |
                                (static_cast<std::uint32_t>(bytes[1]) << 8U) |
                                (static_cast<std::uint32_t>(bytes[2]) << 16U) |
                                (static_cast<std::uint32_t>(bytes[3]) << 24U);
}

TensorCatalog read_shard(const std::filesystem::path& path, const std::string& shard_name, bool load_mcg_payload,
                         BindingReport* report) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        if (report != nullptr) {
            ++report->io_errors;
            add_error(*report, "cannot open shard " + shard_name);
        } else {
            throw std::runtime_error("cannot open shard " + path.string());
        }
        return {};
    }
    unsigned char length_bytes[8]{};
    input.read(reinterpret_cast<char*>(length_bytes), 8);
    if (input.gcount() != 8) throw std::runtime_error("short safetensors header length in " + shard_name);
    std::uint64_t header_len = 0;
    for (int i = 0; i < 8; ++i) {
        header_len |= static_cast<std::uint64_t>(length_bytes[i]) << static_cast<unsigned>(i * 8);
    }
    if (header_len == 0U || header_len > kMaxSafetensorsHeader) {
        throw std::runtime_error("invalid safetensors header length in " + shard_name);
    }
    std::string header(static_cast<std::size_t>(header_len), '\0');
    input.read(header.data(), static_cast<std::streamsize>(header_len));
    if (static_cast<std::uint64_t>(input.gcount()) != header_len) {
        throw std::runtime_error("truncated safetensors header in " + shard_name);
    }
    const auto data_start = 8U + header_len;
    auto parsed = parse_safetensors_header(header);
    TensorCatalog catalog;
    catalog.tensors.reserve(parsed.size());
    for (auto& tensor : parsed) {
        tensor.observed.shard = shard_name;
        if (load_mcg_payload) read_mcg_payload(input, data_start, tensor);
        if (!catalog.tensors.emplace(tensor.name, tensor.observed).second) {
            if (report != nullptr) {
                ++report->io_errors;
                add_error(*report, "duplicate tensor in shard header: " + tensor.name);
            } else {
                throw std::runtime_error("duplicate tensor in shard header: " + tensor.name);
            }
        }
    }
    return catalog;
}

std::string names_sha256(const TensorCatalog& catalog) {
    std::vector<std::string> names;
    names.reserve(catalog.tensors.size());
    for (const auto& entry : catalog.tensors) names.push_back(entry.first);
    std::sort(names.begin(), names.end());
    detail::Sha256 hasher;
    for (const auto& name : names) {
        hasher.update(name);
        hasher.update("\n");
    }
    return hasher.hex_digest();
}

void finish(BindingReport& report) {
    report.complete = report.io_errors == 0 && report.schema_errors == 0 && report.missing == 0 &&
                      report.unexpected == 0 && report.dtype_mismatches == 0 && report.shape_mismatches == 0 &&
                      report.mcg_mismatches == 0;
}

}  // namespace

std::string sha256_hex(std::string_view bytes) { return detail::sha256_hex(bytes); }

std::string_view to_string(CheckpointKind kind) noexcept {
    switch (kind) {
        case CheckpointKind::kGlm53Nvfp4Target:
            return "glm53_flash_nvfp4_target";
        case CheckpointKind::kDflash2Draft:
            return "dflash2_draft";
    }
    return "unknown";
}

TensorCatalog read_safetensors_file(const std::filesystem::path& path, bool read_mcg_payload) {
    return read_shard(path, path.filename().string(), read_mcg_payload, nullptr);
}

BindingReport bind_catalog(CheckpointKind kind, const std::vector<ExpectedTensor>& expected,
                           const TensorCatalog& observed, bool metadata_required) {
    BindingReport report;
    report.kind = kind;
    report.expected_tensors = expected.size();
    report.observed_tensors = observed.tensors.size();
    report.shapes_checked = metadata_required;
    report.logical_catalog_sha256 = logical_catalog_sha256(expected);
    report.observed_names_sha256 = names_sha256(observed);
    bool any_mcg = false;
    for (const auto& tensor : expected) {
        if (tensor.mcg_payload) any_mcg = true;
    }
    report.mcg_payload_checked = metadata_required && any_mcg;

    const auto schema_errors = validate_parameter_catalog(expected);
    report.schema_errors = schema_errors.size();
    for (const auto& error : schema_errors) add_error(report, error);

    std::unordered_map<std::string, std::size_t> by_source;
    by_source.reserve(expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) by_source.emplace(expected[i].source_name, i);

    std::vector<char> seen(expected.size(), 0);
    for (const auto& [name, observed_tensor] : observed.tensors) {
        const auto found = by_source.find(name);
        if (found == by_source.end()) {
            ++report.unexpected;
            add_error(report, "unexpected tensor " + name);
            continue;
        }
        seen[found->second] = 1;
        ++report.bound;
        const auto& required = expected[found->second];
        if (!observed_tensor.has_metadata) {
            if (metadata_required) {
                ++report.dtype_mismatches;
                add_error(report, "missing dtype and shape for " + name);
            }
            continue;
        }
        if (observed_tensor.dtype != to_string(required.dtype)) {
            ++report.dtype_mismatches;
            add_error(report, "dtype mismatch for " + required.logical_id + ": expected " +
                                  std::string(to_string(required.dtype)) + ", got " + observed_tensor.dtype);
        }
        if (observed_tensor.shape != required.shape) {
            ++report.shape_mismatches;
            add_error(report, "shape mismatch for " + required.logical_id + ": expected [" +
                                  shape_to_string(required.shape) + "], got [" +
                                  shape_to_string(observed_tensor.shape) + "]");
        }
        if (required.mcg_payload) {
            ++report.mcg_mismatches;
            add_error(report, "NVFP4 binding does not accept an MCG payload for " + required.logical_id);
        }
    }
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (seen[i] == 0) {
            ++report.missing;
            add_error(report, "missing tensor " + expected[i].source_name);
        }
    }
    finish(report);
    return report;
}

BindingReport bind_checkpoint_directory(const std::filesystem::path& directory, bool names_only) {
    BindingReport report;
    const auto root = std::filesystem::absolute(directory);
    report.snapshot_revision = snapshot_revision(root);
    const auto config_path = root / "config.json";
    if (!std::filesystem::is_regular_file(config_path)) {
        ++report.io_errors;
        add_error(report, "missing config.json");
        finish(report);
        return report;
    }
    const auto config = read_text(config_path);
    report.config_sha256 = detail::sha256_hex(config);
    report.kind = detect_kind(config, report);
    check_config(report, report.kind, config);

    if (const auto mirror = root / "MIRROR.json"; std::filesystem::is_regular_file(mirror)) {
        const auto text = read_text(mirror);
        if (const auto revision = json_scan::find_string(text, "upstream_revision")) {
            report.mirror_upstream_revision = *revision;
        }
    }
    std::optional<std::string> receipt_config_sha;
    std::optional<std::string> receipt_index_sha;
    std::optional<std::int64_t> receipt_tensor_count;
    if (const auto receipt = root / "materialization-receipt.json"; std::filesystem::is_regular_file(receipt)) {
        const auto text = read_text(receipt);
        if (const auto revision = json_scan::find_string(text, "source_model_revision")) {
            report.source_model_revision = *revision;
        }
        receipt_config_sha = json_scan::find_string(text, "config_sha256");
        receipt_index_sha = json_scan::find_string(text, "index_sha256");
        receipt_tensor_count = json_scan::find_int(text, "output_tensor_count");
    }
    if (receipt_config_sha && *receipt_config_sha != report.config_sha256) {
        ++report.io_errors;
        add_error(report, "config.json sha256 does not match materialization-receipt.json");
    }

    const auto index_path = root / "model.safetensors.index.json";
    std::unordered_map<std::string, std::string> weight_map;
    if (std::filesystem::is_regular_file(index_path)) {
        const auto index_text = read_text(index_path);
        report.index_sha256 = detail::sha256_hex(index_text);
        try {
            weight_map = parse_weight_map(index_text);
        } catch (const std::exception& error) {
            ++report.io_errors;
            add_error(report, std::string("weight map parse failed: ") + error.what());
        }
        if (weight_map.empty()) {
            ++report.io_errors;
            add_error(report, "weight_map is empty");
        }
        if (receipt_index_sha && *receipt_index_sha != report.index_sha256) {
            ++report.io_errors;
            add_error(report, "model.safetensors.index.json sha256 does not match materialization-receipt.json");
        }
    }

    std::vector<std::string> shard_names;
    if (!weight_map.empty()) {
        std::unordered_map<std::string, std::uint64_t> seen_shards;
        for (const auto& entry : weight_map) seen_shards.emplace(entry.second, 0U);
        shard_names.reserve(seen_shards.size());
        for (const auto& entry : seen_shards) shard_names.push_back(entry.first);
        std::sort(shard_names.begin(), shard_names.end());
    } else if (report.index_sha256.empty()) {
        for (const auto& entry : std::filesystem::directory_iterator(root)) {
            if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
                shard_names.push_back(entry.path().filename().string());
            }
        }
        std::sort(shard_names.begin(), shard_names.end());
        if (shard_names.empty()) {
            ++report.io_errors;
            add_error(report, "no weight index and no safetensors shards");
        }
    }

    for (const auto& shard : shard_names) {
        const auto path = root / shard;
        if (!std::filesystem::is_regular_file(path)) {
            if (!names_only) {
                ++report.io_errors;
                add_error(report, "missing shard " + shard);
            }
            continue;
        }
        report.weight_bytes += static_cast<std::uint64_t>(std::filesystem::file_size(path));
    }
    if (names_only && report.weight_bytes == 0U && !shard_names.empty()) {
        report.warnings.emplace_back("names-only bind did not open weight shards; shapes were not checked");
    }

    TensorCatalog catalog;
    if (names_only && !weight_map.empty()) {
        catalog.tensors.reserve(weight_map.size());
        for (const auto& [name, shard] : weight_map) {
            ObservedTensor observed;
            observed.shard = shard;
            catalog.tensors.emplace(name, std::move(observed));
        }
    } else if (report.io_errors == 0 || !shard_names.empty()) {
        for (const auto& shard : shard_names) {
            const auto path = root / shard;
            if (!std::filesystem::is_regular_file(path)) continue;
            TensorCatalog shard_catalog;
            try {
                shard_catalog = read_shard(path, shard, !names_only, &report);
            } catch (const std::exception& error) {
                ++report.io_errors;
                add_error(report, error.what());
                continue;
            }
            for (auto& [name, observed] : shard_catalog.tensors) {
                if (!catalog.tensors.emplace(name, std::move(observed)).second) {
                    ++report.io_errors;
                    add_error(report, "duplicate tensor across shards: " + name);
                }
            }
        }
        if (!weight_map.empty()) {
            for (const auto& [name, owner] : weight_map) {
                const auto found = catalog.tensors.find(name);
                if (found == catalog.tensors.end()) {
                    ++report.io_errors;
                    add_error(report, "indexed tensor missing from shard headers: " + name);
                } else if (found->second.shard != owner) {
                    ++report.io_errors;
                    add_error(report, "indexed tensor " + name + " is in " + found->second.shard +
                                          " but weight_map says " + owner);
                }
            }
            for (const auto& entry : catalog.tensors) {
                if (!weight_map.contains(entry.first)) {
                    ++report.io_errors;
                    add_error(report, "header tensor missing from weight_map: " + entry.first);
                }
            }
        }
    }

    if (receipt_tensor_count && static_cast<std::uint64_t>(*receipt_tensor_count) != catalog.tensors.size() &&
        !names_only) {
        ++report.io_errors;
        add_error(report, "observed tensor count does not match materialization-receipt output_tensor_count");
    }

    const auto expected = report.kind == CheckpointKind::kGlm53Nvfp4Target ? expected_glm53_flash_tensors()
                                                                           : expected_dflash2_tensors();
    const bool metadata_required = !names_only;
    auto bound = bind_catalog(report.kind, expected, catalog, metadata_required);
    bound.io_errors += report.io_errors;
    bound.warnings.insert(bound.warnings.end(), report.warnings.begin(), report.warnings.end());
    for (const auto& error : report.errors) add_error(bound, error);
    bound.weight_bytes = report.weight_bytes;
    bound.config_sha256 = std::move(report.config_sha256);
    bound.index_sha256 = std::move(report.index_sha256);
    bound.mirror_upstream_revision = std::move(report.mirror_upstream_revision);
    bound.source_model_revision = std::move(report.source_model_revision);
    bound.snapshot_revision = std::move(report.snapshot_revision);
    bound.kind = report.kind;
    finish(bound);
    return bound;
}

std::string binding_receipt_json(const BindingReport& report) {
    std::ostringstream out;
    out << "{\n";
    out << "  \"schema\": \"ninfer-glm53.binding-receipt.v1\",\n";
    out << "  \"kind\": \"" << to_string(report.kind) << "\",\n";
    out << "  \"complete\": " << (report.complete ? "true" : "false") << ",\n";
    out << "  \"shapes_checked\": " << (report.shapes_checked ? "true" : "false") << ",\n";
    out << "  \"mcg_payload_checked\": " << (report.mcg_payload_checked ? "true" : "false") << ",\n";
    out << "  \"expected_tensors\": " << report.expected_tensors << ",\n";
    out << "  \"observed_tensors\": " << report.observed_tensors << ",\n";
    out << "  \"bound\": " << report.bound << ",\n";
    out << "  \"missing\": " << report.missing << ",\n";
    out << "  \"unexpected\": " << report.unexpected << ",\n";
    out << "  \"dtype_mismatches\": " << report.dtype_mismatches << ",\n";
    out << "  \"shape_mismatches\": " << report.shape_mismatches << ",\n";
    out << "  \"mcg_mismatches\": " << report.mcg_mismatches << ",\n";
    out << "  \"schema_errors\": " << report.schema_errors << ",\n";
    out << "  \"io_errors\": " << report.io_errors << ",\n";
    out << "  \"weight_bytes\": " << report.weight_bytes << ",\n";
    out << "  \"config_sha256\": \"" << json_escape(report.config_sha256) << "\",\n";
    out << "  \"index_sha256\": \"" << json_escape(report.index_sha256) << "\",\n";
    out << "  \"logical_catalog_sha256\": \"" << json_escape(report.logical_catalog_sha256) << "\",\n";
    out << "  \"observed_names_sha256\": \"" << json_escape(report.observed_names_sha256) << "\",\n";
    out << "  \"mirror_upstream_revision\": \"" << json_escape(report.mirror_upstream_revision) << "\",\n";
    out << "  \"source_model_revision\": \"" << json_escape(report.source_model_revision) << "\",\n";
    out << "  \"snapshot_revision\": \"" << json_escape(report.snapshot_revision) << "\",\n";
    out << "  \"warnings\": [";
    for (std::size_t i = 0; i < report.warnings.size(); ++i) {
        if (i != 0U) out << ',';
        out << "\n    \"" << json_escape(report.warnings[i]) << "\"";
    }
    if (!report.warnings.empty()) out << "\n  ";
    out << "],\n";
    out << "  \"errors\": [";
    for (std::size_t i = 0; i < report.errors.size(); ++i) {
        if (i != 0U) out << ',';
        out << "\n    \"" << json_escape(report.errors[i]) << "\"";
    }
    if (!report.errors.empty()) out << "\n  ";
    out << "]\n";
    out << "}\n";
    return out.str();
}

}  // namespace ninfer::glm53
