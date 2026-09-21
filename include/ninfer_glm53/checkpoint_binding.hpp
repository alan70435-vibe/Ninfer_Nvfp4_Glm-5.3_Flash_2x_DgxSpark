#pragma once

#include "ninfer_glm53/parameter_schema.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace ninfer::glm53 {

enum class CheckpointKind : std::uint8_t {
    kGlm53Nvfp4Target,
    kDflash2Draft,
};

struct ObservedTensor {
    std::string dtype;
    std::vector<std::int64_t> shape;
    std::string shard;
    bool has_metadata{false};
    bool has_mcg_value{false};
    std::uint32_t mcg_value{0};
};

struct TensorCatalog {
    std::unordered_map<std::string, ObservedTensor> tensors;
};

struct BindingReport {
    bool complete{false};
    bool shapes_checked{false};
    bool mcg_payload_checked{false};
    CheckpointKind kind{CheckpointKind::kGlm53Nvfp4Target};
    std::size_t expected_tensors{0};
    std::size_t observed_tensors{0};
    std::size_t bound{0};
    std::size_t missing{0};
    std::size_t unexpected{0};
    std::size_t dtype_mismatches{0};
    std::size_t shape_mismatches{0};
    std::size_t mcg_mismatches{0};
    std::size_t schema_errors{0};
    std::size_t io_errors{0};
    std::uint64_t weight_bytes{0};
    std::vector<std::string> errors;
    std::vector<std::string> warnings;
    std::string config_sha256;
    std::string index_sha256;
    std::string logical_catalog_sha256;
    std::string observed_names_sha256;
    std::string mirror_upstream_revision;
    std::string source_model_revision;
    std::string snapshot_revision;
};

[[nodiscard]] std::string_view to_string(CheckpointKind kind) noexcept;
[[nodiscard]] std::string sha256_hex(std::string_view bytes);

[[nodiscard]] TensorCatalog read_safetensors_file(const std::filesystem::path& path, bool read_mcg_payload);

[[nodiscard]] BindingReport bind_catalog(CheckpointKind kind, const std::vector<ExpectedTensor>& expected,
                                         const TensorCatalog& observed, bool metadata_required);

[[nodiscard]] BindingReport bind_checkpoint_directory(const std::filesystem::path& directory, bool names_only);

[[nodiscard]] std::string binding_receipt_json(const BindingReport& report);

}  // namespace ninfer::glm53
