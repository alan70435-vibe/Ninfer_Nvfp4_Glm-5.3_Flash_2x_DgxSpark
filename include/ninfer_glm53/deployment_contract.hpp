#pragma once

#include "ninfer_glm53/model_spec.hpp"

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::glm53 {

struct ValidationReport {
    std::vector<std::string> errors;
    std::vector<std::string> warnings;

    [[nodiscard]] bool ok() const noexcept { return errors.empty(); }
};

struct DeploymentContract {
    std::string head_ip;
    std::string worker_ip;
    std::string head_cx7_if;
    std::string worker_cx7_if;
    std::string head_cx7_ib;
    std::string worker_cx7_ib;
    std::uint32_t nccl_ib_gid_index{};
    std::uint32_t head_gid{};
    std::uint32_t worker_gid{};

    std::string model;
    std::string model_revision;
    std::string served_model_name;
    std::string quantization;
    std::string kv_cache_dtype;

    std::string speculative_method;
    std::string dflash_model;
    std::string dflash_revision;
    std::uint32_t dflash_tokens{};
    std::uint32_t dflash_draft_tp{};

    std::uint32_t tensor_parallel{};
    std::uint32_t nodes{};
    std::uint32_t master_port{};
    std::uint32_t api_port{};
    std::uint32_t max_model_len{};
    std::uint32_t max_num_seqs{};
    std::uint32_t max_num_batched_tokens{};
    double gpu_memory_utilization{};

    bool enforce_eager{};
    bool use_host_nccl{};

    std::string indexer_workspace;
    std::string adaptive_k;
    std::string adaptive_k_set;
};

using EnvMap = std::map<std::string, std::string, std::less<>>;

[[nodiscard]] EnvMap parse_env_file(const std::filesystem::path& path);
[[nodiscard]] DeploymentContract deployment_from_env(const EnvMap& env);
[[nodiscard]] DeploymentContract load_deployment_contract(const std::filesystem::path& path);
[[nodiscard]] ValidationReport validate_deployment_contract(
    const DeploymentContract& deployment, const ModelSpec& model);

}  // namespace ninfer::glm53
