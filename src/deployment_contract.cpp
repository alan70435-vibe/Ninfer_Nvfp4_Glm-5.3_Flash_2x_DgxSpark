#include "ninfer_glm53/deployment_contract.hpp"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace ninfer::glm53 {
namespace {

std::string trim(std::string value) {
    const auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

std::string strip_inline_comment(std::string value) {
    bool in_single = false;
    bool in_double = false;
    bool escaped = false;
    for (std::size_t i = 0; i < value.size(); ++i) {
        const char ch = value[i];
        if (escaped) {
            escaped = false;
            continue;
        }
        if (ch == '\\' && in_double) {
            escaped = true;
            continue;
        }
        if (ch == '\'' && !in_double) in_single = !in_single;
        if (ch == '"' && !in_single) in_double = !in_double;
        if (ch == '#' && !in_single && !in_double && (i == 0 || std::isspace(static_cast<unsigned char>(value[i - 1])))) {
            return trim(value.substr(0, i));
        }
    }
    return trim(value);
}

std::string unquote(std::string value) {
    value = trim(value);
    if (value.size() >= 2U) {
        const char first = value.front();
        const char last = value.back();
        if ((first == '"' && last == '"') || (first == '\'' && last == '\'')) {
            value = value.substr(1, value.size() - 2U);
        }
    }
    return value;
}

const std::string& require(const EnvMap& env, std::string_view key) {
    const auto it = env.find(key);
    if (it == env.end() || it->second.empty()) {
        throw std::runtime_error("missing required deployment key: " + std::string(key));
    }
    return it->second;
}

std::string optional_string(const EnvMap& env, std::string_view key) {
    if (const auto it = env.find(key); it != env.end()) return it->second;
    return {};
}

std::uint32_t parse_u32(const EnvMap& env, std::string_view key) {
    const auto& value = require(env, key);
    std::uint32_t parsed{};
    const auto* begin = value.data();
    const auto* end = value.data() + value.size();
    const auto [ptr, ec] = std::from_chars(begin, end, parsed);
    if (ec != std::errc{} || ptr != end) {
        throw std::runtime_error("invalid unsigned integer for " + std::string(key) + ": " + value);
    }
    return parsed;
}

std::uint32_t parse_u32_default(const EnvMap& env, std::string_view key, std::uint32_t fallback) {
    if (const auto it = env.find(key); it == env.end() || it->second.empty()) return fallback;
    return parse_u32(env, key);
}

double parse_double(const EnvMap& env, std::string_view key) {
    const auto& value = require(env, key);
    std::size_t consumed = 0;
    const double parsed = std::stod(value, &consumed);
    if (consumed != value.size()) {
        throw std::runtime_error("invalid floating-point value for " + std::string(key) + ": " + value);
    }
    return parsed;
}

bool parse_bool01_default(const EnvMap& env, std::string_view key, bool fallback) {
    if (const auto it = env.find(key); it == env.end() || it->second.empty()) return fallback;
    const auto& value = require(env, key);
    if (value == "1" || value == "true" || value == "TRUE") return true;
    if (value == "0" || value == "false" || value == "FALSE") return false;
    throw std::runtime_error("invalid boolean for " + std::string(key) + ": expected 0/1/true/false");
}

void require_equal(
    ValidationReport& report, std::string_view label, std::string_view actual, std::string_view expected) {
    if (actual != expected) {
        report.errors.emplace_back(std::string(label) + " must be '" + std::string(expected) + "' (got '" +
                                   std::string(actual) + "')");
    }
}

}  // namespace

EnvMap parse_env_file(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("unable to open deployment env: " + path.string());

    EnvMap env;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line_number == 1U && line.starts_with("\xEF\xBB\xBF")) line.erase(0, 3);
        line = trim(line);
        if (line.empty() || line.front() == '#') continue;
        if (line.starts_with("export ")) line = trim(line.substr(7));

        const auto eq = line.find('=');
        if (eq == std::string::npos) {
            throw std::runtime_error(path.string() + ":" + std::to_string(line_number) +
                                     ": expected KEY=VALUE");
        }
        auto key = trim(line.substr(0, eq));
        auto value = unquote(strip_inline_comment(line.substr(eq + 1)));
        if (key.empty()) {
            throw std::runtime_error(path.string() + ":" + std::to_string(line_number) + ": empty key");
        }
        if (!std::all_of(key.begin(), key.end(), [](unsigned char ch) {
                return std::isalnum(ch) || ch == '_';
            })) {
            throw std::runtime_error(path.string() + ":" + std::to_string(line_number) +
                                     ": invalid key: " + key);
        }
        env.insert_or_assign(std::move(key), std::move(value));
    }
    return env;
}

DeploymentContract deployment_from_env(const EnvMap& env) {
    DeploymentContract result;
    result.head_ip = require(env, "HEAD_IP");
    result.worker_ip = require(env, "WORKER_IP");
    result.head_cx7_if = require(env, "HEAD_CX7_IF");
    result.worker_cx7_if = require(env, "WORKER_CX7_IF");
    result.head_cx7_ib = require(env, "HEAD_CX7_IB");
    result.worker_cx7_ib = require(env, "WORKER_CX7_IB");
    result.nccl_ib_gid_index = parse_u32(env, "NCCL_IB_GID_INDEX");
    result.head_gid = parse_u32_default(env, "HEAD_GID", result.nccl_ib_gid_index);
    result.worker_gid = parse_u32_default(env, "WORKER_GID", result.nccl_ib_gid_index);

    result.model = require(env, "MODEL");
    result.model_revision = optional_string(env, "MODEL_REVISION");
    result.served_model_name = require(env, "SERVED_MODEL_NAME");
    result.quantization = require(env, "QUANTIZATION");
    result.kv_cache_dtype = require(env, "KV_CACHE_DTYPE");

    result.speculative_method = require(env, "SPEC_METHOD");
    result.dflash_model = optional_string(env, "DFLASH_MODEL");
    result.dflash_revision = optional_string(env, "DFLASH_REVISION");
    result.dflash_tokens = parse_u32_default(env, "DFLASH_TOKENS", 0);
    result.dflash_draft_tp = parse_u32_default(env, "DFLASH_DRAFT_TP", 0);

    result.tensor_parallel = parse_u32(env, "TP");
    result.nodes = parse_u32(env, "NNODES");
    result.master_port = parse_u32(env, "MASTER_PORT");
    result.api_port = parse_u32(env, "PORT");
    result.max_model_len = parse_u32(env, "MAX_MODEL_LEN");
    result.max_num_seqs = parse_u32(env, "MAX_NUM_SEQS");
    result.max_num_batched_tokens = parse_u32(env, "MAX_NUM_BATCHED_TOKENS");
    result.gpu_memory_utilization = parse_double(env, "GPU_MEM_UTIL");

    result.enforce_eager = parse_bool01_default(env, "ENFORCE_EAGER", false);
    result.use_host_nccl = parse_bool01_default(env, "USE_HOST_NCCL", false);

    result.indexer_workspace = optional_string(env, "GLM53_INDEXER_WORKSPACE");
    result.adaptive_k = optional_string(env, "GLM53_ADAPTIVE_K");
    result.adaptive_k_set = optional_string(env, "GLM53_ADAPTIVE_K_SET");
    return result;
}

DeploymentContract load_deployment_contract(const std::filesystem::path& path) {
    return deployment_from_env(parse_env_file(path));
}

ValidationReport validate_deployment_contract(const DeploymentContract& d, const ModelSpec& model) {
    ValidationReport report;

    if (d.tensor_parallel != 2U) report.errors.emplace_back("TP must be exactly 2 for the target runtime");
    if (d.nodes != 2U) report.errors.emplace_back("NNODES must be exactly 2 for the target runtime");
    if (d.head_ip.empty() || d.worker_ip.empty() || d.head_ip == d.worker_ip) {
        report.errors.emplace_back("HEAD_IP and WORKER_IP must be non-empty and distinct");
    }
    if (d.head_cx7_if.empty() || d.worker_cx7_if.empty()) {
        report.errors.emplace_back("both CX-7 Ethernet interfaces must be specified");
    }
    if (d.head_cx7_ib.empty() || d.worker_cx7_ib.empty()) {
        report.errors.emplace_back("both CX-7 RDMA devices must be specified");
    }
    if (d.head_gid != d.nccl_ib_gid_index || d.worker_gid != d.nccl_ib_gid_index) {
        report.errors.emplace_back("HEAD_GID and WORKER_GID must match NCCL_IB_GID_INDEX");
    }

    require_equal(report, "QUANTIZATION", d.quantization, "nvfp4");
    require_equal(report, "KV_CACHE_DTYPE", d.kv_cache_dtype, "fp8");
    if (d.model.find("GLM-5.3-Flash") == std::string::npos || d.model.find("NVFP4") == std::string::npos) {
        report.errors.emplace_back("MODEL must identify a GLM-5.3-Flash NVFP4 checkpoint");
    }
    if (d.model.find("EXL3") != std::string::npos) {
        report.errors.emplace_back("MODEL must not be an EXL3 checkpoint; this runtime consumes NVFP4");
    }
    if (d.model_revision.empty()) {
        report.warnings.emplace_back("MODEL_REVISION is empty; reproducible checkpoint loading requires a pin");
    }

    if (d.speculative_method != "dflash") {
        report.warnings.emplace_back("SPEC_METHOD is not dflash; the first optimized runtime targets DFlash2");
    } else {
        if (d.dflash_model.empty()) report.errors.emplace_back("DFLASH_MODEL is required for dflash");
        if (d.dflash_revision.empty()) {
            report.warnings.emplace_back("DFLASH_REVISION is empty; draft/target pairing is not reproducible");
        }
        if (d.dflash_tokens == 0U) report.errors.emplace_back("DFLASH_TOKENS must be positive");
        if (d.dflash_draft_tp != 2U) {
            report.warnings.emplace_back("DFLASH_DRAFT_TP differs from the validated TP=2 baseline");
        }
        if (d.dflash_tokens != 7U) {
            report.warnings.emplace_back("DFLASH_TOKENS differs from the validated k=7 baseline");
        }
    }

    if (d.max_model_len == 0U || d.max_model_len > model.max_position_embeddings) {
        report.errors.emplace_back("MAX_MODEL_LEN must be in 1..model.max_position_embeddings");
    }
    if (d.max_num_seqs == 0U) report.errors.emplace_back("MAX_NUM_SEQS must be positive");
    if (d.max_num_batched_tokens == 0U) report.errors.emplace_back("MAX_NUM_BATCHED_TOKENS must be positive");
    if (!(d.gpu_memory_utilization > 0.0 && d.gpu_memory_utilization <= 1.0)) {
        report.errors.emplace_back("GPU_MEM_UTIL must be in (0,1]");
    }
    if (d.master_port == 0U || d.master_port > 65535U || d.api_port == 0U || d.api_port > 65535U) {
        report.errors.emplace_back("MASTER_PORT and PORT must be valid TCP ports");
    }

    // These are intentional warnings rather than hard requirements. They make
    // drift from the evidence-backed baseline visible without making the core
    // parser unusable for controlled A/B experiments.
    if (d.max_model_len != 1000000U) {
        report.warnings.emplace_back("MAX_MODEL_LEN differs from the validated 1,000,000-token baseline");
    }
    if (d.max_num_seqs != 1U) {
        report.warnings.emplace_back("MAX_NUM_SEQS differs from the validated single-inflight baseline");
    }
    if (d.max_num_batched_tokens != 7168U) {
        report.warnings.emplace_back("MAX_NUM_BATCHED_TOKENS differs from the current 7168-token baseline");
    }
    if (d.gpu_memory_utilization > 0.85) {
        report.warnings.emplace_back("GPU_MEM_UTIL exceeds the current validated 0.85 baseline");
    }
    if (d.enforce_eager) {
        report.warnings.emplace_back("ENFORCE_EAGER=1 disables the intended CUDA-graph decode path");
    }
    if (d.use_host_nccl) {
        report.warnings.emplace_back("USE_HOST_NCCL=1 departs from the validated image-NCCL baseline");
    }
    if (d.indexer_workspace != "rightsize") {
        report.warnings.emplace_back("GLM53_INDEXER_WORKSPACE differs from the current rightsize baseline");
    }

    return report;
}

}  // namespace ninfer::glm53
