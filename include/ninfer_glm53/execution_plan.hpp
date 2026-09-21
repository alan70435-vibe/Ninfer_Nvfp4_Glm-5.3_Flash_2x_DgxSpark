#pragma once

#include "ninfer_glm53/deployment_contract.hpp"
#include "ninfer_glm53/model_spec.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::glm53 {

enum class CollectiveKind : std::uint8_t {
    kNone,
    kAllReduce,
};

enum class PersistentStateKind : std::uint8_t {
    kKdaRecurrent,
    kSparseMlaKvAndIndex,
};

struct RankGeometry {
    std::uint32_t world_size{};
    std::uint32_t hyper_connection_streams{};
    std::uint32_t local_attention_heads{};
    std::uint32_t local_kv_heads{};
    std::uint32_t local_index_heads{};
    std::uint32_t local_linear_attention_heads{};
};

struct LayerExecutionPlan {
    std::uint32_t layer{};
    MixerKind mixer{MixerKind::kKda};
    FfnKind ffn{FfnKind::kSparseMoe};
    PersistentStateKind persistent_state{PersistentStateKind::kKdaRecurrent};
    CollectiveKind mixer_output_collective{CollectiveKind::kAllReduce};
    CollectiveKind ffn_output_collective{CollectiveKind::kAllReduce};
};

struct SpeculativePlan {
    bool enabled{};
    std::uint32_t proposal_tokens{};
    std::uint32_t draft_tensor_parallel{};
    std::string draft_model;
    std::string draft_revision;
};

struct ExecutionPlan {
    RankGeometry rank_geometry;
    std::vector<LayerExecutionPlan> layers;
    SpeculativePlan speculative;
    std::uint32_t max_context_tokens{};
    std::uint32_t max_inflight_sequences{};
    std::uint32_t max_batched_tokens{};
};

[[nodiscard]] ExecutionPlan build_tp2_execution_plan(
    const ModelSpec& model, const DeploymentContract& deployment);
[[nodiscard]] ValidationReport validate_execution_plan(
    const ExecutionPlan& plan, const ModelSpec& model, const DeploymentContract& deployment);
[[nodiscard]] std::string_view to_string(CollectiveKind kind) noexcept;
[[nodiscard]] std::string_view to_string(PersistentStateKind kind) noexcept;

}  // namespace ninfer::glm53
