#include "ninfer_glm53/execution_plan.hpp"

#include <algorithm>
#include <stdexcept>

namespace ninfer::glm53 {
namespace {

// Validate only execution inputs here; host/network admission remains the
// DeploymentContract/preflight responsibility. No division precedes these checks.
ValidationReport validate_inputs(const ModelSpec& model, const DeploymentContract& deployment) {
    ValidationReport report;
    report.errors = validate_model_spec(model);
    if (deployment.tensor_parallel != 2U || deployment.nodes != 2U)
        report.errors.emplace_back("execution requires exactly two ranks on two nodes");
    if (deployment.max_model_len == 0U || deployment.max_model_len > model.max_position_embeddings ||
        deployment.max_num_seqs == 0U || deployment.max_num_batched_tokens == 0U)
        report.errors.emplace_back("execution scheduler limits are invalid");
    if (deployment.speculative_method != "none" && deployment.speculative_method != "dflash")
        report.errors.emplace_back("speculative method must be none or dflash");
    if (deployment.speculative_method == "dflash") {
        if (deployment.dflash_tokens == 0U || deployment.dflash_tokens > 8U ||
            deployment.dflash_draft_tp != 2U)
            report.errors.emplace_back("DFlash requires 1..8 proposals and draft TP=2");
        const auto& revision = deployment.dflash_revision;
        const bool pinned = revision.size() == 40U && std::all_of(revision.begin(), revision.end(), [](char ch) {
            return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f') || (ch >= 'A' && ch <= 'F');
        });
        if (deployment.dflash_model.empty() || !pinned)
            report.errors.emplace_back("DFlash requires a model identity and 40-hex revision");
    }
    return report;
}

}  // namespace

ExecutionPlan build_tp2_execution_plan(const ModelSpec& model, const DeploymentContract& deployment) {
    const auto inputs = validate_inputs(model, deployment);
    if (!inputs.ok()) throw std::invalid_argument(inputs.errors.front());

    const auto tp = deployment.tensor_parallel;
    const auto divisible = [tp](std::uint32_t value) { return value % tp == 0U; };
    if (!divisible(model.attention_heads) || !divisible(model.kv_heads) ||
        !divisible(model.index_heads) || !divisible(model.linear_attention_heads)) {
        throw std::invalid_argument("model head geometry is not divisible by tensor parallel size");
    }

    ExecutionPlan plan;
    plan.rank_geometry = RankGeometry{
        .world_size = tp,
        .hyper_connection_streams = model.hyper_connection_streams,
        .local_attention_heads = model.attention_heads / tp,
        .local_kv_heads = model.kv_heads / tp,
        .local_index_heads = model.index_heads / tp,
        .local_linear_attention_heads = model.linear_attention_heads / tp,
    };
    plan.layers.reserve(model.layers.size());
    for (const auto& layer : model.layers) {
        plan.layers.push_back(LayerExecutionPlan{
            .layer = layer.index,
            .mixer = layer.mixer,
            .ffn = layer.ffn,
            .persistent_state = layer.mixer == MixerKind::kKda
                                    ? PersistentStateKind::kKdaRecurrent
                                    : PersistentStateKind::kSparseMlaKvAndIndex,
            // These mark semantic TP2 synchronization boundaries for branch
            // outputs before the corresponding mHC stream update. Kernel
            // fusion may absorb a boundary, but not change its result.
            .mixer_output_collective = CollectiveKind::kAllReduce,
            .ffn_output_collective = CollectiveKind::kAllReduce,
        });
    }

    if (deployment.speculative_method == "dflash") {
        plan.speculative = SpeculativePlan{
            .enabled = true,
            .proposal_tokens = deployment.dflash_tokens,
            .draft_tensor_parallel = deployment.dflash_draft_tp,
            .draft_model = deployment.dflash_model,
            .draft_revision = deployment.dflash_revision,
        };
    }  // Disabled plans are canonical: no stale draft identity or geometry.
    plan.max_context_tokens = deployment.max_model_len;
    plan.max_inflight_sequences = deployment.max_num_seqs;
    plan.max_batched_tokens = deployment.max_num_batched_tokens;
    return plan;
}

ValidationReport validate_execution_plan(
    const ExecutionPlan& plan, const ModelSpec& model, const DeploymentContract& deployment) {
    auto report = validate_inputs(model, deployment);

    if (plan.rank_geometry.world_size != 2U ||
        plan.rank_geometry.world_size != deployment.tensor_parallel) {
        report.errors.emplace_back("execution plan world_size must be exactly 2");
    }
    if (plan.rank_geometry.hyper_connection_streams != 4U ||
        plan.rank_geometry.hyper_connection_streams != model.hyper_connection_streams) {
        report.errors.emplace_back("execution plan must preserve four mHC streams");
    }
    if (plan.layers.size() != model.layers.size()) {
        report.errors.emplace_back("execution plan layer count does not match model contract");
        return report;
    }
    const auto world = plan.rank_geometry.world_size;
    const auto head_matches = [world](std::uint32_t local, std::uint32_t global) {
        return world == 2U && global > 0U && global % world == 0U && local == global / world;
    };
    if (!head_matches(plan.rank_geometry.local_attention_heads, model.attention_heads) ||
        !head_matches(plan.rank_geometry.local_kv_heads, model.kv_heads) ||
        !head_matches(plan.rank_geometry.local_index_heads, model.index_heads) ||
        !head_matches(plan.rank_geometry.local_linear_attention_heads, model.linear_attention_heads)) {
        report.errors.emplace_back("per-rank head geometry does not reconstruct the logical model geometry");
    }

    for (std::size_t i = 0; i < plan.layers.size(); ++i) {
        const auto& actual = plan.layers[i];
        const auto& expected = model.layers[i];
        if (actual.layer != expected.index || actual.mixer != expected.mixer || actual.ffn != expected.ffn) {
            report.errors.emplace_back("execution plan layer topology diverges at layer " + std::to_string(i));
        }
        const auto expected_state = expected.mixer == MixerKind::kKda
                                        ? PersistentStateKind::kKdaRecurrent
                                        : PersistentStateKind::kSparseMlaKvAndIndex;
        if (actual.persistent_state != expected_state) {
            report.errors.emplace_back("persistent state kind diverges at layer " + std::to_string(i));
        }
        if (actual.mixer_output_collective != CollectiveKind::kAllReduce ||
            actual.ffn_output_collective != CollectiveKind::kAllReduce) {
            report.errors.emplace_back("TP2 branch outputs require explicit all-reduce boundaries at layer " +
                                       std::to_string(i));
        }
    }

    if (plan.max_context_tokens != deployment.max_model_len ||
        plan.max_inflight_sequences != deployment.max_num_seqs ||
        plan.max_batched_tokens != deployment.max_num_batched_tokens) {
        report.errors.emplace_back("scheduler limits in execution plan differ from deployment contract");
    }

    const bool enabled = deployment.speculative_method == "dflash";
    if (plan.speculative.enabled != enabled)
        report.errors.emplace_back("speculative enabled state differs from deployment");
    if (enabled) {
        if (plan.speculative.proposal_tokens != deployment.dflash_tokens ||
            plan.speculative.draft_tensor_parallel != deployment.dflash_draft_tp ||
            plan.speculative.draft_model != deployment.dflash_model ||
            plan.speculative.draft_revision != deployment.dflash_revision)
            report.errors.emplace_back("DFlash identity or geometry differs from deployment");
    } else if (plan.speculative.proposal_tokens != 0U || plan.speculative.draft_tensor_parallel != 0U ||
               !plan.speculative.draft_model.empty() || !plan.speculative.draft_revision.empty()) {
        report.errors.emplace_back("disabled speculative plan contains stale draft fields");
    }

    return report;
}

std::string_view to_string(CollectiveKind kind) noexcept {
    switch (kind) {
        case CollectiveKind::kNone:
            return "none";
        case CollectiveKind::kAllReduce:
            return "all_reduce";
    }
    return "unknown";
}

std::string_view to_string(PersistentStateKind kind) noexcept {
    switch (kind) {
        case PersistentStateKind::kKdaRecurrent:
            return "kda_recurrent";
        case PersistentStateKind::kSparseMlaKvAndIndex:
            return "sparse_mla_kv_index";
    }
    return "unknown";
}

}  // namespace ninfer::glm53
