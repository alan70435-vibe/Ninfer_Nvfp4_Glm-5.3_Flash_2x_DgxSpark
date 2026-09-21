#include "ninfer_glm53/execution_plan.hpp"

#include <stdexcept>

namespace ninfer::glm53 {

ExecutionPlan build_tp2_execution_plan(const ModelSpec& model, const DeploymentContract& deployment) {
    if (deployment.tensor_parallel == 0U) {
        throw std::invalid_argument("tensor parallel size cannot be zero");
    }

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

    plan.speculative = SpeculativePlan{
        .enabled = deployment.speculative_method == "dflash",
        .proposal_tokens = deployment.dflash_tokens,
        .draft_tensor_parallel = deployment.dflash_draft_tp,
        .draft_model = deployment.dflash_model,
        .draft_revision = deployment.dflash_revision,
    };
    plan.max_context_tokens = deployment.max_model_len;
    plan.max_inflight_sequences = deployment.max_num_seqs;
    plan.max_batched_tokens = deployment.max_num_batched_tokens;
    return plan;
}

ValidationReport validate_execution_plan(
    const ExecutionPlan& plan, const ModelSpec& model, const DeploymentContract& deployment) {
    ValidationReport report;

    if (plan.rank_geometry.world_size != 2U) {
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
    if (plan.rank_geometry.local_attention_heads * plan.rank_geometry.world_size != model.attention_heads ||
        plan.rank_geometry.local_kv_heads * plan.rank_geometry.world_size != model.kv_heads ||
        plan.rank_geometry.local_index_heads * plan.rank_geometry.world_size != model.index_heads ||
        plan.rank_geometry.local_linear_attention_heads * plan.rank_geometry.world_size !=
            model.linear_attention_heads) {
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

    if (deployment.speculative_method == "dflash") {
        if (!plan.speculative.enabled) report.errors.emplace_back("DFlash deployment produced disabled speculative plan");
        if (plan.speculative.proposal_tokens != deployment.dflash_tokens ||
            plan.speculative.draft_tensor_parallel != deployment.dflash_draft_tp) {
            report.errors.emplace_back("DFlash execution geometry differs from deployment contract");
        }
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
