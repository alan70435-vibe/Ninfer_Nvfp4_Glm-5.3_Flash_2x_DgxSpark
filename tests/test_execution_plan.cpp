#include "ninfer_glm53/execution_plan.hpp"

#include <cassert>

int main() {
    using namespace ninfer::glm53;
    const auto& model = glm53_flash_spec();

    DeploymentContract deployment;
    deployment.tensor_parallel = 2;
    deployment.nodes = 2;
    deployment.max_model_len = 1000000;
    deployment.max_num_seqs = 1;
    deployment.max_num_batched_tokens = 7168;
    deployment.speculative_method = "dflash";
    deployment.dflash_tokens = 7;
    deployment.dflash_draft_tp = 2;
    deployment.dflash_model = "incoai/GLM-5.3-Flash-DFlash2";
    deployment.dflash_revision = "dc77ff1c99eeb2df044ee3d4f0094eb033fee410";

    const auto plan = build_tp2_execution_plan(model, deployment);
    assert(plan.rank_geometry.world_size == 2U);
    assert(plan.rank_geometry.hyper_connection_streams == 4U);
    assert(plan.rank_geometry.local_attention_heads == 32U);
    assert(plan.rank_geometry.local_kv_heads == 32U);
    assert(plan.rank_geometry.local_index_heads == 16U);
    assert(plan.rank_geometry.local_linear_attention_heads == 32U);
    assert(plan.layers.size() == 45U);
    assert(plan.layers[3].persistent_state == PersistentStateKind::kSparseMlaKvAndIndex);
    assert(plan.layers[4].persistent_state == PersistentStateKind::kKdaRecurrent);
    assert(plan.speculative.enabled);
    assert(plan.speculative.proposal_tokens == 7U);
    assert(validate_execution_plan(plan, model, deployment).errors.empty());
    return 0;
}
