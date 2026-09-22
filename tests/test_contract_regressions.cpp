#include "ninfer_glm53/execution_plan.hpp"
#include "test_support.hpp"

#include <array>
#include <stdexcept>

int main() {
    using namespace ninfer::glm53;
    using ninfer::test::throws;
    const auto model = glm53_flash_spec();
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
    CHECK(validate_model_spec(model).empty());
    CHECK(validate_execution_plan(plan, model, deployment).ok());

    constexpr std::array<std::uint32_t ModelSpec::*, 11> fields = {
        &ModelSpec::intermediate_size, &ModelSpec::vocab_size, &ModelSpec::qk_head_dim,
        &ModelSpec::qk_nope_head_dim, &ModelSpec::v_head_dim, &ModelSpec::index_kpool,
        &ModelSpec::linear_attention_heads, &ModelSpec::linear_attention_head_dim,
        &ModelSpec::short_conv_kernel_size, &ModelSpec::shared_experts, &ModelSpec::moe_intermediate_size};
    for (const auto field : fields) {
        for (const auto value : {0U, model.*field + 1U}) {
            auto bad = model;
            bad.*field = value;
            CHECK(!validate_model_spec(bad).empty());
            CHECK(!validate_execution_plan(plan, bad, deployment).ok());
            CHECK(throws<std::invalid_argument>([&] { (void)build_tp2_execution_plan(bad, deployment); }));
        }
    }
    constexpr std::array<std::uint32_t RankGeometry::*, 4> heads = {
        &RankGeometry::local_attention_heads, &RankGeometry::local_kv_heads,
        &RankGeometry::local_index_heads, &RankGeometry::local_linear_attention_heads};
    for (const auto field : heads) {
        for (const auto value : {0U, (plan.rank_geometry.*field) - 1U,
                                0x80000000U + (plan.rank_geometry.*field)}) {
            auto bad = plan;
            bad.rank_geometry.*field = value;
            CHECK(!validate_execution_plan(bad, model, deployment).ok());
        }
    }
    auto bad = plan;
    bad.speculative.draft_model = "wrong/checkpoint";
    CHECK(!validate_execution_plan(bad, model, deployment).ok());
    bad = plan;
    bad.speculative.draft_revision = std::string(40, '0');
    CHECK(!validate_execution_plan(bad, model, deployment).ok());
    bad = plan;
    bad.speculative.enabled = false;
    CHECK(!validate_execution_plan(bad, model, deployment).ok());
    bad = plan;
    bad.speculative.proposal_tokens = 2;
    CHECK(!validate_execution_plan(bad, model, deployment).ok());
    bad = plan;
    bad.speculative.draft_tensor_parallel = 1;
    CHECK(!validate_execution_plan(bad, model, deployment).ok());

    auto disabled = deployment;
    disabled.speculative_method = "none";  // Stale env fields are not copied into a disabled plan.
    CHECK(!validate_execution_plan(plan, model, disabled).ok());
    const auto off = build_tp2_execution_plan(model, disabled);
    CHECK(!off.speculative.enabled && off.speculative.draft_model.empty());
    CHECK(off.speculative.draft_revision.empty() && off.speculative.proposal_tokens == 0U);
    CHECK(off.speculative.draft_tensor_parallel == 0U);
    CHECK(validate_execution_plan(off, model, disabled).ok());
    bad = off;
    bad.speculative.draft_model = "stale";
    CHECK(!validate_execution_plan(bad, model, disabled).ok());

    for (const auto tp : {0U, 1U, 4U}) {
        auto invalid = deployment;
        invalid.tensor_parallel = tp;
        CHECK(throws<std::invalid_argument>([&] { (void)build_tp2_execution_plan(model, invalid); }));
        CHECK(!validate_execution_plan(plan, model, invalid).ok());
        bad = plan;
        bad.rank_geometry.world_size = tp;
        CHECK(!validate_execution_plan(bad, model, deployment).ok());
    }
    auto invalid = deployment;
    invalid.speculative_method = "dfalsh";
    CHECK(throws<std::invalid_argument>([&] { (void)build_tp2_execution_plan(model, invalid); }));
    invalid = deployment;
    invalid.dflash_revision = "main";
    CHECK(throws<std::invalid_argument>([&] { (void)build_tp2_execution_plan(model, invalid); }));
    for (const auto k : {0U, 9U}) {
        invalid = deployment;
        invalid.dflash_tokens = k;
        CHECK(throws<std::invalid_argument>([&] { (void)build_tp2_execution_plan(model, invalid); }));
    }
    for (const auto k : {1U, 2U, 4U, 7U, 8U}) {
        auto experiment = deployment;
        experiment.dflash_tokens = k;
        const auto experimental = build_tp2_execution_plan(model, experiment);
        CHECK(validate_execution_plan(experimental, model, experiment).ok());
    }
    for (auto field : {&DeploymentContract::nodes, &DeploymentContract::max_model_len,
                       &DeploymentContract::max_num_seqs, &DeploymentContract::max_num_batched_tokens}) {
        invalid = deployment;
        invalid.*field = 0;
        CHECK(throws<std::invalid_argument>([&] { (void)build_tp2_execution_plan(model, invalid); }));
    }
    return 0;
}
