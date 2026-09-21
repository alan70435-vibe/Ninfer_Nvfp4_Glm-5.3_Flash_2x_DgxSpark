#include "ninfer_glm53/deployment_contract.hpp"
#include "ninfer_glm53/execution_plan.hpp"
#include "ninfer_glm53/model_spec.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace {

void print_report(std::string_view title, const ninfer::glm53::ValidationReport& report) {
    std::cout << title << ": " << (report.ok() ? "OK" : "FAILED") << '\n';
    for (const auto& warning : report.warnings) std::cout << "  warning: " << warning << '\n';
    for (const auto& error : report.errors) std::cout << "  error: " << error << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: ninfer-glm53-plan <deployment.env>\n";
        return 64;
    }

    try {
        const auto& model = ninfer::glm53::glm53_flash_spec();
        const auto model_errors = ninfer::glm53::validate_model_spec(model);
        if (!model_errors.empty()) {
            std::cerr << "model contract FAILED\n";
            for (const auto& error : model_errors) std::cerr << "  error: " << error << '\n';
            return 2;
        }

        const auto deployment = ninfer::glm53::load_deployment_contract(std::filesystem::path(argv[1]));
        const auto deployment_report = ninfer::glm53::validate_deployment_contract(deployment, model);
        print_report("deployment contract", deployment_report);
        if (!deployment_report.ok()) return 2;

        const auto plan = ninfer::glm53::build_tp2_execution_plan(model, deployment);
        const auto plan_report = ninfer::glm53::validate_execution_plan(plan, model, deployment);
        print_report("execution plan", plan_report);
        if (!plan_report.ok()) return 2;

        std::cout << "\nmodel\n"
                  << "  architecture: " << model.architecture << '\n'
                  << "  layers: " << model.layers.size() << '\n'
                  << "  kda_layers: " << ninfer::glm53::count_mixer(model, ninfer::glm53::MixerKind::kKda) << '\n'
                  << "  sparse_mla_layers: "
                  << ninfer::glm53::count_mixer(model, ninfer::glm53::MixerKind::kSparseMla) << '\n'
                  << "  dense_ffn_layers: "
                  << ninfer::glm53::count_ffn(model, ninfer::glm53::FfnKind::kDense) << '\n'
                  << "  sparse_moe_layers: "
                  << ninfer::glm53::count_ffn(model, ninfer::glm53::FfnKind::kSparseMoe) << '\n'
                  << "  mhc_streams: " << model.hyper_connection_streams << '\n'
                  << "  mhc_sinkhorn_iterations: " << model.hyper_connection_sinkhorn_iterations << '\n'
                  << "  max_native_context: " << model.max_position_embeddings << '\n';

        std::cout << "\ndeployment\n"
                  << "  ranks: " << plan.rank_geometry.world_size << '\n'
                  << "  head: " << deployment.head_ip << " / " << deployment.head_cx7_if << " / "
                  << deployment.head_cx7_ib << '\n'
                  << "  worker: " << deployment.worker_ip << " / " << deployment.worker_cx7_if << " / "
                  << deployment.worker_cx7_ib << '\n'
                  << "  mhc_streams_per_rank: " << plan.rank_geometry.hyper_connection_streams << '\n'
                  << "  local_attention_heads: " << plan.rank_geometry.local_attention_heads << '\n' 
                  << "  local_index_heads: " << plan.rank_geometry.local_index_heads << '\n'
                  << "  context: " << plan.max_context_tokens << '\n'
                  << "  max_inflight: " << plan.max_inflight_sequences << '\n'
                  << "  max_batched_tokens: " << plan.max_batched_tokens << '\n'
                  << "  speculative: " << (plan.speculative.enabled ? "dflash" : "disabled") << '\n'
                  << "  proposal_tokens: " << plan.speculative.proposal_tokens << '\n'
                  << "  draft_tp: " << plan.speculative.draft_tensor_parallel << '\n';

        std::cout << "\nlayer plan\n";
        for (const auto& layer : plan.layers) {
            std::cout << "  L" << layer.layer << " mixer=" << ninfer::glm53::to_string(layer.mixer)
                      << " state=" << ninfer::glm53::to_string(layer.persistent_state)
                      << " ffn=" << ninfer::glm53::to_string(layer.ffn)
                      << " mixer_sync=" << ninfer::glm53::to_string(layer.mixer_output_collective)
                      << " ffn_sync=" << ninfer::glm53::to_string(layer.ffn_output_collective) << '\n';
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "fatal: " << ex.what() << '\n';
        return 2;
    }
}
