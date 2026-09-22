#include "ninfer_glm53/model_spec.hpp"

#include "test_support.hpp"

int main() {
    using namespace ninfer::glm53;
    const auto& spec = glm53_flash_spec();
    CHECK(validate_model_spec(spec).empty());
    CHECK(spec.layers.size() == 45U);
    CHECK(count_mixer(spec, MixerKind::kKda) == 34U);
    CHECK(count_mixer(spec, MixerKind::kSparseMla) == 11U);
    CHECK(count_ffn(spec, FfnKind::kDense) == 3U);
    CHECK(count_ffn(spec, FfnKind::kSparseMoe) == 42U);
    CHECK(spec.manifold_hyper_connections);
    CHECK(spec.hyper_connection_streams == 4U);
    CHECK(spec.hyper_connection_sinkhorn_iterations == 20U);
    CHECK(spec.swiglu_limit == 10.0);
    CHECK(spec.mhc_parameters_fp32);
    CHECK(spec.kda_decay_parameters_fp32);

    CHECK(spec.layers[0].mixer == MixerKind::kKda);
    CHECK(spec.layers[2].ffn == FfnKind::kDense);
    CHECK(spec.layers[3].mixer == MixerKind::kSparseMla);
    CHECK(spec.layers[3].ffn == FfnKind::kSparseMoe);
    CHECK(spec.layers[43].mixer == MixerKind::kSparseMla);
    CHECK(spec.layers[44].mixer == MixerKind::kKda);
    return 0;
}
