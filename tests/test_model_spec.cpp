#include "ninfer_glm53/model_spec.hpp"

#include <cassert>

int main() {
    using namespace ninfer::glm53;
    const auto& spec = glm53_flash_spec();
    assert(validate_model_spec(spec).empty());
    assert(spec.layers.size() == 45U);
    assert(count_mixer(spec, MixerKind::kKda) == 34U);
    assert(count_mixer(spec, MixerKind::kSparseMla) == 11U);
    assert(count_ffn(spec, FfnKind::kDense) == 3U);
    assert(count_ffn(spec, FfnKind::kSparseMoe) == 42U);
    assert(spec.manifold_hyper_connections);
    assert(spec.hyper_connection_streams == 4U);
    assert(spec.hyper_connection_sinkhorn_iterations == 20U);
    assert(spec.swiglu_limit == 10.0);
    assert(spec.mhc_parameters_fp32);
    assert(spec.kda_decay_parameters_fp32);

    assert(spec.layers[0].mixer == MixerKind::kKda);
    assert(spec.layers[2].ffn == FfnKind::kDense);
    assert(spec.layers[3].mixer == MixerKind::kSparseMla);
    assert(spec.layers[3].ffn == FfnKind::kSparseMoe);
    assert(spec.layers[43].mixer == MixerKind::kSparseMla);
    assert(spec.layers[44].mixer == MixerKind::kKda);
    return 0;
}
