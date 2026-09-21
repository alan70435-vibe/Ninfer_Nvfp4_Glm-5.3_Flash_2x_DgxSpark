#include "ninfer_glm53/deployment_contract.hpp"

#include <cassert>
#include <filesystem>
#include <fstream>

namespace {

const char* kBaseline = R"ENV(
HEAD_IP=192.168.100.10
WORKER_IP=192.168.100.11
HEAD_CX7_IF=enp1s0f0np0
WORKER_CX7_IF=enp1s0f1np1
HEAD_CX7_IB=rocep1s0f0
WORKER_CX7_IB=rocep1s0f1
NCCL_IB_GID_INDEX=3
HEAD_GID=3
WORKER_GID=3
MODEL=RedHatAI/GLM-5.3-Flash-NVFP4
MODEL_REVISION=18d55bfd5a2194887738da73753975c9d3842f46
PORT=8888
SERVED_MODEL_NAME=GLM-5.3-Flash-NVFP4
TP=2
NNODES=2
MASTER_PORT=29521
QUANTIZATION=nvfp4
ENFORCE_EAGER=0
SPEC_METHOD=dflash
DFLASH_MODEL=incoai/GLM-5.3-Flash-DFlash2
DFLASH_REVISION=dc77ff1c99eeb2df044ee3d4f0094eb033fee410
DFLASH_TOKENS=7
DFLASH_DRAFT_TP=2
MAX_MODEL_LEN=1000000
MAX_NUM_SEQS=1
MAX_NUM_BATCHED_TOKENS=7168
GPU_MEM_UTIL=0.85
KV_CACHE_DTYPE=fp8
GLM53_INDEXER_WORKSPACE=rightsize
GLM53_ADAPTIVE_K=ema
GLM53_ADAPTIVE_K_SET=2,4,7
USE_HOST_NCCL=0
)ENV";

}  // namespace

int main() {
    using namespace ninfer::glm53;
    const auto path = std::filesystem::temp_directory_path() / "ninfer-glm53-test.env";
    {
        std::ofstream out(path);
        out << kBaseline;
    }

    const auto env = parse_env_file(path);
    assert(env.at("TP") == "2");
    const auto deployment = deployment_from_env(env);
    const auto report = validate_deployment_contract(deployment, glm53_flash_spec());
    assert(report.errors.empty());
    assert(report.warnings.empty());

    auto invalid = deployment;
    invalid.tensor_parallel = 1;
    const auto invalid_report = validate_deployment_contract(invalid, glm53_flash_spec());
    assert(!invalid_report.errors.empty());

    std::filesystem::remove(path);
    return 0;
}
