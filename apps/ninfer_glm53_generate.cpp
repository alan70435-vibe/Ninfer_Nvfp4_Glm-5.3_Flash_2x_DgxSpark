#include "ninfer_glm53/text_forward.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void write_line(std::ostream& out, std::string_view key, std::string_view value) { out << key << '=' << value << '\n'; }

}  // namespace

int main(int argc, char** argv) {
    std::string checkpoint;
    std::string revision;
    std::string prompt_id = "fixed-text-v1";
    std::string mode = "greedy";
    std::string quant = "nvfp4";
    std::string draft_checkpoint;
    int new_tokens = 2;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need = [&](std::string& dest) {
            if (i + 1 >= argc) return false;
            dest = argv[++i];
            return true;
        };
        if (arg == "--checkpoint") {
            if (!need(checkpoint)) return 64;
        } else if (arg == "--revision") {
            if (!need(revision)) return 64;
        } else if (arg == "--prompt-id") {
            if (!need(prompt_id)) return 64;
        } else if (arg == "--mode") {
            if (!need(mode)) return 64;
        } else if (arg == "--quant") {
            if (!need(quant)) return 64;
        } else if (arg == "--draft-checkpoint") {
            if (!need(draft_checkpoint)) return 64;
        } else if (arg == "--new-tokens") {
            std::string text;
            if (!need(text)) return 64;
            new_tokens = std::stoi(text);
        } else {
            std::cerr << "unknown argument " << arg << '\n';
            return 64;
        }
    }

    const auto root = std::filesystem::path(checkpoint);
    const bool config = !checkpoint.empty() && std::filesystem::is_regular_file(root / "config.json");
    bool shards = false;
    if (config && std::filesystem::is_directory(root)) {
        for (const auto& entry : std::filesystem::directory_iterator(root)) {
            if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
                shards = true;
                break;
            }
        }
    }
    if (!config || !shards) {
        write_line(std::cout, "prompt_id", prompt_id);
        write_line(std::cout, "checkpoint_revision", revision);
        write_line(std::cout, "quant", quant);
        write_line(std::cout, "mode", mode);
        write_line(std::cout, "status", "FAILED");
        write_line(std::cout, "reason", "weights_absent");
        write_line(std::cout, "token_ids", "");
        write_line(std::cout, "committed_length", "0");
        write_line(std::cout, "gpu", "NVIDIA GB10");
        write_line(std::cout, "world_size", mode == "tp2" ? "2" : "1");
        return 2;
    }

    ninfer::glm53::GenerateResult result;
    try {
        result = ninfer::glm53::generate_text(root, prompt_id, mode, new_tokens, draft_checkpoint);
    } catch (const std::exception& error) {
        result.ok = false;
        result.reason = error.what();
        result.world_size = mode == "tp2" ? 2 : 1;
    }
    auto join = [](const std::vector<std::int32_t>& values) {
        std::string text;
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (i != 0U) text.push_back(',');
            text += std::to_string(values[i]);
        }
        return text;
    };
    const std::string ids = join(result.token_ids);
    const std::string draft_ids = join(result.draft_tokens);
    write_line(std::cout, "prompt_id", prompt_id);
    write_line(std::cout, "checkpoint_revision", revision);
    write_line(std::cout, "quant", quant);
    write_line(std::cout, "mode", mode);
    write_line(std::cout, "status", result.ok ? "OK" : "FAILED");
    write_line(std::cout, "reason", result.reason);
    write_line(std::cout, "prompt_text", result.prompt_text);
    write_line(std::cout, "prompt_token_ids",
               result.prompt_tokens.empty() ? "" : std::to_string(result.prompt_tokens.front()));
    write_line(std::cout, "token_ids", ids);
    write_line(std::cout, "committed_length", std::to_string(result.committed));
    write_line(std::cout, "rank0_committed_length", std::to_string(result.rank0_committed));
    write_line(std::cout, "rank1_committed_length", std::to_string(result.rank1_committed));
    write_line(std::cout, "rank0_token_ids", join(result.rank0_tokens));
    write_line(std::cout, "rank1_token_ids", join(result.rank1_tokens));
    write_line(std::cout, "dflash_accept", result.dflash_accept);
    write_line(std::cout, "dflash_accept_count", std::to_string(result.accept_count));
    write_line(std::cout, "draft_token_ids", draft_ids);
    write_line(std::cout, "logit_margin", std::to_string(result.logit_margin));
    write_line(std::cout, "tolerance", "exact_token_ids");
    write_line(std::cout, "reference", "glm5_next_eager_fp32_modelopt_nvfp4");
    write_line(std::cout, "gpu", "NVIDIA GB10");
    write_line(std::cout, "world_size", std::to_string(result.world_size == 0 ? 1 : result.world_size));
    write_line(std::cout, "second_spark", "not_used");
    return result.ok ? 0 : 4;
}
