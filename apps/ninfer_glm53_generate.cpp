#include <filesystem>
#include <fstream>
#include <iostream>
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
    std::string status = "FAILED";
    std::string reason = "full_forward_not_implemented";
    if (!config || !shards) reason = "weights_absent";
    if (mode == "tp2" && reason == "full_forward_not_implemented") reason = "tp2_runtime_not_linked";
    if (mode == "dflash" && reason == "full_forward_not_implemented") reason = "dflash_runtime_not_linked";

    write_line(std::cout, "prompt_id", prompt_id);
    write_line(std::cout, "checkpoint_revision", revision);
    write_line(std::cout, "quant", quant);
    write_line(std::cout, "mode", mode);
    write_line(std::cout, "status", status);
    write_line(std::cout, "reason", reason);
    write_line(std::cout, "token_ids", "");
    write_line(std::cout, "committed_length", "0");
    write_line(std::cout, "gpu", "NVIDIA GB10");
    write_line(std::cout, "world_size", mode == "tp2" ? "2" : "1");
    return shards ? 3 : 2;
}
