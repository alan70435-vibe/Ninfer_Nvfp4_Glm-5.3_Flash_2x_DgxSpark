#include "ninfer_glm53/checkpoint_binding.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void print_summary(const ninfer::glm53::BindingReport& report) {
    std::cerr << ninfer::glm53::to_string(report.kind) << ": " << (report.complete ? "COMPLETE" : "FAILED") << '\n'
              << "  expected: " << report.expected_tensors << '\n'
              << "  observed: " << report.observed_tensors << '\n'
              << "  bound: " << report.bound << '\n'
              << "  missing: " << report.missing << '\n'
              << "  unexpected: " << report.unexpected << '\n'
              << "  dtype_mismatches: " << report.dtype_mismatches << '\n'
              << "  shape_mismatches: " << report.shape_mismatches << '\n'
              << "  mcg_mismatches: " << report.mcg_mismatches << '\n'
              << "  shapes_checked: " << (report.shapes_checked ? "yes" : "no") << '\n'
              << "  mcg_payload_checked: " << (report.mcg_payload_checked ? "yes" : "no") << '\n'
              << "  config_sha256: " << report.config_sha256 << '\n'
              << "  index_sha256: " << report.index_sha256 << '\n'
              << "  logical_catalog_sha256: " << report.logical_catalog_sha256 << '\n'
              << "  mirror_upstream_revision: " << report.mirror_upstream_revision << '\n'
              << "  source_model_revision: " << report.source_model_revision << '\n'
              << "  snapshot_revision: " << report.snapshot_revision << '\n';
    for (const auto& warning : report.warnings) std::cerr << "  warning: " << warning << '\n';
    for (const auto& error : report.errors) std::cerr << "  error: " << error << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    std::filesystem::path directory;
    std::filesystem::path output;
    bool names_only = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--names-only") {
            names_only = true;
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 >= argc) {
                std::cerr << "missing path after " << arg << '\n';
                return 64;
            }
            output = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cerr << "usage: ninfer-glm53-bind <checkpoint-dir> [-o receipt.json] [--names-only]\n";
            return 0;
        } else if (arg.starts_with('-')) {
            std::cerr << "unknown argument: " << arg << '\n';
            return 64;
        } else if (directory.empty()) {
            directory = arg;
        } else {
            std::cerr << "unexpected argument: " << arg << '\n';
            return 64;
        }
    }
    if (directory.empty()) {
        std::cerr << "usage: ninfer-glm53-bind <checkpoint-dir> [-o receipt.json] [--names-only]\n";
        return 64;
    }

    try {
        const auto report = ninfer::glm53::bind_checkpoint_directory(directory, names_only);
        const auto receipt = ninfer::glm53::binding_receipt_json(report);
        if (output.empty()) {
            std::cout << receipt;
        } else {
            if (output.has_parent_path()) std::filesystem::create_directories(output.parent_path());
            std::ofstream file(output);
            if (!file) {
                std::cerr << "cannot write " << output << '\n';
                return 1;
            }
            file << receipt;
        }
        print_summary(report);
        return report.complete ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "bind failed: " << error.what() << '\n';
        return 1;
    }
}
