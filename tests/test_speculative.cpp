#include "ninfer_glm53/speculative.hpp"

#include <cstdlib>
#include <iostream>

namespace {

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

}  // namespace

int main() {
    using namespace ninfer::glm53;
    const std::int32_t same_a[] = {3, 4, 5};
    const std::int32_t same_b[] = {3, 4, 5};
    const std::int32_t other[] = {3, 9, 5};
    RankRecord rank0;
    RankRecord rank1;
    expect(commit_same_tokens(CommitView{same_a, 3}, CommitView{same_b, 3}, &rank0, &rank1), "matching commit");
    expect(rank0.committed == 3 && rank1.committed == 3, "both ranks record length 3");
    expect(rank0.tokens[2] == 5 && rank1.tokens[2] == 5, "both ranks store the token");

    rank0 = {};
    rank1 = {};
    expect(!commit_same_tokens(CommitView{same_a, 3}, CommitView{other, 3}, &rank0, &rank1), "mismatch rejected");
    expect(rank0.committed == 0 && rank1.committed == 0, "mismatch commits nothing");

    const std::int32_t draft[7] = {10, 11, 12, 13, 14, 15, 16};
    const std::int32_t partial[8] = {10, 11, 12, 99, 14, 15, 16, 77};
    auto partial_result = commit_dflash2(draft, kDflashProposals, partial, &rank0, &rank1);
    expect(partial_result.ranks_agree && partial_result.committed == 4, "partial accept keeps the correction");
    expect(rank0.committed == 4 && rank1.committed == 4, "partial length on both ranks");
    expect(partial_result.tokens[0] == 10 && partial_result.tokens[2] == 12 && partial_result.tokens[3] == 99,
           "partial tokens");

    const std::int32_t full_target[8] = {10, 11, 12, 13, 14, 15, 16, 42};
    auto full = commit_dflash2(draft, kDflashProposals, full_target, &rank0, &rank1);
    expect(full.committed == 8 && full.tokens[7] == 42, "full accept keeps the bonus token");
    expect(rank0.committed == 8 && rank1.committed == 8, "full length on both ranks");

    const std::int32_t reject_target[8] = {7, 11, 12, 13, 14, 15, 16, 42};
    auto rejected = commit_dflash2(draft, kDflashProposals, reject_target, &rank0, &rank1);
    expect(rejected.committed == 1 && rejected.tokens[0] == 7, "reject keeps the target token");
    expect(rank0.committed == 1 && rank1.committed == 1 && rank0.tokens[0] == 7 && rank1.tokens[0] == 7,
           "reject recorded on both ranks");
    return 0;
}
