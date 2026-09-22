#include "ninfer_glm53/speculative.hpp"

#include <cstring>

namespace ninfer::glm53 {

bool commit_same_tokens(const CommitView& rank0_view, const CommitView& rank1_view, RankRecord* rank0,
                        RankRecord* rank1) {
    if (rank0 == nullptr || rank1 == nullptr || rank0_view.tokens == nullptr || rank1_view.tokens == nullptr) {
        return false;
    }
    if (rank0_view.count != rank1_view.count || rank0_view.count > 16u) return false;
    if (std::memcmp(rank0_view.tokens, rank1_view.tokens, rank0_view.count * sizeof(std::int32_t)) != 0) {
        return false;
    }
    rank0->committed = rank0_view.count;
    rank1->committed = rank1_view.count;
    std::memcpy(rank0->tokens, rank0_view.tokens, rank0_view.count * sizeof(std::int32_t));
    std::memcpy(rank1->tokens, rank1_view.tokens, rank1_view.count * sizeof(std::int32_t));
    return true;
}

SpecResult commit_dflash2(const std::int32_t* draft, std::uint32_t k, const std::int32_t* target, RankRecord* rank0,
                          RankRecord* rank1) {
    SpecResult result;
    if (draft == nullptr || target == nullptr || k == 0u || k > 8u) return result;
    std::uint32_t matched = 0;
    while (matched < k && draft[matched] == target[matched]) ++matched;
    std::int32_t accepted[16];
    std::uint32_t count = 0;
    if (matched == 0u) {
        accepted[count++] = target[0];
    } else {
        for (std::uint32_t i = 0; i < matched; ++i) accepted[count++] = draft[i];
        accepted[count++] = target[matched];
    }
    const CommitView view{accepted, count};
    result.ranks_agree = commit_same_tokens(view, view, rank0, rank1);
    if (result.ranks_agree) {
        result.committed = count;
        std::memcpy(result.tokens, accepted, count * sizeof(std::int32_t));
    }
    return result;
}

}  // namespace ninfer::glm53
