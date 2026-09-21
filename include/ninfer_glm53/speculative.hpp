#pragma once

#include <cstdint>

namespace ninfer::glm53 {

inline constexpr std::uint32_t kDflashProposals = 7;

struct RankRecord {
    std::int32_t tokens[16]{};
    std::uint32_t committed = 0;
};

struct CommitView {
    const std::int32_t* tokens = nullptr;
    std::uint32_t count = 0;
};

// Writes the same accepted tokens onto both ranks. A mismatch commits nothing.
[[nodiscard]] bool commit_same_tokens(const CommitView& rank0_view, const CommitView& rank1_view,
                                      RankRecord* rank0, RankRecord* rank1);

// Temperature-0 DFlash2 check. `target` has k verified positions plus one bonus token.
// A full match of the k draft tokens also commits the bonus. A first-token miss commits
// only target[0].
struct SpecResult {
    std::int32_t tokens[16]{};
    std::uint32_t committed = 0;
    bool ranks_agree = false;
};

[[nodiscard]] SpecResult commit_dflash2(const std::int32_t* draft, std::uint32_t k, const std::int32_t* target,
                                        RankRecord* rank0, RankRecord* rank1);

}  // namespace ninfer::glm53
