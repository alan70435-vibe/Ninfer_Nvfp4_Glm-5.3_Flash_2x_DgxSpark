#pragma once

#include "ninfer_glm53/speculative.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::glm53 {

// Stop ids shared by greedy and DFlash. 154822 is not an end-of-sequence id.
[[nodiscard]] inline bool is_generation_eos(std::int32_t token) {
    return token == 154820 || token == 154827 || token == 154829;
}

struct DflashOutput {
    std::vector<std::int32_t> tokens;
    std::string finish;
    int accepted_drafts = 0;
    int verify_rounds = 0;
    std::vector<std::int32_t> last_proposals;
    std::vector<std::int32_t> materialized_generated;
};

// One temperature-0 continuation. Verify steps are restored before the accepted
// round is stepped, so a rejected proposal does not stay in the cache. The last
// emitted token is not stepped.
template <typename Model, typename Propose>
[[nodiscard]] DflashOutput generate_dflash_continuation(Model& model, Propose&& propose,
                                                       const std::vector<std::int32_t>& prompt, int new_tokens) {
    DflashOutput out;
    model.set_capture(true);
    for (const std::int32_t token : prompt) model.step(token);
    model.set_capture(false);

    while (static_cast<int>(out.tokens.size()) < new_tokens) {
        const std::int32_t anchor = model.argmax();
        if (is_generation_eos(anchor)) {
            out.tokens.push_back(anchor);
            out.finish = "eos";
            break;
        }
        if (static_cast<int>(out.tokens.size()) + 1 == new_tokens) {
            out.tokens.push_back(anchor);
            out.finish = "budget";
            break;
        }

        out.last_proposals = propose(anchor);
        if (out.last_proposals.size() != static_cast<std::size_t>(kDflashProposals)) {
            throw std::runtime_error("DFlash proposal count");
        }
        const auto snapshot = model.save();
        SpecResult commit;
        try {
            std::vector<std::int32_t> posterior;
            posterior.reserve(static_cast<std::size_t>(kDflashProposals) + 1U);
            model.step(anchor);
            posterior.push_back(model.argmax());
            for (const std::int32_t proposal : out.last_proposals) {
                model.step(proposal);
                posterior.push_back(model.argmax());
            }
            RankRecord rank0;
            RankRecord rank1;
            commit = commit_dflash2(out.last_proposals.data(), kDflashProposals, posterior.data(), &rank0, &rank1);
            if (!commit.ranks_agree) throw std::runtime_error("dflash ranks disagree");
        } catch (...) {
            model.restore(snapshot);
            throw;
        }
        model.restore(snapshot);

        std::uint32_t matched = 0;
        while (matched < commit.committed && matched < kDflashProposals &&
               commit.tokens[matched] == out.last_proposals[static_cast<std::size_t>(matched)]) {
            ++matched;
        }

        std::vector<std::int32_t> round;
        round.reserve(static_cast<std::size_t>(commit.committed) + 1U);
        round.push_back(anchor);
        for (std::uint32_t index = 0; index < commit.committed; ++index) round.push_back(commit.tokens[index]);
        const int remaining = new_tokens - static_cast<int>(out.tokens.size());
        if (remaining >= 0 && static_cast<int>(round.size()) > remaining) {
            round.resize(static_cast<std::size_t>(remaining));
        }
        for (std::size_t index = 0; index < round.size(); ++index) {
            if (!is_generation_eos(round[index])) continue;
            round.resize(index + 1U);
            break;
        }

        const bool ends_eos = !round.empty() && is_generation_eos(round.back());
        const bool fills = static_cast<int>(out.tokens.size() + round.size()) >= new_tokens;
        const bool terminal = ends_eos || fills;
        out.tokens.insert(out.tokens.end(), round.begin(), round.end());
        out.accepted_drafts += static_cast<int>(matched);
        ++out.verify_rounds;

        std::size_t materialize = round.size();
        if (terminal && materialize > 0U) --materialize;
        for (std::size_t index = 0; index < materialize; ++index) {
            model.step(round[index]);
            out.materialized_generated.push_back(round[index]);
        }
        if (terminal) {
            out.finish = ends_eos ? "eos" : "budget";
            break;
        }
    }
    return out;
}

}  // namespace ninfer::glm53
