#include "ninfer_glm53/dflash_loop.hpp"
#include "ninfer_glm53/speculative.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

struct StepDouble {
    std::vector<std::int32_t> stepped;
    int eos_at = -1;
    std::int32_t eos_token = 154820;
    bool capture = false;

    void set_capture(bool enabled) { capture = enabled; }
    void step(std::int32_t token) { stepped.push_back(token); }
    [[nodiscard]] std::int32_t argmax() const {
        const int seen = static_cast<int>(stepped.size());
        if (seen == eos_at) return eos_token;
        return static_cast<std::int32_t>(1000 + seen);
    }
    [[nodiscard]] std::vector<std::int32_t> save() const { return stepped; }
    void restore(const std::vector<std::int32_t>& snapshot) { stepped = snapshot; }
};

[[nodiscard]] bool oracle_eos(std::int32_t token) {
    return token == 154820 || token == 154827 || token == 154829;
}

struct OracleRun {
    std::vector<std::int32_t> tokens;
    std::vector<std::int32_t> stepped;
    std::string finish;
};

[[nodiscard]] OracleRun greedy_oracle(const std::vector<std::int32_t>& prompt, int budget, int eos_at,
                                      std::int32_t eos_token = 154820) {
    StepDouble model;
    model.eos_at = eos_at;
    model.eos_token = eos_token;
    for (const std::int32_t token : prompt) model.step(token);
    OracleRun out;
    while (static_cast<int>(out.tokens.size()) < budget) {
        const std::int32_t token = model.argmax();
        out.tokens.push_back(token);
        if (oracle_eos(token) || static_cast<int>(out.tokens.size()) == budget) {
            out.finish = oracle_eos(token) ? "eos" : "budget";
            break;
        }
        model.step(token);
    }
    out.stepped = model.stepped;
    return out;
}

struct DflashRun {
    ninfer::glm53::DflashOutput output;
    std::vector<std::int32_t> stepped;
    int propose_calls = 0;
};

[[nodiscard]] DflashRun run_dflash(const std::vector<std::int32_t>& prompt, int budget, int accept, int eos_at,
                                   std::int32_t eos_token = 154820) {
    StepDouble model;
    model.eos_at = eos_at;
    model.eos_token = eos_token;
    int calls = 0;
    auto propose = [&](std::int32_t anchor) {
        ++calls;
        std::vector<std::int32_t> ids(7);
        for (int index = 0; index < 7; ++index) {
            ids[static_cast<std::size_t>(index)] = index < accept ? anchor + 1 + index : 999;
        }
        return ids;
    };
    DflashRun run;
    run.output = ninfer::glm53::generate_dflash_continuation(model, propose, prompt, budget);
    run.stepped = model.stepped;
    run.propose_calls = calls;
    return run;
}

void test_dflash_loop() {
    const std::vector<std::int32_t> prompt = {11, 22};
    for (int accept = 0; accept <= 7; ++accept) {
        for (int budget = 1; budget <= 8; ++budget) {
            const OracleRun oracle = greedy_oracle(prompt, budget, -1);
            const DflashRun got = run_dflash(prompt, budget, accept, -1);
            if (got.output.tokens != oracle.tokens || got.stepped != oracle.stepped ||
                static_cast<int>(got.output.tokens.size()) != budget || got.output.finish != "budget") {
                std::cerr << "dflash/oracle mismatch accept=" << accept << " budget=" << budget << '\n';
                std::exit(1);
            }
            if (budget == 1 && got.propose_calls != 0) {
                std::cerr << "budget 1 called propose\n";
                std::exit(1);
            }
            std::vector<std::int32_t> materialized;
            for (std::size_t index = prompt.size(); index < got.stepped.size(); ++index) {
                materialized.push_back(got.stepped[index]);
            }
            if (materialized != got.output.materialized_generated) {
                std::cerr << "materialized state mismatch accept=" << accept << " budget=" << budget << '\n';
                std::exit(1);
            }
        }
    }

    const DflashRun rejected = run_dflash(prompt, 8, 0, -1);
    for (const std::int32_t token : rejected.stepped) expect(token != 999, "accept 0 budget 8 leaves no 999");
    const DflashRun full = run_dflash(prompt, 8, 7, -1);
    expect(static_cast<int>(full.output.tokens.size()) == 8, "accept 7 budget 8 returns 8 tokens");

    const int fourth = static_cast<int>(prompt.size()) + 3;
    const OracleRun eos_oracle = greedy_oracle(prompt, 8, fourth);
    const DflashRun eos_run = run_dflash(prompt, 8, 7, fourth);
    expect(eos_run.output.tokens == eos_oracle.tokens, "eos tokens match the oracle");
    expect(eos_run.stepped == eos_oracle.stepped, "eos state matches the oracle");
    expect(!eos_run.output.tokens.empty() && eos_run.output.tokens.back() == 154820, "output ends with eos");
    expect(eos_run.output.finish == "eos", "eos finish");
    for (const std::int32_t token : eos_run.stepped) expect(token != 154820, "eos is not stepped");

    const int immediate = static_cast<int>(prompt.size());
    const DflashRun stopped = run_dflash(prompt, 8, 7, immediate);
    expect(stopped.propose_calls == 0, "eos at the post-prompt length does not propose");
    expect(stopped.output.tokens.size() == 1U && stopped.output.tokens[0] == 154820, "one eos token");
    expect(stopped.stepped == prompt, "immediate eos leaves the prompt state");
    expect(stopped.output.finish == "eos", "immediate eos finish");

    for (const std::int32_t eos_token : {154827, 154829}) {
        const DflashRun other = run_dflash(prompt, 4, 3, immediate, eos_token);
        const OracleRun other_oracle = greedy_oracle(prompt, 4, immediate, eos_token);
        expect(other.propose_calls == 0, "other eos ids do not propose");
        expect(other.output.tokens == other_oracle.tokens && other.stepped == other_oracle.stepped, "other eos ids");
        expect(other.output.finish == "eos" && other.output.tokens.size() == 1U && other.output.tokens[0] == eos_token,
               "other eos token is emitted");
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
    test_dflash_loop();
    return 0;
}
