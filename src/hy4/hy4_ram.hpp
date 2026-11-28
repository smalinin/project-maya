#pragma once

#include "hy4_cost.hpp"
#include <array>
#include <cstdint>
#include <limits>
#include <map>

namespace hy4 {

// Compare the same chunk count and approximately the same last-chunk fullness. Preparation is excluded from timing.
class RamTuning {
public:
    static constexpr std::array<int, 4> budgets{8, 16, 32, 64};
    struct Sample { double ms_token = 0; int count = 0; };
    explicit RamTuning(std::string key, int layers = 0)
        : key_(std::move(key)), pack_ms_((size_t) layers), work_ms_((size_t) layers) {}
    void layer(int il, double pack, double work) {
        if (il < 0 || (size_t) il >= pack_ms_.size() || !std::isfinite(pack) || !std::isfinite(work)) return;
        if (pack > 0) pack_ms_[(size_t) il] = pack_ms_[(size_t) il] ? 0.7 * pack_ms_[(size_t) il] + 0.3 * pack : pack;
        if (work > 0 && (pack > 0 || pack_ms_[(size_t) il] == 0))
            work_ms_[(size_t) il] = work_ms_[(size_t) il] ? 0.7 * work_ms_[(size_t) il] + 0.3 * work : work;
    }
    double packing(int il) const { return pack_ms_[(size_t) il]; }
    double fixed(int il) const { return std::max(0.0, work_ms_[(size_t) il] - pack_ms_[(size_t) il]); }
    bool calibrated(int maximum) const {
        for (const auto& [id, row] : samples_) if (complete(row, maximum)) return true;
        return false;
    }
    int initial(int maximum) const { return feasible(selected_, maximum); }
    int choose(int tokens, int chunk, int maximum) const {
        if (tokens <= chunk + 16) return initial(maximum);
        const auto found = samples_.find(shape(tokens, chunk));
        if (calibrated(maximum)) return initial(maximum);
        if (found == samples_.end()) return feasible(8, maximum);
        for (size_t i = 0; i < budgets.size(); ++i)
            if (budgets[i] <= maximum && found->second[i].count < 2) return budgets[i];
        return fastest(found->second, maximum);
    }
    void observe(int budget, int tokens, int chunk, double milliseconds, int maximum = 64) {
        if (tokens <= chunk + 16 || milliseconds <= 0 || !std::isfinite(milliseconds)) return;
        size_t i = 0;
        while (i < budgets.size() && budgets[i] != budget) ++i;
        if (i == budgets.size()) return;
        if (!primed_) { primed_ = true; return; } // A first real prompt may still fault the uncached file pages in.
        auto& row = samples_[shape(tokens, chunk)];
        auto& sample = row[i];
        const double value = milliseconds / tokens;
        sample.ms_token = sample.count ? 0.5 * sample.ms_token + 0.5 * value : value;
        sample.count = std::min(1000000, sample.count + 1);
        if (complete(row, maximum) || !calibrated(maximum)) selected_ = fastest(row, maximum);
    }
    bool load(const std::string& path) {
        std::ifstream in(path);
        std::string magic, key;
        int selected = 0;
        size_t count = 0, layers = 0;
        if (!(in >> magic >> selected >> count >> layers) || magic != "HY4RAM1" || count > 4096 || layers != pack_ms_.size() ||
            std::find(budgets.begin(), budgets.end(), selected) == budgets.end()) return false;
        in.ignore(1000000, '\n');
        if (!std::getline(in, key) || key != key_) return false;
        auto pack = pack_ms_, work = work_ms_;
        for (size_t il = 0; il < layers; ++il)
            if (!(in >> pack[il] >> work[il]) || !std::isfinite(pack[il]) || !std::isfinite(work[il]) ||
                pack[il] < 0 || work[il] < 0) return false;
        decltype(samples_) next;
        for (size_t i = 0; i < count; ++i) {
            uint64_t id = 0;
            if (!(in >> id) || next.count(id)) return false;
            auto& row = next[id];
            for (auto& s : row)
                if (!(in >> s.ms_token >> s.count) || !std::isfinite(s.ms_token) || s.ms_token < 0 ||
                    s.count < 0 || (s.count && s.ms_token <= 0)) return false;
        }
        samples_ = std::move(next);
        pack_ms_ = std::move(pack); work_ms_ = std::move(work);
        selected_ = selected;
        return true;
    }
    bool save(const std::string& path) const {
        std::ostringstream out;
        out << "HY4RAM1 " << selected_ << ' ' << samples_.size() << ' ' << pack_ms_.size() << '\n' << key_ << '\n' << std::setprecision(12);
        for (size_t il = 0; il < pack_ms_.size(); ++il) out << pack_ms_[il] << ' ' << work_ms_[il] << '\n';
        for (const auto& [id, row] : samples_) {
            out << id;
            for (const auto& s : row) out << ' ' << s.ms_token << ' ' << s.count;
            out << '\n';
        }
        return save_tuning_text(path, out.str());
    }
private:
    static bool complete(const std::array<Sample, 4>& row, int maximum) {
        bool any = false;
        for (size_t i = 0; i < budgets.size(); ++i) if (budgets[i] <= maximum) {
            any = true;
            if (row[i].count < 2) return false;
        }
        return any;
    }
    static int feasible(int desired, int maximum) {
        int result = 0;
        for (int gb : budgets) if (gb <= maximum && gb <= desired) result = gb;
        return result;
    }
    static uint64_t shape(int tokens, int chunk) {
        const int count = (tokens + chunk - 1) / chunk;
        const int tail = tokens % chunk ? tokens % chunk : chunk;
        return ((uint64_t) (uint32_t) chunk << 32) | ((uint64_t) count << 8) | (unsigned) ((4LL * tail + chunk / 2) / chunk);
    }
    static int fastest(const std::array<Sample, 4>& row, int maximum) {
        double best = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < budgets.size(); ++i)
            if (budgets[i] <= maximum && row[i].count >= 2) best = std::min(best, row[i].ms_token);
        for (size_t i = 0; i < budgets.size(); ++i)
            if (budgets[i] <= maximum && row[i].count >= 2 && row[i].ms_token <= best * 1.03) return budgets[i];
        return feasible(8, maximum);
    }
    std::string key_;
    std::vector<double> pack_ms_, work_ms_;
    int selected_ = 8;
    bool primed_ = false;
    std::map<uint64_t, std::array<Sample, 4>> samples_;
};

} // namespace hy4
