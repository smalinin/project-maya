#pragma once

#include "hy4_plan.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <string>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace hy4 {

inline double pipeline_cost(const std::vector<double>& stages, int chunks) {
    if (stages.empty()) return std::numeric_limits<double>::infinity();
    return std::accumulate(stages.begin(), stages.end(), 0.0) +
           (std::max(1, chunks) - 1) * *std::max_element(stages.begin(), stages.end());
}

class PrefillTuning {
public:
    static constexpr std::array<int, 3> chunks{2048, 1024, 512};
    struct Sample { double ms = 0; int slots = 0, observations = 0; };
    PrefillTuning(int layers, int gpus, std::string key)
        : layers_(layers), gpus_(gpus), key_(std::move(key)), samples_(3 * layers * gpus) {}
    bool load(const std::string& file) {
        std::ifstream in(file);
        std::string magic, key;
        int layers = 0, gpus = 0;
        if (!(in >> magic >> layers >> gpus) || magic != "HY4PREF2" || layers != layers_ || gpus != gpus_) return false;
        in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        if (!std::getline(in, key) || key != key_) return false;
        auto pipeline = pipeline_;
        for (auto& sample : pipeline)
            if (!(in >> sample.ratio >> sample.observations) || !std::isfinite(sample.ratio) || sample.ratio <= 0 ||
                sample.observations < 0) return false;
        auto next = samples_;
        for (Sample& s : next)
            if (!(in >> s.ms >> s.slots >> s.observations) || !std::isfinite(s.ms) || s.ms < 0 || s.slots < 0 ||
                s.observations < 0 || (s.observations && s.ms <= 0)) return false;
        samples_ = std::move(next);
        pipeline_ = pipeline;
        return true;
    }
    bool save(const std::string& file) const {
        const std::string tmp = file + ".tmp";
        std::ofstream out(tmp, std::ios::trunc);
        out << "HY4PREF2 " << layers_ << ' ' << gpus_ << '\n' << key_ << '\n' << std::setprecision(12);
        for (const auto& sample : pipeline_) out << sample.ratio << ' ' << sample.observations << '\n';
        for (const Sample& s : samples_) out << s.ms << ' ' << s.slots << ' ' << s.observations << '\n';
        out.flush();
        const bool ok = out.good();
        out.close();
        if (!ok || out.fail()) { std::remove(tmp.c_str()); return false; }
#ifdef _WIN32
        return MoveFileExW(std::filesystem::path(tmp).c_str(), std::filesystem::path(file).c_str(),
                           MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
        return std::rename(tmp.c_str(), file.c_str()) == 0;
#endif
    }
    void observe(int chunk, int gpu, int layer, double ms, int slots) {
        const int c = index(chunk);
        if (c < 0 || ms <= 0 || !std::isfinite(ms)) return;
        Sample& sample = at(c, gpu, layer);
        sample.ms = sample.observations ? 0.65 * sample.ms + 0.35 * ms : ms;
        sample.slots = slots;
        sample.observations = std::min(1000000, sample.observations + 1);
    }
    bool ready(int chunk, const std::vector<int>& placement) const {
        const int c = index(chunk);
        if (c < 0) return false;
        for (int il = 0; il < layers_; ++il)
            if (!at(c, placement[(size_t) il], il).observations) return false;
        return true;
    }
    void observe_pipeline(int chunk, int tokens, const std::vector<int>& placement, double ms) {
        if (!ready(chunk, placement) || !std::isfinite(ms) || ms <= 0) return;
        const int c = index(chunk);
        const double expected = estimate(chunk, tokens, placement);
        if (!std::isfinite(expected) || expected <= 0) return;
        auto& sample = pipeline_[(size_t) c];
        const double ratio = ms / expected;
        if (!std::isfinite(ratio) || ratio <= 0) return;
        sample.ratio = sample.observations ? 0.65 * sample.ratio + 0.35 * ratio : ratio;
        sample.observations = std::min(1000000, sample.observations + 1);
    }
    int choose(int tokens, int maximum, const std::vector<int>& placement, bool explore = true) const {
        for (int c : chunks)
            if (c <= maximum && tokens >= 2 * c - 16 && !ready(c, placement)) return c;
        int observations = 0, least = -1;
        for (int c : chunks) {
            if (c > maximum || tokens < 2 * c - 16 || !ready(c, placement)) continue;
            const auto& sample = pipeline_[(size_t) index(c)];
            observations += sample.observations;
            if (least < 0 || sample.observations < pipeline_[(size_t) index(least)].observations) least = c;
        }
        // Revisit an alternative occasionally, so a cold first prompt or a changing CPU load cannot fix a bad choice.
        if (explore && least > 0 && observations >= 16 && observations % 16 == 0) return least;
        int best = std::min(maximum, 2048);
        double score = std::numeric_limits<double>::infinity();
        for (int c : chunks) {
            if (c > maximum || tokens < 2 * c - 16 || !ready(c, placement)) continue;
            const double cost = estimate(c, tokens, placement) * pipeline_[(size_t) index(c)].ratio;
            if (cost < score) { score = cost; best = c; }
        }
        return best;
    }
    // Cross-device estimates use the per-GPU rates observed on the current layout. Known layer/device pairs win.
    std::vector<std::vector<double>> costs(int chunk, const std::vector<int>& placement,
                                          const std::vector<double>& work) const {
        if (!ready(chunk, placement)) return {};
        std::vector<double> rate((size_t) gpus_), amount((size_t) gpus_);
        for (int il = 0; il < layers_; ++il) {
            const int gi = placement[(size_t) il];
            rate[(size_t) gi] += at(index(chunk), gi, il).ms;
            amount[(size_t) gi] += work[(size_t) il];
        }
        for (int gi = 0; gi < gpus_; ++gi) {
            if (amount[(size_t) gi] <= 0) return {};
            rate[(size_t) gi] /= amount[(size_t) gi];
        }
        std::vector<std::vector<double>> result((size_t) gpus_, std::vector<double>((size_t) layers_));
        for (int gi = 0; gi < gpus_; ++gi)
            for (int il = 0; il < layers_; ++il) {
                const Sample& known = at(index(chunk), gi, il);
                const int old = placement[(size_t) il];
                result[(size_t) gi][(size_t) il] = known.observations ? known.ms :
                    at(index(chunk), old, il).ms * rate[(size_t) gi] / rate[(size_t) old];
            }
        return result;
    }
private:
    double estimate(int chunk, int tokens, const std::vector<int>& placement) const {
        std::vector<double> stages((size_t) gpus_);
        for (int il = 0; il < layers_; ++il)
            stages[(size_t) placement[(size_t) il]] += at(index(chunk), placement[(size_t) il], il).ms;
        return pipeline_cost(stages, (tokens + chunk - 1) / chunk);
    }
    static int index(int chunk) { for (int i = 0; i < 3; ++i) if (chunks[(size_t) i] == chunk) return i; return -1; }
    Sample& at(int chunk, int gpu, int layer) { return samples_[(size_t) (chunk * gpus_ + gpu) * layers_ + layer]; }
    const Sample& at(int chunk, int gpu, int layer) const { return samples_[(size_t) (chunk * gpus_ + gpu) * layers_ + layer]; }
    int layers_, gpus_;
    std::string key_;
    std::vector<Sample> samples_;
    struct PipelineSample { double ratio = 1; int observations = 0; };
    std::array<PipelineSample, 3> pipeline_{};
};

struct PrefillPlacement {
    std::vector<int> gpu, slots;
    double cost = std::numeric_limits<double>::infinity();
};

// Shift consecutive layer boundaries using measured stage costs. Every candidate refits its own staging and experts.
inline PrefillPlacement balance_prefill(const std::vector<double>& budgets, const std::vector<LayerVram>& layers,
    const std::vector<int>& stage_slots, const std::vector<std::vector<double>>& costs,
    const std::vector<int>& current, const std::vector<int>& resident, int limit, int experts,
    const std::vector<float>& heat = {}, int chunks = 4) {
    const int n = (int) layers.size(), ng = (int) budgets.size();
    PrefillPlacement best{current, resident};
    if (ng < 2 || costs.size() != budgets.size()) return best;
    std::vector<std::vector<double>> prefixes((size_t) n, std::vector<double>((size_t) experts + 1));
    for (int il = 0; il < n; ++il) {
        std::vector<double> values((size_t) experts, 1.0);
        if (heat.size() == (size_t) n * experts)
            for (int e = 0; e < experts; ++e) values[(size_t) e] = std::max(0.0f, heat[(size_t) il * experts + e]);
        std::sort(values.rbegin(), values.rend());
        for (int e = 0; e < experts; ++e) prefixes[(size_t) il][(size_t) e + 1] = prefixes[(size_t) il][(size_t) e] + values[(size_t) e];
    }
    auto miss = [&](int il, int count) {
        const auto& p = prefixes[(size_t) il];
        return p.back() > 0 ? 1.0 - p[(size_t) std::clamp(count, 0, experts)] / p.back() : 0.0;
    };
    int before_slots = 0, routed = 0;
    double before_miss = 0;
    std::vector<double> initial((size_t) ng);
    std::vector<int> cuts{0};
    for (int il = 0; il < n; ++il) {
        if (il && current[(size_t) il] != current[(size_t) il - 1]) cuts.push_back(il);
        initial[(size_t) current[(size_t) il]] += costs[(size_t) current[(size_t) il]][(size_t) il];
        if (layers[(size_t) il].expert) { before_slots += resident[(size_t) il]; before_miss += miss(il, resident[(size_t) il]); ++routed; }
    }
    cuts.push_back(n);
    best.cost = pipeline_cost(initial, chunks);
    if ((int) cuts.size() != ng + 1) return best;
    auto evaluate = [&](const std::vector<int>& boundaries) {
        PrefillPlacement candidate{std::vector<int>((size_t) n), std::vector<int>((size_t) n)};
        std::vector<double> times((size_t) ng);
        int all_slots = 0;
        double all_miss = 0;
        for (int gi = 0; gi < ng; ++gi) {
            double fixed = 0, bytes = 0, stage = 0;
            const int begin = boundaries[(size_t) gi], end = boundaries[(size_t) gi + 1];
            if (begin >= end) return candidate;
            for (int il = begin; il < end; ++il) {
                fixed += layers[(size_t) il].fixed;
                bytes += layers[(size_t) il].expert;
                stage = std::max(stage, (double) layers[(size_t) il].expert * stage_slots[(size_t) gi]);
            }
            if (fixed + stage > budgets[(size_t) gi]) return candidate;
            const int slots = bytes > 0 ? std::min(limit, (int) ((budgets[(size_t) gi] - fixed - stage) / bytes)) : 0;
            double left = budgets[(size_t) gi] - fixed - stage - slots * bytes;
            for (int il = begin; il < end; ++il) {
                candidate.gpu[(size_t) il] = gi;
                const double size = layers[(size_t) il].expert;
                int count = size > 0 ? slots : 0;
                if (size > 0 && count < limit && left >= size) { ++count; left -= size; }
                candidate.slots[(size_t) il] = count;
                all_slots += count;
                if (size > 0) all_miss += miss(il, count);
                const double ratio = size > 0 ? (double) (experts - count + 1) / (experts - resident[(size_t) il] + 1) : 1.0;
                times[(size_t) gi] += costs[(size_t) gi][(size_t) il] * (0.5 + 0.5 * ratio);
            }
        }
        if (all_slots < before_slots * 0.98 || all_miss > before_miss * 1.04 + routed * 0.001) return candidate;
        candidate.cost = pipeline_cost(times, chunks);
        return candidate;
    };
    for (int iteration = 0; iteration < 4; ++iteration) {
        bool changed = false;
        for (int boundary = 1; boundary < ng; ++boundary) {
            auto selected = cuts;
            for (int delta : {-4, -3, -2, -1, 1, 2, 3, 4}) {
                auto next = cuts;
                next[(size_t) boundary] += delta;
                if (next[(size_t) boundary] <= next[(size_t) boundary - 1] || next[(size_t) boundary] >= next[(size_t) boundary + 1]) continue;
                auto candidate = evaluate(next);
                if (candidate.cost < best.cost * 0.97) { best = std::move(candidate); selected = std::move(next); changed = true; }
            }
            cuts = std::move(selected);
        }
        if (!changed) break;
    }
    return best;
}

}  // namespace hy4
