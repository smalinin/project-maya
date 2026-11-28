#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace hy4 {

// Proposal plus verification time per committed token. Joint mode also explores confidence thresholds and no draft.
class AdaptiveDraft {
public:
    AdaptiveDraft(int initial, int maximum, bool enabled, float floor = 0.55f, bool joint = false,
                  bool tune_floor = true)
        : maximum_(enabled && initial >= 2 ? std::min(5, maximum) : initial),
          enabled_(enabled && initial >= 2 && maximum_ >= 2), joint_(joint && enabled_) {
        floor = std::isfinite(floor) ? std::clamp(floor, 0.0f, 1.0f) : 0.55f;
        auto add = [&](int n, float threshold) {
            for (const auto& p : plans_) if (p.n == n && std::abs(p.floor - threshold) < 1e-6f) return;
            plans_.push_back({n, threshold, {}});
        };
        add(enabled_ ? std::min(initial, maximum_) : initial, floor);
        if (!enabled_) return;
        if (joint_) { add(0, floor); add(1, floor); }
        for (int n = 2; n <= maximum_; ++n) add(n, floor);
        if (joint_ && tune_floor)
            for (float threshold : {0.0f, 0.8f})
                for (int n = 1; n <= maximum_; ++n) add(n, threshold);
    }

    int width() const { return plans_[current_].n; }
    float floor() const { return plans_[current_].floor; }
    int capacity() const { return enabled_ ? maximum_ : width(); }
    bool enabled() const { return enabled_; }

    void observe(int requested, int offered, int accepted, double milliseconds) {
        if (!enabled_ || requested != width() || requested < (joint_ ? 0 : 2) || offered < 0 || offered > requested || accepted < 0 ||
            accepted > offered || !std::isfinite(milliseconds) || milliseconds <= 0.0) return;
        Cost& c = plans_[current_].cost;
        c.milliseconds = c.samples ? 0.75 * c.milliseconds + 0.25 * milliseconds : milliseconds;
        c.tokens = c.samples ? 0.75 * c.tokens + 0.25 * (1 + accepted) : 1 + accepted;
        ++c.samples;
        const int minimum = joint_ ? 2 : 4;
        if (++run_ < minimum) return;
        run_ = 0;
        for (size_t i = 0; i < plans_.size(); ++i) {
            if (plans_[i].cost.samples < minimum) {
                current_ = i;
                return;
            }
        }
        size_t best = current_;
        for (size_t i = 0; i < plans_.size(); ++i)
            if (plans_[i].cost.rate() < plans_[best].cost.rate() * 0.97) best = i;
        current_ = best;
        if (++settled_ % (joint_ ? 32 : 8) == 0) {
            current_ = probe_;
            probe_ = (probe_ + 1) % plans_.size();
        }
    }

private:
    struct Cost {
        double milliseconds = 0, tokens = 1;
        int samples = 0;
        double rate() const { return milliseconds / tokens; }
    };
    struct Plan { int n; float floor; Cost cost; };
    int maximum_;
    bool enabled_, joint_;
    size_t current_ = 0, probe_ = 0;
    int run_ = 0, settled_ = 0;
    std::vector<Plan> plans_;
};

}  // namespace hy4
