#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace hy4 {

// Cost per committed token includes both proposing and verifying a window. Probe each width in short runs,
// then keep the fastest. Periodic probes let a later part of the answer choose a different width.
class AdaptiveDraft {
public:
    AdaptiveDraft(int initial, int maximum, bool enabled)
        : current_(initial), maximum_(enabled && initial >= 2 ? std::min(5, maximum) : initial),
          enabled_(enabled && initial >= 2 && maximum_ >= 2) {
        if (enabled_) current_ = std::min(current_, maximum_);
    }

    int width() const { return current_; }
    int capacity() const { return enabled_ ? maximum_ : current_; }
    bool enabled() const { return enabled_; }

    void observe(int requested, int offered, int accepted, double milliseconds) {
        if (!enabled_ || requested != current_ || requested < 2 || offered < 0 || offered > requested || accepted < 0 ||
            accepted > offered || !std::isfinite(milliseconds) || milliseconds <= 0.0) return;
        Cost& c = costs_[(size_t) current_];
        c.milliseconds = c.samples ? 0.75 * c.milliseconds + 0.25 * milliseconds : milliseconds;
        c.tokens = c.samples ? 0.75 * c.tokens + 0.25 * (1 + accepted) : 1 + accepted;
        ++c.samples;
        if (++run_ < 4) return;
        run_ = 0;
        for (int n = 2; n <= maximum_; ++n) {
            if (costs_[(size_t) n].samples < 4) {
                current_ = n;
                return;
            }
        }
        int best = current_;
        for (int n = 2; n <= maximum_; ++n)
            if (costs_[(size_t) n].rate() < costs_[(size_t) best].rate() * 0.97) best = n;
        current_ = best;
        if (++settled_ % 8 == 0) {
            current_ = probe_;
            probe_ = probe_ == maximum_ ? 2 : probe_ + 1;
        }
    }

private:
    struct Cost {
        double milliseconds = 0, tokens = 1;
        int samples = 0;
        double rate() const { return milliseconds / tokens; }
    };
    int current_, maximum_;
    bool enabled_;
    int run_ = 0, settled_ = 0, probe_ = 2;
    std::array<Cost, 6> costs_{};
};

}  // namespace hy4
