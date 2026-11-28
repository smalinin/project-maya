#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace hy4 {

inline bool save_tuning_text(const std::string& path, const std::string& text) {
    const std::string temporary = path + ".tmp." +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    std::ofstream out(temporary, std::ios::trunc);
    out << text;
    out.flush();
    const bool good = out.good();
    out.close();
    bool replaced = false;
    if (good && !out.fail()) {
#ifdef _WIN32
        replaced = MoveFileExW(std::filesystem::path(temporary).c_str(), std::filesystem::path(path).c_str(),
                               MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
        replaced = std::rename(temporary.c_str(), path.c_str()) == 0;
#endif
    }
    if (!replaced) std::remove(temporary.c_str());
    return replaced;
}

class ExpertCosts {
public:
    struct Sample { double cpu = 0, gpu = 0; int cpu_samples = 0, gpu_samples = 0; };
    ExpertCosts(int layers, std::string key) : key_(std::move(key)), samples_((size_t) layers) {}
    void cpu(int layer, double milliseconds, int entries) { observe(layer, milliseconds, entries, false); }
    void gpu(int layer, double milliseconds, int entries) { observe(layer, milliseconds, entries, true); }
    bool ready(int layer) const { return samples_[(size_t) layer].cpu_samples >= 4; }
    int ready_layers() const {
        return (int) std::count_if(samples_.begin(), samples_.end(), [](const auto& s) { return s.cpu_samples >= 4; });
    }
    double miss_ms(int layer) const {
        const auto& s = samples_[(size_t) layer];
        return ready(layer) ? std::max(0.0, s.cpu - s.gpu) : 0;
    }
    double copy_ms(size_t bytes, double host_gbs, double upload_gbs, bool evict_to_ram) const {
        return bytes * 1e-6 * (1 / std::max(0.1, host_gbs) +
                               (evict_to_ram ? 2 : 1) / std::max(0.1, upload_gbs));
    }
    double promotion_ms(int layer, double frequency_gain, double total, int used, double horizon, double copy) const {
        if (!ready(layer) || total <= 0) return 0;
        return std::max(0.0, frequency_gain) / total * used * horizon * miss_ms(layer) - copy;
    }
    bool load(const std::string& path) {
        std::ifstream in(path);
        std::string magic, key;
        int layers = 0;
        if (!(in >> magic >> layers) || magic != "HY4COST1" || layers != (int) samples_.size()) return false;
        in.ignore(1000000, '\n');
        if (!std::getline(in, key) || key != key_) return false;
        auto next = samples_;
        for (auto& s : next)
            if (!(in >> s.cpu >> s.gpu >> s.cpu_samples >> s.gpu_samples) || !std::isfinite(s.cpu) ||
                !std::isfinite(s.gpu) || s.cpu < 0 || s.gpu < 0 || s.cpu_samples < 0 || s.gpu_samples < 0 ||
                (s.cpu_samples && s.cpu <= 0) || (s.gpu_samples && s.gpu <= 0)) return false;
        samples_ = std::move(next);
        return true;
    }
    bool save(const std::string& path) const {
        std::ostringstream out;
        out << "HY4COST1 " << samples_.size() << '\n' << key_ << '\n' << std::setprecision(12);
        for (const auto& s : samples_) out << s.cpu << ' ' << s.gpu << ' ' << s.cpu_samples << ' ' << s.gpu_samples << '\n';
        return save_tuning_text(path, out.str());
    }
private:
    void observe(int layer, double milliseconds, int entries, bool gpu) {
        if (entries <= 0 || milliseconds <= 0 || !std::isfinite(milliseconds)) return;
        auto& s = samples_[(size_t) layer];
        double& value = gpu ? s.gpu : s.cpu;
        int& count = gpu ? s.gpu_samples : s.cpu_samples;
        const double next = milliseconds / entries;
        value = count ? 0.9 * value + 0.1 * next : next;
        count = std::min(1000000, count + 1);
    }
    std::string key_;
    std::vector<Sample> samples_;
};

} // namespace hy4
