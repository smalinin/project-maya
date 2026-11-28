#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace hy4 {

// A request contributes at most 4096 tokens to the long-term prior. Its first 512 tokens have a separate table.
// Only settled generation routes are observed. Repeated checkpoints replace this request's contribution.
class ExpertHistory {
public:
    static constexpr uint64_t kRequestCap = 4096, kStart = 512;
    struct Data {
        uint64_t identity = 0, layers = 0, experts = 0, tokens = 0, requests = 0;
        std::vector<double> all, start;
    };

    ExpertHistory(uint64_t identity, int layers, int experts)
        : data_{identity, (uint64_t) layers, (uint64_t) experts, 0, 0,
                std::vector<double>((size_t) layers * experts), std::vector<double>((size_t) layers * experts)},
          picks_(data_.all.size()), start_(picks_.size()), saved_(picks_.size()), saved_start_(picks_.size()) {}

    const Data& data() const { return data_; }
    uint64_t pending_tokens() const { return tokens_ - saved_tokens_; }
    bool active() const { return active_; }

    bool load(const std::string& path, std::string& error) {
        Data next = data_;
        if (!read(path, next, error)) return false;
        data_ = std::move(next);
        return true;
    }

    void begin(int64_t first) {
        std::fill(picks_.begin(), picks_.end(), 0);
        std::fill(start_.begin(), start_.end(), 0);
        std::fill(saved_.begin(), saved_.end(), 0);
        std::fill(saved_start_.begin(), saved_start_.end(), 0);
        tokens_ = saved_tokens_ = 0;
        saved_request_ = false;
        first_ = end_ = first;
        active_ = true;
    }

    void observe(int64_t position, int rows, int stride, int used, const int* routes, const char* present) {
        if (!active_ || rows <= 0 || position + rows <= end_) return;
        const int from = (int) std::max<int64_t>(0, std::max(first_, end_) - position);
        if (from >= rows) return;
        bool any = false;
        for (size_t il = 0; il < data_.layers; ++il) {
            if (!present[il]) continue;
            any = true;
            for (int r = from; r < rows; ++r)
                for (int j = 0; j < used; ++j) {
                    const int e = routes[(il * stride + (size_t) r) * used + j];
                    if (e < 0 || (uint64_t) e >= data_.experts) continue;
                    const size_t key = il * data_.experts + e;
                    ++picks_[key];
                    if (tokens_ + (uint64_t) (r - from) < kStart) ++start_[key];
                }
        }
        if (any) tokens_ += (uint64_t) (rows - from);
        end_ = position + rows;
    }

    void finish() { active_ = false; }

    // Normalize each layer before mixing; a long request or a hot-profile scale cannot dominate by magnitude.
    std::vector<float> initial(const std::vector<float>& hot, int used, double weight = 0.8) const {
        if (data_.tokens == 0 || data_.requests == 0) return {};
        if (!std::isfinite(weight)) weight = 0.8;
        const double history_weight = std::clamp(weight, 0.0, 1.0) * std::min(1.0, data_.tokens / 4096.0);
        std::vector<float> result(data_.all.size());
        for (size_t il = 0; il < data_.layers; ++il) {
            const size_t b = il * data_.experts, e = b + data_.experts;
            double all = 0, start = 0, recent = 0;
            for (size_t i = b; i < e; ++i) {
                all += data_.all[i]; start += data_.start[i];
                if (i < hot.size()) recent += std::max(0.0f, hot[i]);
            }
            for (size_t i = b; i < e; ++i) {
                const double full = all > 0 ? data_.all[i] / all : 1.0 / data_.experts;
                const double early = start > 0 ? data_.start[i] / start : full;
                const double prior = 0.6 * early + 0.4 * full;
                const double now = recent > 0 && i < hot.size() ? std::max(0.0f, hot[i]) / recent : prior;
                result[i] = (float) ((history_weight * prior + (1 - history_weight) * now) * used * 1024.0);
            }
        }
        return result;
    }

    bool save(const std::string& path, std::string& error) {
        if (tokens_ == saved_tokens_) return true;
        Lock lock(path + ".lock");
        if (!lock.ok()) { error = "cannot lock " + path; return false; }
        Data next = data_;
        std::error_code ec;
        const bool exists = std::filesystem::exists(path, ec);
        if (ec) { error = "cannot inspect " + path; return false; }
        if (exists && !read(path, next, error)) return false;
        const double scale = std::min(1.0, (double) kRequestCap / std::max<uint64_t>(1, tokens_));
        for (size_t i = 0; i < picks_.size(); ++i) {
            next.all[i] = std::max(0.0, next.all[i] + picks_[i] * scale - saved_[i]);
            next.start[i] = std::max(0.0, next.start[i] + (double) start_[i] - saved_start_[i]);
        }
        next.tokens += tokens_ - saved_tokens_;
        next.requests += !saved_request_;
        if (!write(path, next, error)) return false;
        for (size_t i = 0; i < picks_.size(); ++i) {
            saved_[i] = picks_[i] * scale;
            saved_start_[i] = (double) start_[i];
        }
        data_ = std::move(next);
        saved_tokens_ = tokens_;
        saved_request_ = true;
        return true;
    }

private:
    class Lock {
    public:
        explicit Lock(const std::string& path) {
#ifdef _WIN32
            handle_ = CreateFileW(std::filesystem::path(path).c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                  nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
            OVERLAPPED overlap{};
            ok_ = handle_ != INVALID_HANDLE_VALUE && LockFileEx(handle_, LOCKFILE_EXCLUSIVE_LOCK, 0, MAXDWORD, MAXDWORD, &overlap);
#else
            fd_ = ::open(path.c_str(), O_CREAT | O_RDWR, 0666);
            ok_ = fd_ >= 0 && flock(fd_, LOCK_EX) == 0;
#endif
        }
        ~Lock() {
#ifdef _WIN32
            if (handle_ != INVALID_HANDLE_VALUE) CloseHandle(handle_);
#else
            if (fd_ >= 0) ::close(fd_);
#endif
        }
        bool ok() const { return ok_; }
    private:
        bool ok_ = false;
#ifdef _WIN32
        HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
        int fd_ = -1;
#endif
    };
    static uint64_t checksum(const Data& d) {
        uint64_t hash = 1469598103934665603ull;
        auto add = [&](const void* p, size_t n) {
            for (const auto* b = (const uint8_t*) p; n--; ++b) hash = (hash ^ *b) * 1099511628211ull;
        };
        for (uint64_t value : {d.identity, d.layers, d.experts, d.tokens, d.requests}) add(&value, sizeof value);
        add(d.all.data(), d.all.size() * sizeof(double));
        add(d.start.data(), d.start.size() * sizeof(double));
        return hash;
    }
    static bool read(const std::string& path, Data& d, std::string& error) {
        std::ifstream in(path, std::ios::binary);
        char magic[8]{};
        uint64_t header[7]{};
        if (!in.read(magic, 8) || std::memcmp(magic, "HY4HIST1", 8) ||
            !in.read((char*) header, sizeof header) || header[0] != 1 || header[1] != d.identity ||
            header[2] != d.layers || header[3] != d.experts) {
            error = "missing or incompatible expert history: " + path; return false;
        }
        Data next = d;
        next.tokens = header[4]; next.requests = header[5];
        const size_t bytes = next.all.size() * sizeof(double);
        if (!in.read((char*) next.all.data(), bytes) || !in.read((char*) next.start.data(), bytes) ||
            in.peek() != std::char_traits<char>::eof() || checksum(next) != header[6]) {
            error = "truncated or corrupt expert history: " + path; return false;
        }
        for (const auto* table : {&next.all, &next.start})
            for (double v : *table) if (!std::isfinite(v) || v < 0) {
                error = "invalid expert history values: " + path; return false;
            }
        d = std::move(next);
        return true;
    }
    static bool write(const std::string& path, const Data& d, std::string& error) {
        const std::string temporary = path + ".tmp";
        const uint64_t header[] = {1, d.identity, d.layers, d.experts, d.tokens, d.requests, checksum(d)};
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        out.write("HY4HIST1", 8);
        out.write((const char*) header, sizeof header);
        out.write((const char*) d.all.data(), d.all.size() * sizeof(double));
        out.write((const char*) d.start.data(), d.start.size() * sizeof(double));
        out.flush();
        const bool written = out.good();
        out.close();
        bool replaced = false;
        if (written && !out.fail()) {
#ifdef _WIN32
            replaced = MoveFileExW(std::filesystem::path(temporary).c_str(), std::filesystem::path(path).c_str(),
                                   MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
            replaced = std::rename(temporary.c_str(), path.c_str()) == 0;
#endif
        }
        if (replaced) return true;
        std::remove(temporary.c_str());
        error = "cannot save expert history: " + path;
        return false;
    }

    Data data_;
    std::vector<uint64_t> picks_, start_;
    std::vector<double> saved_, saved_start_;
    uint64_t tokens_ = 0, saved_tokens_ = 0;
    int64_t first_ = 0, end_ = 0;
    bool active_ = false, saved_request_ = false;
};

}  // namespace hy4
