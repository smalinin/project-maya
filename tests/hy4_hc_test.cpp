#include "../src/hy4/hy4_hc.hpp"

#include <cmath>
#include <cstdio>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); return 1; } } while (0)

int main() {
    constexpr int cols = 512;
    for (int rows : {4, 8}) {
        std::vector<float> source((size_t) rows * cols);
        for (size_t i = 0; i < source.size(); ++i) source[i] = (float) ((int) (i % 251) - 125) * 0.03125f;
        for (int type : {0, 1, 30, 8, 10, 11, 12, 13, 14, 20}) {
            const auto* traits = ggml_get_type_traits((ggml_type) type);
            CHECK(type == 0 || traits->from_float_ref);
            const size_t stride = cols / traits->blck_size * traits->type_size;
            std::vector<uint8_t> bytes((size_t) rows * stride);
            std::vector<float> reference(source.size());
            if (type == 0) {
                std::memcpy(bytes.data(), source.data(), bytes.size()); reference = source;
            } else for (int r = 0; r < rows; ++r) {
                traits->from_float_ref(source.data() + (size_t) r * cols, bytes.data() + (size_t) r * stride, cols);
                traits->to_float(bytes.data() + (size_t) r * stride, reference.data() + (size_t) r * cols, cols);
            }
            std::vector<float> decoded, by_element;
            std::string error;
            CHECK(hy4::decode_hc_weights(bytes.data(), bytes.size(), type, {cols, rows}, rows, cols, false, decoded, error));
            CHECK(decoded == reference);
            CHECK(hy4::decode_hc_weights(bytes.data(), bytes.size(), type, {cols, rows, 1}, rows, cols, true, by_element, error));
            for (int r = 0; r < rows; ++r) for (int c = 0; c < cols; ++c)
                CHECK(by_element[(size_t) c * rows + r] == reference[(size_t) r * cols + c]);
            CHECK(!hy4::decode_hc_weights(bytes.data(), bytes.size() - 1, type, {cols, rows}, rows, cols, false, decoded, error));
            CHECK(!hy4::decode_hc_weights(bytes.data(), bytes.size(), type, {rows, cols}, rows, cols, false, decoded, error));
            CHECK(!hy4::decode_hc_weights(bytes.data(), bytes.size(), type, {cols, rows, 2}, rows, cols, false, decoded, error));
            CHECK(decoded == reference); // A rejected tensor leaves the previous result intact.
        }
    }
    // Independent Q8_0 values: FP16 scale 2, signed codes, distinct blocks and rows.
    std::vector<uint8_t> q8(8 * 34);
    for (int b = 0; b < 8; ++b) {
        const uint16_t scale = 0x4000;
        std::memcpy(q8.data() + b * 34, &scale, 2);
        for (int i = 0; i < 32; ++i) q8[b * 34 + 2 + i] = (uint8_t) (int8_t) (i + b - 16);
    }
    std::vector<float> decoded;
    std::string error;
    CHECK(hy4::decode_hc_weights(q8.data(), q8.size(), 8, {64, 4}, 4, 64, true, decoded, error));
    for (int r = 0; r < 4; ++r) for (int c = 0; c < 64; ++c)
        CHECK(decoded[(size_t) c * 4 + r] == 2.0f * (c % 32 + (r * 64 + c) / 32 - 16));
    CHECK(!hy4::decode_hc_weights(q8.data(), q8.size(), -1, {64, 4}, 4, 64, false, decoded, error));
    CHECK(!hy4::decode_hc_weights(q8.data(), q8.size(), 8, {63, 4}, 4, 63, false, decoded, error));
    CHECK(!hy4::decode_hc_weights(q8.data(), q8.size(), 26, {64, 4}, 4, 64, false, decoded, error));
    std::puts("hy4_hc_test: ok");
}
