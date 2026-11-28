#pragma once

#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#if defined(MAYA_NATIVE_EXPERTS)
#include "ggml.h"
#endif

namespace hy4 {

// HC kernels consume F32 in row or element order. Third-party GGUFs can quantize these small matrices too.
inline bool decode_hc_weights(const uint8_t* source, size_t bytes, int type, const std::vector<int64_t>& dims,
                              int rows, int cols, bool by_element, std::vector<float>& values, std::string& error) {
    if (rows <= 0 || cols <= 0 || dims.size() < 2 || dims[0] != cols || dims[1] != rows) {
        error = "HC matrix shape does not match [" + std::to_string(cols) + ", " + std::to_string(rows) + "]";
        return false;
    }
    for (size_t i = 2; i < dims.size(); ++i) if (dims[i] != 1) {
        error = "HC matrix has an unexpected batch dimension";
        return false;
    }
    if ((size_t) cols > std::numeric_limits<size_t>::max() / sizeof(float) / (size_t) rows) {
        error = "HC matrix is too large";
        return false;
    }
    size_t stride = (size_t) cols * sizeof(float);
#if defined(MAYA_NATIVE_EXPERTS)
    if (type < 0 || type >= GGML_TYPE_COUNT) {
        error = "unsupported HC matrix type";
        return false;
    }
    const auto* traits = ggml_get_type_traits((ggml_type) type);
    if (!traits || traits->blck_size <= 0 || cols % traits->blck_size || (type != 0 && !traits->to_float)) {
        error = "HC matrix type has no decoder or its row has incomplete blocks";
        return false;
    }
    stride = (size_t) (cols / traits->blck_size) * traits->type_size;
#else
    if (type != 0) {
        error = "quantized HC matrices require MAYA_NATIVE_EXPERTS";
        return false;
    }
#endif
    if (!source || stride > std::numeric_limits<size_t>::max() / (size_t) rows || bytes != stride * (size_t) rows) {
        error = "HC matrix byte size does not match its shape and type";
        return false;
    }
    std::vector<float> decoded((size_t) rows * cols);
    if (type == 0) std::memcpy(decoded.data(), source, bytes);
#if defined(MAYA_NATIVE_EXPERTS)
    else for (int r = 0; r < rows; ++r)
        traits->to_float(source + (size_t) r * stride, decoded.data() + (size_t) r * cols, cols);
#endif
    if (by_element) {
        values.resize(decoded.size());
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                values[(size_t) c * rows + r] = decoded[(size_t) r * cols + c];
    } else values = std::move(decoded);
    return true;
}

} // namespace hy4
