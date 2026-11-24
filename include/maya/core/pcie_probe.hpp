#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstring>

namespace maya::core {

// Effective host-to-device bandwidth of the current GPU, in decimal GB/s.
// A warm-up copy raises the link out of its idle state before timing the pinned-memory transfers.
inline double probe_pcie_h2d_gbps() {
    constexpr std::size_t bytes = 256ull << 20;
    constexpr int copies = 4;
    void* host = nullptr;
    void* device = nullptr;
    cudaEvent_t begin = nullptr, end = nullptr;
    double gbps = -1.0;
    if (cudaMallocHost(&host, bytes) == cudaSuccess && cudaMalloc(&device, bytes) == cudaSuccess &&
        cudaEventCreate(&begin) == cudaSuccess && cudaEventCreate(&end) == cudaSuccess) {
        std::memset(host, 0, bytes);
        bool ok = cudaMemcpyAsync(device, host, bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
                  cudaEventRecord(begin) == cudaSuccess;
        for (int i = 0; i < copies && ok; ++i)
            ok = cudaMemcpyAsync(device, host, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
        float ms = 0;
        if (ok && cudaEventRecord(end) == cudaSuccess && cudaEventSynchronize(end) == cudaSuccess &&
            cudaEventElapsedTime(&ms, begin, end) == cudaSuccess && ms > 0.01f)
            gbps = (double) copies * bytes / (ms * 1e6);
        else
            cudaStreamSynchronize(nullptr);   // an earlier copy can still be in flight on a failed probe
    }
    if (end != nullptr) cudaEventDestroy(end);
    if (begin != nullptr) cudaEventDestroy(begin);
    if (device != nullptr) cudaFree(device);
    if (host != nullptr) cudaFreeHost(host);
    if (gbps < 0) cudaGetLastError();
    return gbps;
}

}   // namespace maya::core
