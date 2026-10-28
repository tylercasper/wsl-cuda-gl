// Copyright 2026 Tyler Casper
// SPDX-License-Identifier: Apache-2.0

#include "pattern.cuh"

static __global__ void pattern_linear_kernel(uint32_t* dst, int pitch_px, int w, int h, int frame) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    dst[(size_t)y * pitch_px + x] = pattern_rgba(x, y, w, h, frame);
}

static __global__ void busy_wait_kernel(unsigned long long ns) {
    unsigned long long start, now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
    do {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    } while (now - start < ns);
}

static __global__ void copy_u32_kernel(uint32_t* dst, const uint32_t* src, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        dst[i] = src[i];
}

void launch_pattern_linear(uint32_t* dst, int pitch_px, int w, int h, int frame, cudaStream_t s) {
    dim3 block(32, 8);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    pattern_linear_kernel<<<grid, block, 0, s>>>(dst, pitch_px, w, h, frame);
}

void launch_busy_wait(int us, cudaStream_t s) {
    if (us > 0) busy_wait_kernel<<<1, 1, 0, s>>>(us * 1000ull);
}

void launch_copy_u32(uint32_t* dst, const uint32_t* src, size_t n, cudaStream_t s) {
    if (!n) return;
    unsigned blocks = (unsigned)((n + 255) / 256);
    copy_u32_kernel<<<blocks < 4096 ? blocks : 4096, 256, 0, s>>>(dst, src, n);
}
