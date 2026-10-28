// Copyright 2026 Tyler Casper
// SPDX-License-Identifier: Apache-2.0

// Source of the PTX embedded in kernels_ptx.h; the build does not compile this file
// (the library has no nvcc step). Regenerate with scripts/generate_ptx.sh.
//
//   wslcg_selftest  CUDA reads what GL wrote, then writes a value for GL to read.
//   wslcg_copy_*    copies to/from a mapped buffer, whose pointer cudaMemcpy rejects.

extern "C" __global__ void wslcg_selftest(unsigned* shared, unsigned* seen, unsigned value) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *seen = *(volatile unsigned*)shared;
        *(volatile unsigned*)shared = value;
    }
}

#define WSLCG_COPY_KERNEL(name, T)                                                                    \
    extern "C" __global__ void name(T* dst, const T* src, unsigned long long n) {                     \
        unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;                       \
        for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; \
             i += stride)                                                                             \
            dst[i] = src[i];                                                                          \
    }
WSLCG_COPY_KERNEL(wslcg_copy_u8, unsigned char)
WSLCG_COPY_KERNEL(wslcg_copy_u32, unsigned)
WSLCG_COPY_KERNEL(wslcg_copy_u128, uint4)
