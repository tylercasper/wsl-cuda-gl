// Copyright 2026 Tyler Casper
// SPDX-License-Identifier: Apache-2.0

#pragma once
// The CUDA driver API, loaded from libcuda.so.1 at runtime.
//
// Using the driver API (not the runtime) means the library adds no second CUDA
// runtime instance to processes that link cudart statically, and cudaStream_t and
// CUstream are interchangeable. Loading it at runtime means binaries using the library
// still start on machines without an NVIDIA driver, where they get a clear Error.

#include <cuda.h>

#include <cstddef>
#include <string>

namespace wslcg::detail {

struct CudaDriver {
    CUresult (*Init)(unsigned int);
    CUresult (*GetErrorString)(CUresult, const char**);
    CUresult (*DeviceGet)(CUdevice*, int);
    CUresult (*DeviceGetName)(char*, int, CUdevice);
    CUresult (*DevicePrimaryCtxRetain)(CUcontext*, CUdevice);
    CUresult (*DevicePrimaryCtxRelease)(CUdevice);
    CUresult (*CtxGetCurrent)(CUcontext*);
    CUresult (*CtxPushCurrent)(CUcontext);
    CUresult (*CtxPopCurrent)(CUcontext*);
    CUresult (*CtxGetDevice)(CUdevice*);
    CUresult (*CtxSynchronize)();
    CUresult (*EventCreate)(CUevent*, unsigned int);
    CUresult (*EventRecord)(CUevent, CUstream);
    CUresult (*EventSynchronize)(CUevent);
    CUresult (*EventDestroy)(CUevent);
    CUresult (*ModuleLoadData)(CUmodule*, const void*);
    CUresult (*ModuleUnload)(CUmodule);
    CUresult (*ModuleGetFunction)(CUfunction*, CUmodule, const char*);
    CUresult (*LaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                             CUstream, void**, void**);
    CUresult (*MemAlloc)(CUdeviceptr*, size_t);
    CUresult (*MemFree)(CUdeviceptr);
    CUresult (*MemcpyDtoH)(void*, CUdeviceptr, size_t);

    // Throws Error("wsl-cuda-gl: <what> failed: <CUDA's message>") unless r succeeded.
    void check(CUresult r, const char* what) const;
    std::string error_string(CUresult r) const;
};

// Loads libcuda.so.1 and calls cuInit once per process. Returns null (and sets
// *why) if CUDA is unavailable.
const CudaDriver* try_cuda_driver(std::string* why = nullptr);
// As above, but throws Error.
const CudaDriver& cuda_driver();

// Makes `ctx` current on this thread for the lifetime of the object.
class ScopedContext {
public:
    ScopedContext(const CudaDriver& cu, CUcontext ctx);
    ~ScopedContext();
    ScopedContext(const ScopedContext&) = delete;
    ScopedContext& operator=(const ScopedContext&) = delete;

private:
    const CudaDriver& cu_;
};

// The library's own kernels (see kernels.cu), JIT-loaded from embedded PTX into the
// current context.
class Kernels {
public:
    explicit Kernels(const CudaDriver& cu);
    ~Kernels();
    Kernels(const Kernels&) = delete;
    Kernels& operator=(const Kernels&) = delete;

    // Reads one uint32 at `shared` into `*seen`, then writes `value` to `shared`.
    void selftest(CUdeviceptr shared, CUdeviceptr seen, unsigned value, CUstream stream) const;
    // Device-to-device copy that works on pointers cuMemcpy rejects.
    void copy(CUdeviceptr dst, CUdeviceptr src, size_t bytes, CUstream stream) const;

private:
    const CudaDriver& cu_;
    CUmodule module_ = nullptr;
    CUfunction selftest_ = nullptr, copy_u8_ = nullptr, copy_u32_ = nullptr, copy_u128_ = nullptr;
};

} // namespace wslcg::detail
