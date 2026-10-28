// Copyright 2026 Tyler Casper
// SPDX-License-Identifier: Apache-2.0

#include "cuda_api.h"

#include <algorithm>
#include <dlfcn.h>
#include <mutex>

#include "kernels_ptx.h"
#include "wsl_cuda_gl.h"

namespace wslcg::detail {
namespace {

template <typename Fn>
bool resolve(void* lib, Fn& fn, const char* name, std::string* missing) {
    fn = reinterpret_cast<Fn>(dlsym(lib, name));
    if (!fn && missing->empty()) *missing = name;
    return fn != nullptr;
}

} // namespace

const CudaDriver* try_cuda_driver(std::string* why) {
    static CudaDriver driver;
    static std::string error;
    static std::once_flag once;
    std::call_once(once, [] {
        void* lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!lib) {
            error = std::string("cannot load the CUDA driver (libcuda.so.1): ") + dlerror();
            return;
        }
        CudaDriver& d = driver;
        std::string missing;
        resolve(lib, d.Init, "cuInit", &missing);
        resolve(lib, d.GetErrorString, "cuGetErrorString", &missing);
        resolve(lib, d.DeviceGet, "cuDeviceGet", &missing);
        resolve(lib, d.DeviceGetName, "cuDeviceGetName", &missing);
        resolve(lib, d.DevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain", &missing);
        resolve(lib, d.DevicePrimaryCtxRelease, "cuDevicePrimaryCtxRelease_v2", &missing);
        resolve(lib, d.CtxGetCurrent, "cuCtxGetCurrent", &missing);
        resolve(lib, d.CtxPushCurrent, "cuCtxPushCurrent_v2", &missing);
        resolve(lib, d.CtxPopCurrent, "cuCtxPopCurrent_v2", &missing);
        resolve(lib, d.CtxGetDevice, "cuCtxGetDevice", &missing);
        resolve(lib, d.CtxSynchronize, "cuCtxSynchronize", &missing);
        resolve(lib, d.EventCreate, "cuEventCreate", &missing);
        resolve(lib, d.EventRecord, "cuEventRecord", &missing);
        resolve(lib, d.EventSynchronize, "cuEventSynchronize", &missing);
        resolve(lib, d.EventDestroy, "cuEventDestroy_v2", &missing);
        resolve(lib, d.ModuleLoadData, "cuModuleLoadData", &missing);
        resolve(lib, d.ModuleUnload, "cuModuleUnload", &missing);
        resolve(lib, d.ModuleGetFunction, "cuModuleGetFunction", &missing);
        resolve(lib, d.LaunchKernel, "cuLaunchKernel", &missing);
        resolve(lib, d.MemAlloc, "cuMemAlloc_v2", &missing);
        resolve(lib, d.MemFree, "cuMemFree_v2", &missing);
        resolve(lib, d.MemcpyDtoH, "cuMemcpyDtoH_v2", &missing);
        if (!missing.empty()) {
            error = "the CUDA driver lacks " + missing;
            return;
        }
        CUresult r = d.Init(0);
        if (r != CUDA_SUCCESS) error = "cuInit failed: " + d.error_string(r);
    });
    if (!error.empty()) {
        if (why) *why = error;
        return nullptr;
    }
    return &driver;
}

const CudaDriver& cuda_driver() {
    std::string why;
    const CudaDriver* d = try_cuda_driver(&why);
    if (!d) throw Error("wsl-cuda-gl: " + why);
    return *d;
}

std::string CudaDriver::error_string(CUresult r) const {
    const char* s = nullptr;
    GetErrorString(r, &s);
    return s ? s : "CUDA error " + std::to_string((int)r);
}

void CudaDriver::check(CUresult r, const char* what) const {
    if (r != CUDA_SUCCESS) throw Error(std::string("wsl-cuda-gl: ") + what + " failed: " + error_string(r));
}

ScopedContext::ScopedContext(const CudaDriver& cu, CUcontext ctx) : cu_(cu) {
    cu.check(cu.CtxPushCurrent(ctx), "cuCtxPushCurrent");
}

ScopedContext::~ScopedContext() {
    CUcontext ignored;
    cu_.CtxPopCurrent(&ignored);
}

Kernels::Kernels(const CudaDriver& cu) : cu_(cu) {
    cu.check(cu.ModuleLoadData(&module_, kKernelsPtx), "loading the library's PTX kernels");
    cu.check(cu.ModuleGetFunction(&selftest_, module_, "wslcg_selftest"), "cuModuleGetFunction");
    cu.check(cu.ModuleGetFunction(&copy_u8_, module_, "wslcg_copy_u8"), "cuModuleGetFunction");
    cu.check(cu.ModuleGetFunction(&copy_u32_, module_, "wslcg_copy_u32"), "cuModuleGetFunction");
    cu.check(cu.ModuleGetFunction(&copy_u128_, module_, "wslcg_copy_u128"), "cuModuleGetFunction");
}

Kernels::~Kernels() {
    if (module_) cu_.ModuleUnload(module_);
}

void Kernels::selftest(CUdeviceptr shared, CUdeviceptr seen, unsigned value, CUstream stream) const {
    void* args[] = {&shared, &seen, &value};
    cu_.check(cu_.LaunchKernel(selftest_, 1, 1, 1, 1, 1, 1, 0, stream, args, nullptr), "launching the self-test");
}

void Kernels::copy(CUdeviceptr dst, CUdeviceptr src, size_t bytes, CUstream stream) const {
    if (!bytes) return;
    // Widest element that dst, src and the size are all aligned to.
    CUfunction fn = copy_u8_;
    size_t width = 1;
    if (((dst | src | bytes) & 15) == 0) {
        fn = copy_u128_;
        width = 16;
    } else if (((dst | src | bytes) & 3) == 0) {
        fn = copy_u32_;
        width = 4;
    }
    unsigned long long n = bytes / width;
    const unsigned threads = 256;
    unsigned blocks = (unsigned)std::min<unsigned long long>((n + threads - 1) / threads, 4096);
    void* args[] = {&dst, &src, &n};
    cu_.check(cu_.LaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, stream, args, nullptr), "launching a copy");
}

} // namespace wslcg::detail
