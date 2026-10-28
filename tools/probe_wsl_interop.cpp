// Copyright 2026 Tyler Casper
// SPDX-License-Identifier: Apache-2.0

// Probe: which cross-API memory/sync handles actually work between CUDA and D3D12
// on this WSL install? (Mesa's GL on WSL imports D3D12 shared resources and fences,
// so D3D12 is the common currency.)
//
// Each test runs in its own process (this binary re-executes itself), because some
// combinations crash inside the driver rather than returning an error.
//
//   probe_wsl_interop            run every test
//   probe_wsl_interop <test>     run one test in-process
//
// Exits with 77 (CTest's "skipped") when not running on WSL with a GPU.

#include <wsl/winadapter.h> // must precede the D3D12 headers on Linux
#include <directx/d3d12.h>
#include <directx/dxcore.h>
#include <dxguids/dxguids.h>
#include <wrl/client.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "pattern.cuh"

#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <functional>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

using Microsoft::WRL::ComPtr;

#ifndef GENERIC_ALL
#define GENERIC_ALL 0x10000000L
#endif

// D3D12 is loaded at run time (RTLD_LOCAL, as Mesa and the library do) so the probe
// builds and starts on any Linux machine.
static PFN_D3D12_CREATE_DEVICE pD3D12CreateDevice;
static HRESULT (*pDXCoreCreateAdapterFactory)(REFIID, void**);
static bool load_d3d12() {
    void* d3d12 = dlopen("libd3d12.so", RTLD_NOW | RTLD_LOCAL);
    void* dxcore = dlopen("libdxcore.so", RTLD_NOW | RTLD_LOCAL);
    if (!d3d12 || !dxcore) return false;
    pD3D12CreateDevice = (PFN_D3D12_CREATE_DEVICE)dlsym(d3d12, "D3D12CreateDevice");
    pDXCoreCreateAdapterFactory = (decltype(pDXCoreCreateAdapterFactory))dlsym(dxcore, "DXCoreCreateAdapterFactory");
    return pD3D12CreateDevice && pDXCoreCreateAdapterFactory;
}
static const GUID kDxcoreD3D12Graphics = {0x0c9ece4d, 0x2f6e, 0x4f01, {0x8c, 0x96, 0xe8, 0x9e, 0x33, 0x1b, 0x47, 0xb1}};

static std::string describe_fd(int fd) {
    char path[64], target[256] = {};
    snprintf(path, sizeof path, "/proc/self/fd/%d", fd);
    ssize_t n = readlink(path, target, sizeof target - 1);
    return n > 0 ? std::string(target, n) : std::string("?");
}

static void line(const char* what, bool ok, const std::string& detail = "") {
    printf("    %-58s %s%s%s\n", what, ok ? "OK" : "FAIL", detail.empty() ? "" : ": ", detail.c_str());
}
static bool cu_ok(const char* what, CUresult r) {
    const char* s = "";
    if (r != CUDA_SUCCESS) cuGetErrorString(r, &s);
    line(what, r == CUDA_SUCCESS, s);
    return r == CUDA_SUCCESS;
}
static bool rt_ok(const char* what, cudaError_t e) {
    line(what, e == cudaSuccess, e == cudaSuccess ? "" : cudaGetErrorString(e));
    cudaGetLastError();
    return e == cudaSuccess;
}
static bool hr_ok(const char* what, HRESULT hr) {
    char buf[32];
    snprintf(buf, sizeof buf, "hr=0x%08x", (unsigned)hr);
    line(what, SUCCEEDED(hr), SUCCEEDED(hr) ? "" : buf);
    return SUCCEEDED(hr);
}

// ---------------------------------------------------------------------------------
// Shared setup

static constexpr size_t kSize = 8u << 20; // 8 MiB: a multiple of CUDA's 2 MiB VMM granularity

static ComPtr<ID3D12Device> make_d3d12_device() {
    ComPtr<IDXCoreAdapterFactory> factory;
    pDXCoreCreateAdapterFactory(IID_PPV_ARGS(&factory));
    ComPtr<IDXCoreAdapterList> list;
    factory->CreateAdapterList(1, &kDxcoreD3D12Graphics, IID_PPV_ARGS(&list));
    for (uint32_t i = 0; i < list->GetAdapterCount(); ++i) {
        ComPtr<IDXCoreAdapter> a;
        list->GetAdapter(i, IID_PPV_ARGS(&a));
        char desc[256] = {};
        a->GetProperty(DXCoreAdapterProperty::DriverDescription, sizeof desc, desc);
        if (!strstr(desc, "NVIDIA")) continue;
        ComPtr<ID3D12Device> dev;
        if (SUCCEEDED(pD3D12CreateDevice(a.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&dev)))) return dev;
    }
    fprintf(stderr, "no NVIDIA D3D12 device\n");
    exit(2);
}

static ComPtr<ID3D12Resource> make_shared_buffer(ID3D12Device* dev, size_t size) {
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC d = {};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = size;
    d.Height = 1;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    d.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ComPtr<ID3D12Resource> r;
    hr_ok("D3D12 CreateCommittedResource(buffer, HEAP_FLAG_SHARED)",
          dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_SHARED, &d, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                       IID_PPV_ARGS(&r)));
    return r;
}

static int share(ID3D12Device* dev, ID3D12DeviceChild* obj, const char* what) {
    HANDLE h = nullptr;
    if (!hr_ok(what, dev->CreateSharedHandle(obj, nullptr, GENERIC_ALL, nullptr, &h))) exit(1);
    int fd = (int)(intptr_t)h;
    printf("      (fd %d -> %s)\n", fd, describe_fd(fd).c_str());
    return fd;
}

static void init_cuda() {
    cudaFree(0); // creates the primary context and makes it current for the driver API
}

// Copy `src` (a buffer) to a readback heap with D3D12 and return its contents.
static std::vector<uint32_t> d3d12_readback(ID3D12Device* dev, ID3D12Resource* src, size_t size) {
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC d = src->GetDesc();
    d.Flags = D3D12_RESOURCE_FLAG_NONE;
    ComPtr<ID3D12Resource> rb;
    dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                 IID_PPV_ARGS(&rb));
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ComPtr<ID3D12CommandQueue> q;
    dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&q));
    ComPtr<ID3D12CommandAllocator> alloc;
    dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc));
    ComPtr<ID3D12GraphicsCommandList> cl;
    dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.Get(), nullptr, IID_PPV_ARGS(&cl));
    cl->CopyBufferRegion(rb.Get(), 0, src, 0, size);
    cl->Close();
    ID3D12CommandList* lists[] = {cl.Get()};
    q->ExecuteCommandLists(1, lists);
    ComPtr<ID3D12Fence> f;
    dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f));
    q->Signal(f.Get(), 1);
    f->SetEventOnCompletion(1, nullptr); // null event = block until reached
    std::vector<uint32_t> out(size / 4);
    void* p = nullptr;
    D3D12_RANGE r = {0, size};
    rb->Map(0, &r, &p);
    memcpy(out.data(), p, size);
    rb->Unmap(0, nullptr);
    return out;
}

// Import a dxgresource fd through CUDA's VMM API and map it. Returns 0 on failure.
static CUdeviceptr cuda_vmm_map(int fd, size_t size) {
    CUmemGenericAllocationHandle mh;
    if (!cu_ok("cuMemImportFromShareableHandle(POSIX_FILE_DESCRIPTOR)",
               cuMemImportFromShareableHandle(&mh, (void*)(intptr_t)fd, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)))
        return 0;
    CUmemAllocationProp ap = {};
    if (cu_ok("cuMemGetAllocationPropertiesFromHandle", cuMemGetAllocationPropertiesFromHandle(&ap, mh)))
        printf("      (type %d, location type %d id %d, handle types 0x%x, compression %d, gpuDirectRDMA %d, usage 0x%x)\n",
               (int)ap.type, (int)ap.location.type, ap.location.id, (unsigned)ap.requestedHandleTypes,
               (int)ap.allocFlags.compressionType, (int)ap.allocFlags.gpuDirectRDMACapable, (unsigned)ap.allocFlags.usage);
    size_t gran = 0;
    cuMemGetAllocationGranularity(&gran, &ap, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    printf("      (CUDA granularity for these props: %zu KiB)\n", gran >> 10);
    CUdeviceptr va = 0;
    if (!cu_ok("cuMemAddressReserve", cuMemAddressReserve(&va, size, 2u << 20, 0, 0))) return 0;
    if (!cu_ok("cuMemMap", cuMemMap(va, size, 0, mh, 0))) return 0;
    CUmemAccessDesc ad = {};
    ad.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    ad.location.id = 0;
    ad.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (!cu_ok("cuMemSetAccess(READWRITE)", cuMemSetAccess(va, size, &ad, 1))) return 0;
    cuMemRelease(mh); // the mapping keeps the allocation alive
    return va;
}

// ---------------------------------------------------------------------------------
// Tests

static int test_info() {
    init_cuda();
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    CUdevice cudev;
    cuDeviceGet(&cudev, 0);
    auto attr = [&](CUdevice_attribute a) {
        int v = -1;
        cuDeviceGetAttribute(&v, a, cudev);
        return v;
    };
    int drv = 0;
    cudaDriverGetVersion(&drv);
    printf("    CUDA device 0: %s, driver API %d\n", prop.name, drv);
    printf("    VMM: %d, POSIX fd export: %d, Win32 handle export: %d\n",
           attr(CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED),
           attr(CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED),
           attr(CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED));
    ComPtr<IDXCoreAdapterFactory> factory;
    pDXCoreCreateAdapterFactory(IID_PPV_ARGS(&factory));
    ComPtr<IDXCoreAdapterList> list;
    factory->CreateAdapterList(1, &kDxcoreD3D12Graphics, IID_PPV_ARGS(&list));
    for (uint32_t i = 0; i < list->GetAdapterCount(); ++i) {
        ComPtr<IDXCoreAdapter> a;
        list->GetAdapter(i, IID_PPV_ARGS(&a));
        char desc[256] = {};
        a->GetProperty(DXCoreAdapterProperty::DriverDescription, sizeof desc, desc);
        printf("    D3D12 adapter %u: %s\n", i, desc);
    }
    return 0;
}

// CUDA's external-memory API with D3D12-created handles (the documented route on Windows).
static int test_d3d12_to_cuda_extmem() {
    auto dev = make_d3d12_device();
    init_cuda();
    auto buf = make_shared_buffer(dev.Get(), kSize);
    int fd = share(dev.Get(), buf.Get(), "D3D12 CreateSharedHandle(buffer)");
    struct { const char* label; cudaExternalMemoryHandleType type; unsigned flags; } variants[] = {
        {"cudaImportExternalMemory(D3D12Resource, dedicated)", cudaExternalMemoryHandleTypeD3D12Resource, cudaExternalMemoryDedicated},
        {"cudaImportExternalMemory(OpaqueFd)", cudaExternalMemoryHandleTypeOpaqueFd, 0},
        {"cudaImportExternalMemory(OpaqueWin32)", cudaExternalMemoryHandleTypeOpaqueWin32, 0},
    };
    for (auto& v : variants) {
        cudaExternalMemoryHandleDesc d = {};
        d.type = v.type;
        int dfd = dup(fd);
        if (v.type == cudaExternalMemoryHandleTypeOpaqueFd) d.handle.fd = dfd;
        else d.handle.win32.handle = (void*)(intptr_t)dfd;
        d.size = kSize;
        d.flags = v.flags;
        cudaExternalMemory_t ext;
        rt_ok(v.label, cudaImportExternalMemory(&ext, &d));
    }
    return 0;
}

static int test_d3d12_fence_to_cuda() {
    auto dev = make_d3d12_device();
    init_cuda();
    ComPtr<ID3D12Fence> fence;
    hr_ok("D3D12 CreateFence(FENCE_FLAG_SHARED)", dev->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&fence)));
    int fd = share(dev.Get(), fence.Get(), "D3D12 CreateSharedHandle(fence)");
    struct { const char* label; cudaExternalSemaphoreHandleType type; } variants[] = {
        {"cudaImportExternalSemaphore(D3D12Fence)", cudaExternalSemaphoreHandleTypeD3D12Fence},
        {"cudaImportExternalSemaphore(TimelineSemaphoreFd)", cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd},
        {"cudaImportExternalSemaphore(OpaqueFd)", cudaExternalSemaphoreHandleTypeOpaqueFd},
    };
    for (auto& v : variants) {
        cudaExternalSemaphoreHandleDesc d = {};
        d.type = v.type;
        int dfd = dup(fd);
        if (v.type == cudaExternalSemaphoreHandleTypeD3D12Fence) d.handle.win32.handle = (void*)(intptr_t)dfd;
        else d.handle.fd = dfd;
        cudaExternalSemaphore_t sem;
        rt_ok(v.label, cudaImportExternalSemaphore(&sem, &d));
    }
    return 0;
}

// CUDA allocates exportable memory; D3D12 tries to open it. `iid` picks the interface.
static int test_cuda_to_d3d12(const std::string& iid) {
    auto dev = make_d3d12_device();
    init_cuda();
    CUmemAllocationProp ap = {};
    ap.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    ap.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    ap.location.id = 0;
    ap.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    CUmemGenericAllocationHandle mh;
    if (!cu_ok("cuMemCreate(POSIX fd exportable)", cuMemCreate(&mh, kSize, &ap, 0))) return 1;
    int fd = -1;
    if (!cu_ok("cuMemExportToShareableHandle(POSIX fd)",
               cuMemExportToShareableHandle(&fd, mh, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0)))
        return 1;
    printf("      (fd %d -> %s)\n", fd, describe_fd(fd).c_str());
    fflush(stdout);
    HANDLE h = (HANDLE)(intptr_t)fd;
    if (iid == "heap") {
        ComPtr<ID3D12Heap> o;
        hr_ok("D3D12 OpenSharedHandle(CUDA fd) as ID3D12Heap", dev->OpenSharedHandle(h, IID_PPV_ARGS(&o)));
    } else if (iid == "resource") {
        ComPtr<ID3D12Resource> o;
        hr_ok("D3D12 OpenSharedHandle(CUDA fd) as ID3D12Resource", dev->OpenSharedHandle(h, IID_PPV_ARGS(&o)));
    } else {
        ComPtr<IUnknown> o;
        hr_ok("D3D12 OpenSharedHandle(CUDA fd) as IUnknown", dev->OpenSharedHandle(h, IID_PPV_ARGS(&o)));
    }
    return 0;
}

// D3D12 allocates shared memory; CUDA maps it through the VMM import path; CUDA
// writes; D3D12 copies it back and checks. Success means both APIs address the same
// device memory.
//   heap_align == 0: a committed buffer of `size` bytes
//   heap_align != 0: a shared heap with that alignment, and a buffer placed at offset 0
static int test_d3d12_to_cuda_vmm(size_t size, size_t heap_align) {
    auto dev = make_d3d12_device();
    init_cuda();
    ComPtr<ID3D12Resource> buf;
    ComPtr<ID3D12Heap> heap;
    int fd;
    if (!heap_align) {
        buf = make_shared_buffer(dev.Get(), size);
        fd = share(dev.Get(), buf.Get(), "D3D12 CreateSharedHandle(committed buffer)");
    } else {
        D3D12_HEAP_DESC hd = {};
        hd.SizeInBytes = size;
        hd.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
        hd.Alignment = heap_align;
        hd.Flags = D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;
        if (!hr_ok("D3D12 CreateHeap(SHARED | ALLOW_ONLY_BUFFERS)", dev->CreateHeap(&hd, IID_PPV_ARGS(&heap)))) return 1;
        D3D12_RESOURCE_DESC d = {};
        d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        d.Width = size;
        d.Height = 1;
        d.DepthOrArraySize = 1;
        d.MipLevels = 1;
        d.SampleDesc.Count = 1;
        d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        if (!hr_ok("D3D12 CreatePlacedResource(buffer @ 0)",
                   dev->CreatePlacedResource(heap.Get(), 0, &d, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&buf))))
            return 1;
        fd = share(dev.Get(), heap.Get(), "D3D12 CreateSharedHandle(heap)");
    }
    CUdeviceptr va = cuda_vmm_map(fd, size);
    if (!va) return 1;
    const uint32_t magic = 0xC0DA5EED;
    if (!cu_ok("cuMemsetD32 through the mapping", cuMemsetD32(va, magic, size / 4))) return 1;
    if (!cu_ok("cuCtxSynchronize", cuCtxSynchronize())) return 1;
    auto data = d3d12_readback(dev.Get(), buf.Get(), size);
    size_t bad = 0;
    for (uint32_t v : data) bad += v != magic;
    char detail[96];
    snprintf(detail, sizeof detail, "%zu of %zu words differ (first word %08x)", bad, data.size(), data[0]);
    line("D3D12 reads back what CUDA wrote", bad == 0, bad ? detail : "");
    return bad ? 1 : 0;
}

// No import API at all: under WDDM the GPU virtual address space is per process, so
// a D3D12 buffer's GPU VA may be directly usable as a CUDA device pointer.
static int test_d3d12_gpuva_to_cuda() {
    auto dev = make_d3d12_device();
    init_cuda();
    auto buf = make_shared_buffer(dev.Get(), kSize);
    D3D12_GPU_VIRTUAL_ADDRESS va = buf->GetGPUVirtualAddress();
    void* cuda_ptr = nullptr;
    cudaMalloc(&cuda_ptr, 1 << 20);
    printf("      (D3D12 buffer GPU VA 0x%llx; for comparison a cudaMalloc pointer is %p)\n",
           (unsigned long long)va, cuda_ptr);
    CUmemorytype mt;
    CUresult r = cuPointerGetAttribute(&mt, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, (CUdeviceptr)va);
    const char* es = "";
    if (r != CUDA_SUCCESS) cuGetErrorString(r, &es);
    printf("      (cuPointerGetAttribute on it: %s)\n", r == CUDA_SUCCESS ? "known to CUDA" : es);

    const int w = 1920, h = 1080, frame = 7;
    launch_pattern_linear((uint32_t*)va, w, w, h, frame, 0);
    if (!rt_ok("CUDA kernel writes through the D3D12 GPU VA", cudaDeviceSynchronize())) return 1;
    auto data = d3d12_readback(dev.Get(), buf.Get(), kSize);
    size_t bad = 0;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) bad += data[(size_t)y * w + x] != pattern_rgba(x, y, w, h, frame);
    char detail[96];
    snprintf(detail, sizeof detail, "%zu of %d pixels differ (first word %08x)", bad, w * h, data[0]);
    line("D3D12 reads back exactly what the CUDA kernel wrote", bad == 0, bad ? detail : "");
    return bad ? 1 : 0;
}

// Mesa's GL_EXT_memory_object import (d3d12_memobj_create_from_handle) opens the
// shared handle as IUnknown and C-casts that pointer to ID3D12Resource*. That only
// works if the runtime's IUnknown identity pointer IS the ID3D12Resource pointer.
static int test_iunknown_identity() {
    auto dev = make_d3d12_device();
    auto buf = make_shared_buffer(dev.Get(), kSize);
    int fd = share(dev.Get(), buf.Get(), "D3D12 CreateSharedHandle(buffer)");
    ComPtr<IUnknown> unk;
    hr_ok("OpenSharedHandle as IUnknown (what Mesa does)", dev->OpenSharedHandle((HANDLE)(intptr_t)fd, IID_PPV_ARGS(&unk)));
    ComPtr<ID3D12Resource> res;
    hr_ok("QueryInterface(ID3D12Resource) on it", unk.As(&res));
    ComPtr<IUnknown> own_unk;
    buf.As(&own_unk);
    printf("      (opened IUnknown %p, its ID3D12Resource %p; original resource %p, its IUnknown %p)\n",
           (void*)unk.Get(), (void*)res.Get(), (void*)buf.Get(), (void*)own_unk.Get());
    line("IUnknown pointer == ID3D12Resource pointer (Mesa's cast is valid)", (void*)unk.Get() == (void*)res.Get());
    return 0;
}

// Which CUDA API calls (not just kernels) accept a D3D12 GPU VA?
static int test_gpuva_api_calls() {
    auto dev = make_d3d12_device();
    init_cuda();
    auto buf = make_shared_buffer(dev.Get(), kSize);
    auto va = (CUdeviceptr)buf->GetGPUVirtualAddress();
    void* p = (void*)va;
    std::vector<uint32_t> host(kSize / 4, 0x11111111), back(kSize / 4);
    void* dsrc = nullptr;
    cudaMalloc(&dsrc, kSize);
    cudaMemset(dsrc, 0x22, kSize);
    auto check = [&](const char* what, cudaError_t e, uint32_t expect) {
        if (e == cudaSuccess) e = cudaDeviceSynchronize();
        bool ok = rt_ok(what, e);
        if (!ok) return;
        auto d = d3d12_readback(dev.Get(), buf.Get(), kSize);
        size_t bad = 0;
        for (uint32_t v : d) bad += v != expect;
        line("    -> D3D12 sees the result", bad == 0);
    };
    check("cudaMemset(ptr)", cudaMemset(p, 0x33, kSize), 0x33333333);
    check("cudaMemcpy H2D", cudaMemcpy(p, host.data(), kSize, cudaMemcpyHostToDevice), 0x11111111);
    check("cudaMemcpy D2D (cudaMalloc -> ptr)", cudaMemcpy(p, dsrc, kSize, cudaMemcpyDeviceToDevice), 0x22222222);
    check("cudaMemcpyAsync D2D", cudaMemcpyAsync(p, dsrc, kSize, cudaMemcpyDeviceToDevice, 0), 0x22222222);
    check("cudaMemcpy2D D2D", cudaMemcpy2D(p, 4096, dsrc, 4096, 4096, kSize / 4096, cudaMemcpyDeviceToDevice), 0x22222222);
    cu_ok("cuMemsetD32(ptr)", cuMemsetD32(va, 0x44444444, kSize / 4));
    rt_ok("cudaMemcpy D2H (ptr -> host)", cudaMemcpy(back.data(), p, kSize, cudaMemcpyDeviceToHost));
    line("    -> host got 0x44444444", back[0] == 0x44444444 && back.back() == 0x44444444);
    return 0;
}

// ---------------------------------------------------------------------------------

struct Test {
    const char* name;
    const char* title;
    std::function<int()> fn;
};

static std::vector<Test> tests() {
    return {
        {"info", "Devices", test_info},
        {"d3d12-to-cuda-extmem", "D3D12 shared buffer -> cudaImportExternalMemory", test_d3d12_to_cuda_extmem},
        {"d3d12-fence-to-cuda", "D3D12 shared fence -> cudaImportExternalSemaphore", test_d3d12_fence_to_cuda},
        {"cuda-to-d3d12-unknown", "CUDA VMM fd -> D3D12 OpenSharedHandle(IUnknown)", [] { return test_cuda_to_d3d12("unknown"); }},
        {"cuda-to-d3d12-resource", "CUDA VMM fd -> D3D12 OpenSharedHandle(ID3D12Resource)", [] { return test_cuda_to_d3d12("resource"); }},
        {"cuda-to-d3d12-heap", "CUDA VMM fd -> D3D12 OpenSharedHandle(ID3D12Heap)", [] { return test_cuda_to_d3d12("heap"); }},
        {"iunknown-identity", "Is Mesa's IUnknown -> ID3D12Resource cast valid here?", test_iunknown_identity},
        {"gpuva-api", "CUDA API calls (not kernels) on a D3D12 GPU VA", test_gpuva_api_calls},
        {"gpuva", "D3D12 buffer GPU VA used directly as a CUDA pointer", test_d3d12_gpuva_to_cuda},
        {"vmm-committed-8M", "D3D12 committed buffer (8 MiB) -> CUDA VMM import + map", [] { return test_d3d12_to_cuda_vmm(8u << 20, 0); }},
        {"vmm-committed-64M", "D3D12 committed buffer (64 MiB) -> CUDA VMM import + map", [] { return test_d3d12_to_cuda_vmm(64u << 20, 0); }},
        {"vmm-heap64K-8M", "D3D12 heap (64 KiB aligned, 8 MiB) -> CUDA VMM import + map", [] { return test_d3d12_to_cuda_vmm(8u << 20, 64u << 10); }},
        {"vmm-heap4M-8M", "D3D12 heap (4 MiB aligned, 8 MiB) -> CUDA VMM import + map", [] { return test_d3d12_to_cuda_vmm(8u << 20, 4u << 20); }},
    };
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (access("/dev/dxg", F_OK) != 0 || !load_d3d12()) {
        printf("skipped: not WSL2 with GPU support (no /dev/dxg or libd3d12.so)\n");
        return 77;
    }
    auto all = tests();
    if (argc > 1) {
        for (auto& t : all)
            if (t.name == std::string(argv[1])) return t.fn();
        fprintf(stderr, "unknown test %s\n", argv[1]);
        return 2;
    }
    for (auto& t : all) {
        printf("\n[%s] %s\n", t.name, t.title);
        pid_t pid = fork();
        if (pid == 0) {
            execl("/proc/self/exe", argv[0], t.name, (char*)nullptr);
            _exit(127);
        }
        int status = 0;
        waitpid(pid, &status, 0);
        if (WIFSIGNALED(status)) printf("    => CRASHED with signal %d (%s)\n", WTERMSIG(status), strsignal(WTERMSIG(status)));
    }
    return 0;
}
