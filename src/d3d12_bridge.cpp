// Copyright 2026 Tyler Casper
// SPDX-License-Identifier: Apache-2.0

#include "d3d12_bridge.h"

#include <wsl/winadapter.h> // must precede the D3D12 headers on Linux
#include <directx/d3d12.h>
#include <directx/dxcore.h>
#include <dxguids/dxguids.h>
#include <wrl/client.h>

#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <mutex>
#include <strings.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

#include "mesa_version.h"
#include "wsl_cuda_gl.h"

using Microsoft::WRL::ComPtr;

#ifndef GENERIC_ALL
#define GENERIC_ALL 0x10000000L
#endif

namespace wslcg::detail {
namespace {

[[noreturn]] void fail(const char* what, HRESULT hr) {
    char buf[256];
    std::snprintf(buf, sizeof buf, "wsl-cuda-gl: D3D12 %s failed (hr=0x%08x)", what, (unsigned)hr);
    throw Error(buf);
}

// Defined here rather than taken from libDirectX-Guids, so DirectX-Headers is a
// header-only, build-time dependency.
constexpr GUID kDxcoreD3D12Graphics = {0x0c9ece4d, 0x2f6e, 0x4f01, {0x8c, 0x96, 0xe8, 0x9e, 0x33, 0x1b, 0x47, 0xb1}};

// libd3d12.so and libdxcore.so (from the WSL GPU driver, /usr/lib/wsl/lib) are
// loaded with RTLD_LOCAL, as Mesa does, rather than linked. Both libd3d12.so and
// libd3d12core.so export D3D12GetInterface; with libd3d12.so in the global symbol
// scope its D3D12GetInterface resolves to itself and recurses until the stack
// overflows the moment Mesa initializes.
struct Entrypoints {
    PFN_D3D12_CREATE_DEVICE D3D12CreateDevice = nullptr;
    HRESULT (*DXCoreCreateAdapterFactory)(REFIID, void**) = nullptr;
};

const Entrypoints& entrypoints() {
    static Entrypoints e;
    static std::string error;
    static std::once_flag once;
    std::call_once(once, [] {
        void* d3d12 = dlopen("libd3d12.so", RTLD_NOW | RTLD_LOCAL);
        void* dxcore = dlopen("libdxcore.so", RTLD_NOW | RTLD_LOCAL);
        if (!d3d12 || !dxcore) {
            error = "wsl-cuda-gl: cannot load D3D12 (libd3d12.so / libdxcore.so); this only works on WSL2 with "
                    "GPU support";
            return;
        }
        e.D3D12CreateDevice = reinterpret_cast<PFN_D3D12_CREATE_DEVICE>(dlsym(d3d12, "D3D12CreateDevice"));
        e.DXCoreCreateAdapterFactory =
            reinterpret_cast<decltype(e.DXCoreCreateAdapterFactory)>(dlsym(dxcore, "DXCoreCreateAdapterFactory"));
        if (!e.D3D12CreateDevice || !e.DXCoreCreateAdapterFactory) error = "wsl-cuda-gl: D3D12 entry points missing";
    });
    if (!error.empty()) throw Error(error);
    return e;
}

} // namespace

struct D3D12SharedBuffer::Impl {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12Resource> resource;
    std::string adapter;
    size_t size = 0;
};

D3D12SharedBuffer::D3D12SharedBuffer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
D3D12SharedBuffer::~D3D12SharedBuffer() = default;

namespace {

struct Device {
    ComPtr<ID3D12Device> device;
    std::string adapter;
};

// The D3D12 device for `adapter_name`, created once per process: adapter enumeration
// and D3D12CreateDevice cost ~50 ms, far more than the buffer itself. (D3D12 devices
// are per-adapter singletons anyway; this one is the same device Mesa uses when it
// picked the same adapter.)
Device device_for(const std::string& adapter_name) {
    static std::mutex mutex;
    static std::vector<std::pair<std::string, Device>> cache;
    std::lock_guard<std::mutex> lock(mutex);
    for (auto& entry : cache)
        if (entry.first == adapter_name) return entry.second;

    const Entrypoints& d3d = entrypoints();
    ComPtr<IDXCoreAdapterFactory> factory;
    HRESULT hr = d3d.DXCoreCreateAdapterFactory(IID_PPV_ARGS(&factory));
    if (FAILED(hr)) fail("DXCoreCreateAdapterFactory", hr);
    ComPtr<IDXCoreAdapterList> list;
    hr = factory->CreateAdapterList(1, &kDxcoreD3D12Graphics, IID_PPV_ARGS(&list));
    if (FAILED(hr)) fail("CreateAdapterList", hr);

    // Prefer an exact name match; fall back to a substring match, as Mesa's
    // MESA_D3D12_DEFAULT_ADAPTER_NAME does.
    ComPtr<IDXCoreAdapter> chosen;
    std::string chosen_desc, all;
    for (uint32_t i = 0; i < list->GetAdapterCount(); ++i) {
        ComPtr<IDXCoreAdapter> adapter;
        if (FAILED(list->GetAdapter(i, IID_PPV_ARGS(&adapter)))) continue;
        char desc[256] = {};
        adapter->GetProperty(DXCoreAdapterProperty::DriverDescription, sizeof desc, desc);
        all += std::string(all.empty() ? "" : ", ") + desc;
        bool exact = !strcasecmp(desc, adapter_name.c_str());
        if (exact || (!chosen && strcasestr(desc, adapter_name.c_str()))) {
            chosen = adapter;
            chosen_desc = desc;
            if (exact) break;
        }
    }
    if (!chosen)
        throw Error("wsl-cuda-gl: no D3D12 adapter matches \"" + adapter_name + "\" (adapters: " + all + ")");
    Device d;
    d.adapter = chosen_desc;
    hr = d3d.D3D12CreateDevice(chosen.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&d.device));
    if (FAILED(hr)) fail("D3D12CreateDevice", hr);
    cache.emplace_back(adapter_name, std::move(d));
    return cache.back().second;
}

} // namespace

std::unique_ptr<D3D12SharedBuffer> D3D12SharedBuffer::create(size_t size, const std::string& adapter_name) {
    Device dev = device_for(adapter_name);
    auto impl = std::make_unique<Impl>();
    impl->size = size;
    impl->device = dev.device;
    impl->adapter = dev.adapter;

    D3D12_HEAP_PROPERTIES heap = {};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT; // device-local VRAM
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS; // so GL can also write it as an SSBO
    // A committed resource, not a heap: Mesa's fd import treats the handle as an
    // ID3D12Resource and has no way to be told otherwise.
    HRESULT hr = impl->device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_SHARED, &desc, D3D12_RESOURCE_STATE_COMMON,
                                               nullptr, IID_PPV_ARGS(&impl->resource));
    if (FAILED(hr)) fail("CreateCommittedResource", hr);

    return std::unique_ptr<D3D12SharedBuffer>(new D3D12SharedBuffer(std::move(impl)));
}

size_t D3D12SharedBuffer::size() const { return impl_->size; }
const std::string& D3D12SharedBuffer::adapter_name() const { return impl_->adapter; }
uint64_t D3D12SharedBuffer::gpu_address() const { return impl_->resource->GetGPUVirtualAddress(); }

int D3D12SharedBuffer::export_fd() const {
    // On WSL an NT "HANDLE" from CreateSharedHandle is a file descriptor
    // (anon_inode:dxgresource).
    HANDLE h = nullptr;
    HRESULT hr = impl_->device->CreateSharedHandle(impl_->resource.Get(), nullptr, GENERIC_ALL, nullptr, &h);
    if (FAILED(hr)) fail("CreateSharedHandle", hr);
    return (int)(intptr_t)h;
}

void* D3D12SharedBuffer::open_shared_handle_vtable_entry() const {
    void** vtable = *reinterpret_cast<void***>(impl_->device.Get());
    return vtable[open_shared_handle_slot()];
}

// ---------------------------------------------------------------------------------
// MesaMemobjImportFix

namespace {

using OpenSharedHandleFn = HRESULT (*)(ID3D12Device* self, HANDLE, REFIID, void**);
std::mutex g_fix_mutex; // one fix at a time: the patch is process-wide
OpenSharedHandleFn g_original_open_shared_handle = nullptr;
int g_redirected_calls = 0;

// STDMETHODCALLTYPE is empty on Linux, so a free function taking `self` first has the
// same ABI as the virtual method.
HRESULT open_shared_handle_as_resource(ID3D12Device* self, HANDLE handle, REFIID riid, void** out) {
    if (riid == __uuidof(IUnknown)) {
        ID3D12Resource* res = nullptr;
        HRESULT hr = g_original_open_shared_handle(self, handle, __uuidof(ID3D12Resource), (void**)&res);
        if (SUCCEEDED(hr)) {
            ++g_redirected_calls;
            *out = static_cast<IUnknown*>(res);
            return hr;
        }
    }
    return g_original_open_shared_handle(self, handle, riid, out);
}

// Protection of the page holding `addr` (vtables usually live in RELRO: read-only
// after relocation).
int page_protection(const void* addr) {
    std::ifstream maps("/proc/self/maps");
    std::string line;
    auto a = reinterpret_cast<uintptr_t>(addr);
    while (std::getline(maps, line)) {
        unsigned long lo, hi;
        char perms[5] = {};
        if (std::sscanf(line.c_str(), "%lx-%lx %4s", &lo, &hi, perms) == 3 && a >= lo && a < hi)
            return (perms[0] == 'r' ? PROT_READ : 0) | (perms[1] == 'w' ? PROT_WRITE : 0) |
                   (perms[2] == 'x' ? PROT_EXEC : 0);
    }
    return PROT_READ;
}

void write_slot(void** slot, void* value) {
    long page = sysconf(_SC_PAGESIZE);
    void* base = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(slot) & ~static_cast<uintptr_t>(page - 1));
    int prot = page_protection(slot);
    if (mprotect(base, page, prot | PROT_WRITE) != 0) throw Error("wsl-cuda-gl: mprotect failed on the D3D12 vtable");
    __atomic_store_n(slot, value, __ATOMIC_SEQ_CST);
    mprotect(base, page, prot);
}

} // namespace

size_t open_shared_handle_slot() {
    // Taken from the header rather than hard-coded: under the Itanium C++ ABI
    // (GCC/Clang on Linux) a pointer to a virtual member function holds 1 + the byte
    // offset of its vtable slot. A unit test cross-checks this against the C-style
    // vtable layout the same header declares.
    auto pmf = static_cast<HRESULT (ID3D12Device::*)(HANDLE, REFIID, void**)>(&ID3D12Device::OpenSharedHandle);
    struct {
        uintptr_t ptr;
        ptrdiff_t adj;
    } raw;
    static_assert(sizeof pmf == sizeof raw, "unexpected pointer-to-member ABI");
    std::memcpy(&raw, &pmf, sizeof raw);
    return (raw.ptr - 1) / sizeof(void*);
}

MesaMemobjImportFix::MesaMemobjImportFix(const D3D12SharedBuffer& buffer, const char* gl_version) {
    if (!mesa_has_memobj_import_bug(gl_version)) return;
    g_fix_mutex.lock();
    void** vtable = *reinterpret_cast<void***>(buffer.impl_->device.Get());
    slot_ = &vtable[open_shared_handle_slot()];
    original_ = *slot_;
    g_original_open_shared_handle = reinterpret_cast<OpenSharedHandleFn>(original_);
    g_redirected_calls = 0;
    try {
        write_slot(slot_, reinterpret_cast<void*>(&open_shared_handle_as_resource));
    } catch (...) {
        g_fix_mutex.unlock();
        throw;
    }
}

MesaMemobjImportFix::~MesaMemobjImportFix() {
    if (!slot_) return;
    try {
        write_slot(slot_, original_);
    } catch (...) {
    }
    g_fix_mutex.unlock();
}

int MesaMemobjImportFix::redirected_calls() const { return g_redirected_calls; }

} // namespace wslcg::detail
