// Copyright 2026 Tyler Casper
// SPDX-License-Identifier: Apache-2.0

#pragma once
// D3D12 is the common currency between CUDA and Mesa's GL on WSL:
//
//  * Mesa's GL runs on D3D12, and its GL_EXT_memory_object_fd import is
//    ID3D12Device::OpenSharedHandle(fd). So a D3D12 shared handle is what GL can import.
//  * CUDA on WSL refuses to import D3D12 handles (cudaImportExternalMemory ->
//    "operation not supported"), but WDDM gives each process one GPU virtual address
//    space shared by every API, so a D3D12 buffer's GPU virtual address is directly
//    usable as a pointer in CUDA kernels.
//
// This header exposes no D3D12/Windows types (they clash with X11/GL headers); the
// implementation lives in d3d12_bridge.cpp.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace wslcg::detail {

class D3D12SharedBuffer {
public:
    // A buffer of `size` bytes in a D3D12 DEFAULT (device-local) heap, on the adapter
    // whose description equals `adapter_name` (case-insensitively), or failing that
    // contains it. Throws Error.
    static std::unique_ptr<D3D12SharedBuffer> create(size_t size, const std::string& adapter_name);
    ~D3D12SharedBuffer();

    size_t size() const;
    const std::string& adapter_name() const;

    // The buffer's GPU virtual address, usable as a pointer in CUDA kernels.
    uint64_t gpu_address() const;

    // A new fd for a D3D12 shared handle to the buffer. The caller owns it;
    // glImportMemoryFdEXT takes ownership on import.
    int export_fd() const;

    // What the device's vtable currently holds for OpenSharedHandle (for tests of
    // MesaMemobjImportFix).
    void* open_shared_handle_vtable_entry() const;

private:
    friend class MesaMemobjImportFix;
    struct Impl;
    explicit D3D12SharedBuffer(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

// Works around a Mesa bug, fixed in Mesa 26.0.3, that breaks every GL_EXT_memory_object
// import on the d3d12 driver (Ubuntu 24.04 ships 25.2.8).
//
// Mesa's d3d12_memobj_create_from_handle() opens the shared handle as IUnknown and
// C-casts the result to ID3D12Resource*. In this D3D12 runtime the IUnknown identity
// pointer is a different address from the ID3D12Resource interface pointer, so Mesa
// then calls resource methods through the wrong vtable and crashes.
//
// While an instance is alive, ID3D12Device::OpenSharedHandle (patched in the device
// vtable, which Mesa's device shares) answers IUnknown requests with the
// ID3D12Resource interface instead. That is a valid IUnknown pointer, and exactly what
// Mesa's cast expects. Keep it alive across glImportMemoryFdEXT followed by glFinish.
// Instances serialize on a global mutex. It does nothing if `gl_version` names a
// fixed Mesa.
class MesaMemobjImportFix {
public:
    MesaMemobjImportFix(const D3D12SharedBuffer& any_buffer_on_the_device, const char* gl_version);
    ~MesaMemobjImportFix();
    MesaMemobjImportFix(const MesaMemobjImportFix&) = delete;
    MesaMemobjImportFix& operator=(const MesaMemobjImportFix&) = delete;

    bool active() const { return slot_ != nullptr; }
    // OpenSharedHandle(IUnknown) calls redirected so far; 0 after an import means Mesa
    // did not go through the patched vtable.
    int redirected_calls() const;

private:
    void** slot_ = nullptr;
    void* original_ = nullptr;
};

// Index of ID3D12Device::OpenSharedHandle in the device vtable.
size_t open_shared_handle_slot();

} // namespace wslcg::detail
