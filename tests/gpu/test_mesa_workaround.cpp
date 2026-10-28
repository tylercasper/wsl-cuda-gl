// Copyright 2026 Tyler Casper
// SPDX-License-Identifier: Apache-2.0

// MesaMemobjImportFix patches a vtable shared with Mesa's D3D12 device, so it must be
// scoped exactly: patched while alive, and the original D3D12 function restored after,
// including after creating a Buffer.

#include <dlfcn.h>

#include "d3d12_bridge.h"
#include "gl_env.h"
#include "harness.h"
#include "mesa_version.h"

static std::string library_of(void* fn) {
    Dl_info info = {};
    return dladdr(fn, &info) && info.dli_fname ? info.dli_fname : "?";
}

int main() {
    HeadlessGL gl;
    require_gpu_environment(gl);
    const char* version = reinterpret_cast<const char*>(glGetString(GL_VERSION));
    cudaDeviceProp prop;
    CUDA_REQUIRE(cudaGetDeviceProperties(&prop, 0));
    auto shared = wslcg::detail::D3D12SharedBuffer::create(4096, prop.name);

    void* original = shared->open_shared_handle_vtable_entry();
    std::printf("OpenSharedHandle lives in %s\n", library_of(original).c_str());
    CHECK(library_of(original).find("d3d12") != std::string::npos);

    harness::section("patched only while the fix is alive");
    {
        wslcg::detail::MesaMemobjImportFix fix(*shared, version);
        std::printf("this Mesa %s the workaround\n", fix.active() ? "needs" : "does not need");
        CHECK(fix.active() == wslcg::detail::mesa_has_memobj_import_bug(version));
        if (fix.active()) CHECK(shared->open_shared_handle_vtable_entry() != original);
    }
    CHECK(shared->open_shared_handle_vtable_entry() == original);

    harness::section("restored after Buffer construction");
    {
        wslcg::Buffer buf(1u << 20);
        CHECK(shared->open_shared_handle_vtable_entry() == original);
    }
    CHECK(shared->open_shared_handle_vtable_entry() == original);
    return harness::finish();
}
