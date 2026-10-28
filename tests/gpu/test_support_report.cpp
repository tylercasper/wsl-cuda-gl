// Copyright 2026 Tyler Casper
// SPDX-License-Identifier: Apache-2.0

// check_support() on a working machine: supported, with the details filled in.

#include <cstring>

#include "gl_env.h"
#include "harness.h"
#include "mesa_version.h"

int main() {
    HeadlessGL gl;
    require_gpu_environment(gl);
    wslcg::SupportReport r = wslcg::check_support();
    std::printf("supported=%d renderer=\"%s\" cuda=\"%s\" mesa_workaround=%d reason=\"%s\"\n", r.supported,
                r.gl_renderer.c_str(), r.cuda_device.c_str(), r.mesa_workaround, r.reason.c_str());
    CHECK(r.supported);
    CHECK(r.reason.empty());
    CHECK(r.gl_renderer.find("D3D12") != std::string::npos);
    CHECK(!r.cuda_device.empty() && r.gl_renderer.find(r.cuda_device) != std::string::npos);
    CHECK(r.gl_version == reinterpret_cast<const char*>(glGetString(GL_VERSION)));
    CHECK(r.mesa_workaround == wslcg::detail::mesa_has_memobj_import_bug(r.gl_version.c_str()));
    return harness::finish();
}
