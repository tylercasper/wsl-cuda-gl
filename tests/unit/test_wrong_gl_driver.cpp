// Copyright 2026 Tyler Casper
// SPDX-License-Identifier: Apache-2.0

// A real GL context that isn't Mesa's d3d12 driver (llvmpipe, Mesa's CPU renderer)
// must be rejected with an explanation. Runs wherever Mesa's EGL is installed,
// including GPU-less CI machines.

#include <cstdlib>

#include "gl_env.h"
#include "harness.h"

int main() {
    setenv("GALLIUM_DRIVER", "llvmpipe", 1);
    setenv("LIBGL_ALWAYS_SOFTWARE", "1", 1);
    HeadlessGL gl;
    std::string why;
    if (!gl.create(&why)) harness::skip(why);
    std::printf("GL renderer: %s\n", gl.renderer().c_str());
    if (gl.renderer().find("D3D12") != std::string::npos) harness::skip("could not get a non-d3d12 context");

    harness::section("Buffer rejects a non-d3d12 context");
    CHECK_THROWS(wslcg::Buffer(1024), "not Mesa's d3d12 driver");

    harness::section("check_support explains why");
    wslcg::SupportReport r = wslcg::check_support();
    std::printf("supported=%d reason: %s\n", r.supported, r.reason.c_str());
    CHECK(!r.supported);
    CHECK(r.gl_renderer == gl.renderer());
    CHECK(!r.gl_version.empty());
    CHECK(r.reason.find("d3d12") != std::string::npos);
    return harness::finish();
}
