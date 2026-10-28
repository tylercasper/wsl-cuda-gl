// Copyright 2026 Tyler Casper
// SPDX-License-Identifier: Apache-2.0

// What a process without a usable GL context gets: clear errors, never a crash. Runs
// anywhere, including CI machines without a GPU.

#include <cstdlib>
#include <cstring>

#include "harness.h"

static wslcg::GLProc null_loader(const char*) { return nullptr; }

static const unsigned char* no_context_get_string(unsigned) { return nullptr; }
static wslcg::GLProc fake_loader(const char* name) {
    // Every function "exists", but glGetString answers as if no context were current.
    if (!std::strcmp(name, "glGetString")) return reinterpret_cast<wslcg::GLProc>(&no_context_get_string);
    return reinterpret_cast<wslcg::GLProc>(&null_loader);
}

int main() {
    harness::section("check_support() reports, never throws");
    wslcg::SupportReport r = wslcg::check_support();
    std::printf("supported=%d reason: %s\n", r.supported, r.reason.c_str());
    CHECK(!r.supported);
    CHECK(r.reason.find("OpenGL context") != std::string::npos);

    harness::section("Buffer without a GL context throws");
    CHECK_THROWS(wslcg::Buffer(1024), "OpenGL context");
    CHECK_THROWS(wslcg::Buffer(0), "greater than 0");

    harness::section("set_gl_proc_loader");
    wslcg::set_gl_proc_loader(null_loader);
    CHECK_THROWS(wslcg::Buffer(1024), "glGetString is not available");
    wslcg::set_gl_proc_loader(fake_loader);
    CHECK_THROWS(wslcg::Buffer(1024), "no OpenGL context is current");
    wslcg::set_gl_proc_loader(nullptr); // back to the EGL/GLX default
    CHECK_THROWS(wslcg::Buffer(1024), "OpenGL context");

    harness::section("configure_mesa respects existing settings");
    setenv("GALLIUM_DRIVER", "llvmpipe", 1);
    unsetenv("MESA_D3D12_DEFAULT_ADAPTER_NAME");
    wslcg::configure_mesa();
    CHECK(std::strcmp(std::getenv("GALLIUM_DRIVER"), "llvmpipe") == 0);
    const char* adapter = std::getenv("MESA_D3D12_DEFAULT_ADAPTER_NAME");
    CHECK(adapter && *adapter);
    std::printf("MESA_D3D12_DEFAULT_ADAPTER_NAME=%s\n", adapter ? adapter : "(unset)");
    unsetenv("GALLIUM_DRIVER");
    wslcg::configure_mesa();
    CHECK(std::strcmp(std::getenv("GALLIUM_DRIVER"), "d3d12") == 0);
    return harness::finish();
}
