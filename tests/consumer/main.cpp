// Copyright 2026 Tyler Casper
// SPDX-License-Identifier: Apache-2.0

// Links and runs anywhere. There is no GL context here, so the library must report
// "unsupported" with a reason rather than crash or throw.

#include <wsl_cuda_gl.h>

#include <cstdio>

int main() {
    std::printf("wsl-cuda-gl %d.%d.%d\n", WSLCG_VERSION_MAJOR, WSLCG_VERSION_MINOR, WSLCG_VERSION_PATCH);
    wslcg::SupportReport r = wslcg::check_support();
    std::printf("supported: %s (%s)\n", r.supported ? "yes" : "no", r.reason.c_str());
    return !r.supported && !r.reason.empty() ? 0 : 1;
}
