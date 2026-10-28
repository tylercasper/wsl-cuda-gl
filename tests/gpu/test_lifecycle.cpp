// Copyright 2026 Tyler Casper
// SPDX-License-Identifier: Apache-2.0

// Creating and destroying many buffers leaks nothing: file descriptors (each import
// passes one to GL), GL objects, or GPU memory.
//
// Note: on WSL cudaMemGetInfo reports a Buffer as using twice its size, because GL's
// opened handle is counted against the process budget again. The GPU holds one copy
// (nvidia-smi's device-wide figure rises by the buffer size only).

#include <dirent.h>
#include <vector>

#include "gl_env.h"
#include "harness.h"

static int open_fds() {
    int n = 0;
    if (DIR* d = opendir("/proc/self/fd")) {
        while (readdir(d)) ++n;
        closedir(d);
    }
    return n;
}

static size_t free_vram() {
    size_t free_b = 0, total = 0;
    CUDA_REQUIRE(cudaMemGetInfo(&free_b, &total));
    return free_b;
}

int main() {
    HeadlessGL gl;
    require_gpu_environment(gl);
    { wslcg::Buffer warm_up(4096); } // first use loads libraries and opens devices

    harness::section("file descriptors and GL objects");
    const int fds_before = open_fds();
    std::vector<unsigned> names;
    for (int i = 0; i < 100; ++i) {
        wslcg::Buffer b(1u << 20);
        names.push_back(b.gl_buffer());
    }
    const int fds_after = open_fds();
    std::printf("open fds: %d before, %d after 100 buffers\n", fds_before, fds_after);
    CHECK(fds_after == fds_before);
    size_t live = 0;
    for (unsigned n : names) live += glIsBuffer(n);
    CHECK(live == 0);

    harness::section("GPU memory");
    const size_t chunk = 64u << 20, count = 16;
    const size_t before = free_vram();
    {
        std::vector<wslcg::Buffer> held;
        for (size_t i = 0; i < count; ++i) held.emplace_back(chunk);
        std::printf("free VRAM: %zu MiB before, %zu MiB with %zu MiB allocated\n", before >> 20, free_vram() >> 20,
                    (chunk * count) >> 20);
    }
    const size_t after = free_vram();
    std::printf("free VRAM after release: %zu MiB\n", after >> 20);
    CHECK(after + (64u << 20) >= before); // all returned, within noise from other processes
    return harness::finish();
}
