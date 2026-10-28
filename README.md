# wsl-cuda-gl

[![CI](https://github.com/tylercasper/wsl-cuda-gl/actions/workflows/ci.yml/badge.svg)](https://github.com/tylercasper/wsl-cuda-gl/actions/workflows/ci.yml)

Share GPU buffers between CUDA and OpenGL on **WSL2**, where CUDA-OpenGL interop
(`cudaGraphicsGLRegisterBuffer` and friends) is not supported. CUDA renders a frame,
OpenGL draws it, and the frame never leaves GPU memory.

```cpp
#include <wsl_cuda_gl.h>

wslcg::configure_mesa();                      // before creating the GL context
// ... create a GL context (GLFW, EGL, ...) and make it current ...

wslcg::Buffer frame(width * height * 4);      // ~ cudaGraphicsGLRegisterBuffer
auto* px = static_cast<uchar4*>(frame.map()); // ~ cudaGraphicsMapResources + GetMappedPointer
render<<<grid, block, 0, stream>>>(px, width, height);
frame.unmap(stream);                          // ~ cudaGraphicsUnmapResources

glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, frame.gl_buffer()); // draw from it directly,
// or bind it as GL_PIXEL_UNPACK_BUFFER and glTexSubImage2D into a texture (a GPU-side copy)
```

## How it works

On WSL, OpenGL is provided by Mesa's d3d12 driver, which runs on Direct3D 12. NVIDIA's
OpenGL driver isn't available, so CUDA-GL interop isn't either. D3D12 is the common
ground:

```
          D3D12 (from the WSL GPU driver)
          allocates a buffer in VRAM and shares it
            │                              │
  its GPU virtual address        a shared handle (an fd on WSL)
            │                              │
            ▼                              ▼
   CUDA kernels write it         Mesa's GL imports it (GL_EXT_memory_object_fd)
```

* **GL side.** Mesa implements `glImportMemoryFdEXT` as `ID3D12Device::OpenSharedHandle`,
  so a D3D12 shared handle is something GL can import.
* **CUDA side.** CUDA on WSL refuses every official import route (`cudaImportExternalMemory`
  returns "operation not supported"). But WDDM gives each process one GPU virtual address
  space shared by all APIs, so the D3D12 buffer's GPU virtual address works directly as a
  pointer in CUDA kernels.

`map()` and `unmap()` provide the same ordering guarantees as CUDA-GL interop's map and
unmap. CUDA on WSL can't import GL or D3D12 fences, so they wait on the CPU: on a GL fence
and on a CUDA event. No data is copied.

`tools/probe_wsl_interop` tests each CUDA↔D3D12 sharing mechanism on your machine. On an
RTX 4090 with driver 615.65 and CUDA 13.2:

| Mechanism | Result |
|---|---|
| `cudaImportExternalMemory` of a D3D12 resource or heap (any handle type) | ❌ not supported |
| `cudaImportExternalSemaphore` of a D3D12 fence | ❌ not supported |
| CUDA VMM export (`cuMemExportToShareableHandle`) → D3D12 `OpenSharedHandle` | 💥 segfault in the D3D12 driver |
| D3D12 handle → `cuMemImportFromShareableHandle` + `cuMemMap` | ❌ import works, map refused |
| `cudaMemcpy` / `cudaMemset` on a D3D12 GPU address | ❌ invalid argument |
| **D3D12 GPU virtual address dereferenced in a CUDA kernel** | ✅ |

## Requirements

* WSL2 with GPU support and an NVIDIA GPU (tested: RTX 4090, Windows driver 616.56, CUDA 13.2)
* Mesa with the d3d12 Gallium driver (stock Ubuntu 22.04/24.04 has it)
* CMake ≥ 3.24, a C++17 compiler (GCC or Clang), the CUDA Toolkit headers
* [DirectX-Headers](https://github.com/microsoft/DirectX-Headers) at build time only:
  `apt install directx-headers-dev`, or downloaded automatically

The library itself needs no nvcc, no GL headers and links no GPU libraries. It loads CUDA
(`libcuda.so.1`), D3D12 and GL at run time. On a machine where they're missing it builds,
links and reports why it can't work rather than failing to start.

## Using it in your project

**FetchContent** (simplest):

```cmake
include(FetchContent)
FetchContent_Declare(wsl_cuda_gl
  GIT_REPOSITORY https://github.com/tylercasper/wsl-cuda-gl.git
  GIT_TAG main)
FetchContent_MakeAvailable(wsl_cuda_gl)
target_link_libraries(my_app PRIVATE wsl_cuda_gl::wsl_cuda_gl)
```

**Installed package:**

```bash
cmake -S wsl-cuda-gl -B build -DWSLCG_BUILD_TESTS=OFF -DWSLCG_BUILD_EXAMPLES=OFF -DWSLCG_BUILD_TOOLS=OFF
cmake --build build
cmake --install build --prefix ~/.local
```
```cmake
find_package(wsl_cuda_gl 0.1 CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE wsl_cuda_gl::wsl_cuda_gl)
```

`add_subdirectory(wsl-cuda-gl)` works too. As a subproject it builds only the library.

## API

Everything is in [`include/wsl_cuda_gl.h`](include/wsl_cuda_gl.h):

| | |
|---|---|
| `wslcg::configure_mesa(cuda_device = 0)` | Before creating the GL context: points Mesa at its d3d12 driver on CUDA's GPU (`GALLIUM_DRIVER`, `MESA_D3D12_DEFAULT_ADAPTER_NAME`; existing values are kept). Otherwise Mesa may use llvmpipe (the CPU) or another GPU. |
| `wslcg::set_gl_proc_loader(fn)` | Optional. How to find GL functions, e.g. `glfwGetProcAddress`. Defaults to EGL or GLX, whichever has a context current. |
| `wslcg::check_support()` | End-to-end check (allocates and self-tests a small buffer). Returns a `SupportReport` with a `reason`; never throws. |
| `wslcg::Buffer(bytes)` | Shared buffer. `gl_buffer()` is its GL name. |
| `map()` | Waits for GL work issued so far, then returns the CUDA pointer. |
| `unmap(stream)` | Waits for CUDA work issued on `stream`, after which GL may use the buffer. |
| `copy_from_device` / `copy_to_device` | While mapped: copies to/from ordinary CUDA memory. Use these instead of `cudaMemcpy`. |

Errors throw `wslcg::Error`, whose message says what to do.

**Rules**

* The pointer from `map()` works **only inside CUDA kernels**. `cudaMemcpy`, `cudaMemset`
  and the driver-API equivalents reject it, and `cudaMemcpyDefault` crashes. Use
  `copy_from_device` / `copy_to_device`.
* Create, map, unmap and destroy a `Buffer` with its GL context current.
* A `Buffer` shares memory with the CUDA context current when it was created. If none was
  current, it uses device 0's primary context, which is the one the CUDA runtime uses.
* To overlap CUDA and GL work, use two buffers and alternate between them
  ([`examples/pipelined.cpp`](examples/pipelined.cpp)).

## Examples

```bash
sudo apt install directx-headers-dev libglfw3-dev libgl-dev libegl-dev ninja-build
cmake -S . -B build -G Ninja -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build
./build/examples/pipelined        # opens a window; close it to exit
```

| Example | |
|---|---|
| `host_copy` | Baseline without the library: CUDA → `cudaMemcpy` to host → `glTexSubImage2D` → draw |
| `zero_copy` | One `wslcg::Buffer`; `--display ssbo` draws from it directly, `--display texture` copies it into a texture on the GPU |
| `pipelined` | Two buffers: CUDA renders frame N while GL presents N−1 |

Flags: `--size WxH`, `--frames N`, `--verify` (compares every displayed pixel with what
CUDA rendered), `--hidden`, `--cuda-work-us N` (simulated CUDA workload), `--dump out.ppm`.

Frame times on an RTX 4090, averaged over 300 frames (`--hidden`):

| | 1080p | 4K | 1080p + 3 ms CUDA work |
|---|---|---|---|
| `host_copy` | 3.08 ms | 8.48 ms | 5.98 ms |
| `zero_copy --display ssbo` | 1.54 ms | 3.82 ms | 4.52 ms |
| `zero_copy --display texture` | 1.64 ms | 3.87 ms | 4.59 ms |
| `pipelined --display ssbo` | 1.63 ms | 3.74 ms | 3.18 ms |

At 4K the host round trip costs about 4.7 ms per frame, and the library removes it. The
remaining time is mostly WSLg's presentation, which copies every finished frame to system
memory for the Windows compositor in any GL app. That copy is separate from the CUDA → GL
path.

## Tests

```bash
WSLCG_REQUIRE_GPU_TESTS=1 ctest --test-dir build --output-on-failure
```

| Label | What | Where it runs |
|---|---|---|
| `unit` | Mesa version parsing; vtable-slot derivation cross-checked against the C vtable layout; errors without a GL context or with a non-d3d12 one (llvmpipe); the public header compiles alone, after GL/EGL/GLFW/CUDA headers, as C++17 and C++20 | anywhere |
| `gpu` | CUDA→GL and GL→CUDA bit-exact at sizes from 1×1 to 4K; PBO→texture; copy helpers at every alignment; double-buffered pipelining; map/unmap ordering (see below); the map/unmap state machine and move semantics; no leaked fds, GL objects or VRAM; `check_support`; the Mesa workaround is scoped | WSL2 + NVIDIA GPU; skipped elsewhere |
| `consumer` | Builds a separate C++20 project against the library via `find_package` (after `cmake --install`), `add_subdirectory` and `FetchContent` | anywhere |

GPU tests skip themselves (exit 77) where the hardware is missing.
`WSLCG_REQUIRE_GPU_TESTS=1` turns that skip into a failure, so they can't skip silently on
a machine that should run them. The ordering test, `test_sync`, was verified to fail when
either `map()`'s or `unmap()`'s wait is removed from the library.

**CI** ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) builds everything with
`-Wall -Wextra -Werror` and runs the tests on GitHub's hosted runners. The matrix covers
GCC and Clang, CUDA 12.6 and 13.0, Ubuntu 22.04 and 24.04, static and shared builds, and
system and downloaded DirectX-Headers. Those runners have no GPU, so CI runs the `unit`
and `consumer` tests; run the `gpu` tests on a WSL2 machine before releasing.

## Two pitfalls this library works around

1. **Mesa < 26.0.3 crashes on every `GL_EXT_memory_object` import** on the d3d12 driver,
   and Ubuntu 24.04 ships 25.2.8. Mesa's `d3d12_memobj_create_from_handle` opens the
   shared handle as `IUnknown` and C-casts the result to `ID3D12Resource*`, but in the
   D3D12 runtime those pointers differ by 0x30 bytes. Upstream fixed this with
   `QueryInterface`. For affected versions, `Buffer` patches `ID3D12Device::OpenSharedHandle`
   in the device vtable for the duration of the import only, so that it answers `IUnknown`
   requests with the `ID3D12Resource` interface Mesa's cast expects. Fixed Mesa versions are
   left alone (that path is untested).
2. **A process that loads Mesa must not link `libd3d12.so`.** `libd3d12.so` and
   `libd3d12core.so` both export `D3D12GetInterface`. With `libd3d12.so` in the global
   symbol scope, its `D3D12GetInterface` resolves to itself and recurses until the stack
   overflows. The library loads D3D12 with `dlopen(RTLD_LOCAL)`, as Mesa does.

## Limitations

* **Relies on undocumented behaviour.** Neither NVIDIA nor Microsoft documents that a
  CUDA kernel can dereference a D3D12 GPU virtual address. Each process verifies it once,
  on its first `Buffer`, and throws a clear `wslcg::Error` if a driver update breaks it.
* **Buffers, not textures.** A D3D12 texture's tiled layout is driver-private, so CUDA
  writes linear memory. A GL texture costs one GPU copy (bind the buffer as
  `GL_PIXEL_UNPACK_BUFFER`).
* **Synchronization waits on the CPU** (no GPU-side semaphores), as explained above.
* **CUDA, D3D12 and GL must be on the same GPU.** They're matched by device name, so two
  identical GPUs can't be told apart.
* **Residency.** D3D12 keeps a buffer resident while it exists, and CUDA plays no part in
  D3D12's residency management. If VRAM is oversubscribed and the OS pages a buffer out, a
  kernel touching it could fault.
* **Memory reporting.** `cudaMemGetInfo` reports a `Buffer` as using twice its size,
  because GL's opened handle is counted again. The GPU holds one copy, which `nvidia-smi`
  confirms.

## Repository layout

```
include/wsl_cuda_gl.h   the public API
src/                    the library (GL, CUDA and D3D12 loaded at run time)
cmake/                  DirectX-Headers lookup, package config
examples/               GLFW demos
tools/                  probe_wsl_interop diagnostic
support/                test-image kernels shared by examples, tests and the probe
tests/                  unit/, gpu/, consumer/ (see above)
scripts/                PTX regeneration, CI dependency installation
```

`src/kernels_ptx.h` is generated from `src/kernels.cu` by `scripts/generate_ptx.sh`. The
library launches those kernels (its self-test and copy helpers) through the CUDA driver
API, so it has no nvcc build step.

## License

Copyright 2026 Tyler Casper. Licensed under the [Apache License 2.0](LICENSE).
Redistributions, including derivative works, must carry the attribution in
[NOTICE](NOTICE).
