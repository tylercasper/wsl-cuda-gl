// wsl-cuda-gl: share GPU buffers between CUDA and OpenGL on WSL2, where CUDA-OpenGL
// interop (cudaGraphicsGLRegisterBuffer and friends) is not supported.
//
// https://github.com/tylercasper/wsl-cuda-gl
// Copyright 2026 Tyler Casper. Licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
//
// Usage mirrors CUDA-GL buffer interop:
//
//   wslcg::configure_mesa();                    // before creating the GL context
//   ... create a GL context (GLFW, EGL, ...) and make it current ...
//
//   wslcg::Buffer buf(width * height * 4);      // ~ cudaGraphicsGLRegisterBuffer
//   auto* px = static_cast<uchar4*>(buf.map()); // ~ cudaGraphicsMapResources + GetMappedPointer
//   my_kernel<<<grid, block, 0, stream>>>(px, ...);
//   buf.unmap(stream);                          // ~ cudaGraphicsUnmapResources
//   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buf.gl_buffer());  // or bind as a PBO
//
// The memory is a D3D12 allocation in VRAM that Mesa's OpenGL imports and CUDA
// addresses directly, so frame data never passes through host memory. README.md
// explains how.
//
// Rules:
//  * The pointer returned by map() works only inside CUDA kernels. cudaMemcpy,
//    cudaMemset and the driver-API equivalents reject it (cudaMemcpyDefault even
//    crashes). Use copy_from_device() / copy_to_device() instead.
//  * Create, map, unmap and destroy a Buffer with its GL context current on the
//    calling thread.
//  * A Buffer shares memory with the CUDA context that was current when it was
//    created (or, if none was, the primary context of device 0, as the CUDA runtime
//    would choose). Streams passed to it must belong to that context.

#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>

#define WSLCG_VERSION_MAJOR 0
#define WSLCG_VERSION_MINOR 1
#define WSLCG_VERSION_PATCH 0

// cudaStream_t and CUstream are both `CUstream_st*`, so either can be passed where
// this header says CUstream_st*, without including any CUDA header here.
struct CUstream_st;

namespace wslcg {

namespace detail {
struct BufferState;
}

// Every failure is reported by throwing this.
class Error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Points Mesa at its D3D12 driver on the GPU of CUDA device `cuda_device`: sets
// GALLIUM_DRIVER=d3d12 and MESA_D3D12_DEFAULT_ADAPTER_NAME=<that GPU's name>, leaving
// either variable alone if it is already set. Call it before creating the GL context
// (or the EGL display / GLX connection). Without it Mesa may fall back to llvmpipe (CPU
// rendering) or pick another GPU such as an iGPU, and sharing is impossible.
void configure_mesa(int cuda_device = 0);

// How the library looks up the GL functions it calls. Optional: by default it uses
// eglGetProcAddress or glXGetProcAddressARB, whichever API has a context current.
// glfwGetProcAddress and eglGetProcAddress can be passed as they are;
// glXGetProcAddressARB needs a cast (its parameter is const GLubyte*).
using GLProc = void (*)();
using GLProcLoader = GLProc (*)(const char* name);
void set_gl_proc_loader(GLProcLoader loader);

struct SupportReport {
    bool supported = false;
    std::string reason;      // why not, when !supported
    std::string gl_renderer; // of the current GL context
    std::string gl_version;
    std::string cuda_device;
    bool mesa_workaround = false; // whether the Mesa < 26.0.3 import fix is applied
};

// Checks, end to end, whether Buffer works in this process: needs a current GL
// context, creates a small Buffer and runs the self-test. Never throws.
SupportReport check_support();

class Buffer {
public:
    // Allocates `bytes` of device memory shared by the current GL context and CUDA.
    // Initial contents are unspecified. Throws Error.
    explicit Buffer(std::size_t bytes);
    ~Buffer();
    Buffer(Buffer&&) noexcept;
    Buffer& operator=(Buffer&&) noexcept;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    // The GL buffer object (a GLuint) backed by the shared memory. 0 if moved from.
    unsigned int gl_buffer() const noexcept;
    std::size_t size() const noexcept;
    bool is_mapped() const noexcept;

    // Hands the buffer to CUDA. Returns once all GL commands issued so far have
    // finished, so CUDA may write the buffer from any stream. The returned device
    // pointer is valid in kernels until unmap().
    void* map();

    // Hands the buffer back to GL. Returns once all CUDA work issued on `stream`
    // (the legacy default stream if null) has finished, so GL may use the buffer.
    void unmap(CUstream_st* stream = nullptr);

    // While mapped: asynchronous copies on `stream` between ordinary CUDA device
    // memory and the buffer, at byte `offset` into the buffer. Use these instead of
    // cudaMemcpy, which rejects the mapped pointer.
    void copy_from_device(const void* src, std::size_t bytes, std::size_t offset = 0,
                          CUstream_st* stream = nullptr);
    void copy_to_device(void* dst, std::size_t bytes, std::size_t offset = 0,
                        CUstream_st* stream = nullptr) const;

private:
    std::unique_ptr<detail::BufferState> impl_;
};

} // namespace wslcg
