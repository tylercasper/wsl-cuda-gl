// Copyright 2026 Tyler Casper
// SPDX-License-Identifier: Apache-2.0

// The public header after the GL, EGL, GLFW and CUDA headers a consumer would include:
// no clashes, cudaStream_t/CUstream pass straight through, and the common GL loaders
// fit set_gl_proc_loader without casts.

#define GL_GLEXT_PROTOTYPES 1
#include <GL/gl.h>
#include <GL/glext.h>
#include <EGL/egl.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "wsl_cuda_gl.h"

#include <type_traits>

static_assert(std::is_same_v<cudaStream_t, CUstream_st*>);
static_assert(std::is_same_v<CUstream, CUstream_st*>);
static_assert(std::is_same_v<GLuint, unsigned int>);
static_assert(std::is_same_v<decltype(&glfwGetProcAddress), wslcg::GLProcLoader>);
static_assert(std::is_same_v<decltype(&eglGetProcAddress), wslcg::GLProcLoader>);

bool header_compiles_after_gl_and_cuda() {
    void (wslcg::Buffer::*unmap)(cudaStream_t) = &wslcg::Buffer::unmap;
    return unmap != nullptr;
}
