// Copyright 2026 Tyler Casper
// SPDX-License-Identifier: Apache-2.0

#pragma once
// The GL entry points and enums the library uses, declared here so building the
// library needs no GL headers (and cannot clash with the consumer's GL loader).

#include <cstddef>
#include <cstdint>

#include "wsl_cuda_gl.h"

namespace wslcg::detail {

using GLenum = unsigned int;
using GLuint = unsigned int;
using GLint = int;
using GLsizei = int;
using GLbitfield = unsigned int;
using GLuint64 = std::uint64_t;
using GLintptr = std::ptrdiff_t;
using GLsizeiptr = std::ptrdiff_t;
using GLubyte = unsigned char;
using GLsync = struct __GLsync*;

namespace gl {
constexpr GLenum RENDERER = 0x1F01;
constexpr GLenum VERSION = 0x1F02;
constexpr GLenum EXTENSIONS = 0x1F03;
constexpr GLenum NUM_EXTENSIONS = 0x821D;
constexpr GLenum UNSIGNED_INT = 0x1405;
constexpr GLenum R32UI = 0x8236;
constexpr GLenum RED_INTEGER = 0x8D94;
constexpr GLenum SYNC_GPU_COMMANDS_COMPLETE = 0x9117;
constexpr GLbitfield SYNC_FLUSH_COMMANDS_BIT = 0x1;
constexpr GLenum TIMEOUT_EXPIRED = 0x911B;
constexpr GLenum WAIT_FAILED = 0x911D;
constexpr GLenum DEDICATED_MEMORY_OBJECT_EXT = 0x9581;
constexpr GLenum HANDLE_TYPE_OPAQUE_FD_EXT = 0x9586;
} // namespace gl

struct GLFunctions {
    const GLubyte* (*GetString)(GLenum);
    const GLubyte* (*GetStringi)(GLenum, GLuint);
    void (*GetIntegerv)(GLenum, GLint*);
    GLenum (*GetError)();
    void (*Finish)();
    GLsync (*FenceSync)(GLenum, GLbitfield);
    GLenum (*ClientWaitSync)(GLsync, GLbitfield, GLuint64);
    void (*DeleteSync)(GLsync);
    void (*CreateBuffers)(GLsizei, GLuint*);
    void (*DeleteBuffers)(GLsizei, const GLuint*);
    void (*ClearNamedBufferSubData)(GLuint, GLenum, GLintptr, GLsizeiptr, GLenum, GLenum, const void*);
    void (*GetNamedBufferSubData)(GLuint, GLintptr, GLsizeiptr, void*);
    // GL_EXT_memory_object / GL_EXT_memory_object_fd
    void (*CreateMemoryObjectsEXT)(GLsizei, GLuint*);
    void (*DeleteMemoryObjectsEXT)(GLsizei, const GLuint*);
    void (*MemoryObjectParameterivEXT)(GLuint, GLenum, const GLint*);
    void (*ImportMemoryFdEXT)(GLuint, GLuint64, GLenum, GLint);
    void (*NamedBufferStorageMemEXT)(GLuint, GLsizeiptr, GLuint, GLuint64);

    bool has_extension(const char* name) const;
    const char* string(GLenum name) const; // "" instead of null
};

void set_gl_loader(GLProcLoader loader);

// The GL functions, loaded on first use. Throws Error if no GL context is current on
// this thread (or the functions can't be found).
const GLFunctions& gl_functions();

} // namespace wslcg::detail
