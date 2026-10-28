// Copyright 2026 Tyler Casper
// SPDX-License-Identifier: Apache-2.0

#include "gl_api.h"

#include <cstring>
#include <dlfcn.h>
#include <mutex>
#include <string>

namespace wslcg::detail {
namespace {

std::mutex g_mutex;
GLProcLoader g_user_loader = nullptr;
GLFunctions g_functions;
bool g_loaded = false;

// A symbol from whichever of `libs` the process has already loaded (with any
// RTLD_LOCAL/GLOBAL flag; GLFW, for one, loads libGL privately).
void* find_loaded_symbol(const char* const* libs, const char* symbol) {
    for (; *libs; ++libs) {
        void* lib = dlopen(*libs, RTLD_LAZY | RTLD_NOLOAD);
        if (!lib) continue;
        void* sym = dlsym(lib, symbol);
        dlclose(lib); // still loaded by whoever loaded it first
        if (sym) return sym;
    }
    return nullptr;
}

// eglGetProcAddress or glXGetProcAddressARB, whichever API has a context current.
GLProcLoader default_loader() {
    static const char* const egl[] = {"libEGL.so.1", "libEGL.so", nullptr};
    static const char* const glx[] = {"libGLX.so.0", "libGL.so.1", "libGL.so", nullptr};
    using GetCurrentContext = void* (*)();
    auto egl_current = reinterpret_cast<GetCurrentContext>(find_loaded_symbol(egl, "eglGetCurrentContext"));
    if (egl_current && egl_current())
        return reinterpret_cast<GLProcLoader>(find_loaded_symbol(egl, "eglGetProcAddress"));
    auto glx_current = reinterpret_cast<GetCurrentContext>(find_loaded_symbol(glx, "glXGetCurrentContext"));
    if (glx_current && glx_current())
        return reinterpret_cast<GLProcLoader>(find_loaded_symbol(glx, "glXGetProcAddressARB"));
    return nullptr;
}

template <typename Fn>
void load(GLProcLoader loader, Fn& fn, const char* name) {
    fn = reinterpret_cast<Fn>(loader(name));
    if (!fn) throw Error(std::string("wsl-cuda-gl: GL function ") + name + " is not available");
}

} // namespace

bool GLFunctions::has_extension(const char* name) const {
    GLint n = 0;
    GetIntegerv(gl::NUM_EXTENSIONS, &n);
    for (GLint i = 0; i < n; ++i) {
        auto ext = reinterpret_cast<const char*>(GetStringi(gl::EXTENSIONS, (GLuint)i));
        if (ext && !std::strcmp(ext, name)) return true;
    }
    return false;
}

const char* GLFunctions::string(GLenum name) const {
    auto s = reinterpret_cast<const char*>(GetString(name));
    return s ? s : "";
}

void set_gl_loader(GLProcLoader loader) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_user_loader = loader;
    g_loaded = false;
}

const GLFunctions& gl_functions() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_loaded) {
        GLProcLoader loader = g_user_loader ? g_user_loader : default_loader();
        if (!loader)
            throw Error("wsl-cuda-gl: no OpenGL context is current on this thread (looked for EGL and GLX). "
                        "Make one current first, or pass your loader to wslcg::set_gl_proc_loader().");
        GLFunctions& f = g_functions;
        load(loader, f.GetString, "glGetString");
        load(loader, f.GetStringi, "glGetStringi");
        load(loader, f.GetIntegerv, "glGetIntegerv");
        load(loader, f.GetError, "glGetError");
        load(loader, f.Finish, "glFinish");
        load(loader, f.FenceSync, "glFenceSync");
        load(loader, f.ClientWaitSync, "glClientWaitSync");
        load(loader, f.DeleteSync, "glDeleteSync");
        load(loader, f.CreateBuffers, "glCreateBuffers");
        load(loader, f.DeleteBuffers, "glDeleteBuffers");
        load(loader, f.ClearNamedBufferSubData, "glClearNamedBufferSubData");
        load(loader, f.GetNamedBufferSubData, "glGetNamedBufferSubData");
        load(loader, f.CreateMemoryObjectsEXT, "glCreateMemoryObjectsEXT");
        load(loader, f.DeleteMemoryObjectsEXT, "glDeleteMemoryObjectsEXT");
        load(loader, f.MemoryObjectParameterivEXT, "glMemoryObjectParameterivEXT");
        load(loader, f.ImportMemoryFdEXT, "glImportMemoryFdEXT");
        load(loader, f.NamedBufferStorageMemEXT, "glNamedBufferStorageMemEXT");
        g_loaded = true;
    }
    // Dispatch stubs exist for every function, so ask the driver whether a context
    // is actually current.
    if (!g_functions.GetString(gl::VERSION))
        throw Error("wsl-cuda-gl: no OpenGL context is current on this thread");
    return g_functions;
}

} // namespace wslcg::detail
