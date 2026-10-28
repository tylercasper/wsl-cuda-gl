// Copyright 2026 Tyler Casper
// SPDX-License-Identifier: Apache-2.0

#include "gl_env.h"

#include <EGL/egl.h>
#include <EGL/eglext.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

#include <cuda_runtime.h>

#include "harness.h"
#include "pattern.cuh"

HeadlessGL::~HeadlessGL() {
    if (!display_) return;
    eglMakeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    if (context_) eglDestroyContext(display_, context_);
    eglTerminate(display_);
}

bool HeadlessGL::create(std::string* why) {
    const char* client_ext = eglQueryString(EGL_NO_DISPLAY, EGL_EXTENSIONS);
    if (!client_ext || !std::strstr(client_ext, "EGL_MESA_platform_surfaceless")) {
        *why = "EGL has no surfaceless platform (is Mesa's EGL installed?)";
        return false;
    }
    display_ = eglGetPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, nullptr);
    if (!display_ || !eglInitialize(display_, nullptr, nullptr)) {
        *why = "eglInitialize failed on the surfaceless platform";
        display_ = nullptr;
        return false;
    }
    eglBindAPI(EGL_OPENGL_API);
    const EGLint config_attribs[] = {EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_NONE};
    EGLConfig config = nullptr;
    EGLint n = 0;
    eglChooseConfig(display_, config_attribs, &config, 1, &n);
    const EGLint ctx_attribs[] = {EGL_CONTEXT_MAJOR_VERSION,       4, EGL_CONTEXT_MINOR_VERSION, 5,
                                  EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
                                  EGL_NONE};
    context_ = eglCreateContext(display_, n ? config : EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, ctx_attribs);
    if (!context_ || !eglMakeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE, context_)) {
        *why = "could not create and make current a GL 4.5 core context";
        return false;
    }
    return true;
}

std::string HeadlessGL::renderer() const {
    auto r = reinterpret_cast<const char*>(glGetString(GL_RENDERER));
    return r ? r : "";
}

void require_gpu_environment(HeadlessGL& gl) {
    const char* strict = std::getenv("WSLCG_REQUIRE_GPU_TESTS");
    auto unavailable = [&](const std::string& why) {
        if (strict && std::strcmp(strict, "0") != 0) {
            std::fprintf(stderr, "GPU test prerequisites missing (WSLCG_REQUIRE_GPU_TESTS is set): %s\n", why.c_str());
            std::exit(1);
        }
        harness::skip(why);
    };
    if (access("/dev/dxg", F_OK) != 0) unavailable("not WSL2 with GPU support (no /dev/dxg)");
    int devices = 0;
    cudaError_t err = cudaGetDeviceCount(&devices);
    if (err != cudaSuccess || devices == 0)
        unavailable(std::string("no CUDA device (") + cudaGetErrorString(err) + ")");
    CUDA_REQUIRE(cudaFree(nullptr)); // create the runtime's primary context
    wslcg::configure_mesa();
    std::string why;
    if (!gl.create(&why)) unavailable(why);
    if (gl.renderer().find("D3D12") == std::string::npos)
        unavailable("GL is not on Mesa's d3d12 driver (renderer: " + gl.renderer() + ")");
    std::printf("GL renderer: %s\nGL version:  %s\n", gl.renderer().c_str(), glGetString(GL_VERSION));
}

static GLuint link(GLuint program) {
    glLinkProgram(program);
    GLint ok = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetProgramInfoLog(program, sizeof log, nullptr, log);
        std::fprintf(stderr, "program link failed:\n%s\n", log);
        std::exit(1);
    }
    return program;
}

static GLuint compile_shader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(s, sizeof log, nullptr, log);
        std::fprintf(stderr, "shader compile failed:\n%s\n", log);
        std::exit(1);
    }
    return s;
}

GLuint compile_program(const char* fragment_source) {
    static const char* vs = R"(#version 450 core
        void main() {
            vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
            gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
        })";
    GLuint p = glCreateProgram();
    GLuint v = compile_shader(GL_VERTEX_SHADER, vs), f = compile_shader(GL_FRAGMENT_SHADER, fragment_source);
    glAttachShader(p, v);
    glAttachShader(p, f);
    link(p);
    glDeleteShader(v);
    glDeleteShader(f);
    static GLuint vao = 0; // core profile needs one bound for any draw
    if (!vao) {
        glCreateVertexArrays(1, &vao);
        glBindVertexArray(vao);
    }
    return p;
}

GLuint compile_compute_program(const char* source) {
    GLuint p = glCreateProgram();
    GLuint c = compile_shader(GL_COMPUTE_SHADER, source);
    glAttachShader(p, c);
    link(p);
    glDeleteShader(c);
    return p;
}

GLuint make_buffer_blit_program(int w, int h) {
    GLuint p = compile_program(R"(#version 450 core
        layout(std430, binding = 0) readonly buffer Frame { uint px[]; };
        layout(location = 0) uniform ivec2 u_size;
        out vec4 o_color;
        void main() {
            ivec2 p = ivec2(gl_FragCoord.xy);
            o_color = unpackUnorm4x8(px[(u_size.y - 1 - p.y) * u_size.x + p.x]);
        })");
    glProgramUniform2i(p, 0, w, h);
    return p;
}

Fbo::Fbo(int w, int h) : w_(w), h_(h) {
    glCreateTextures(GL_TEXTURE_2D, 1, &tex_);
    glTextureStorage2D(tex_, 1, GL_RGBA8, w, h);
    glCreateFramebuffers(1, &fbo_);
    glNamedFramebufferTexture(fbo_, GL_COLOR_ATTACHMENT0, tex_, 0);
}

Fbo::~Fbo() {
    glDeleteFramebuffers(1, &fbo_);
    glDeleteTextures(1, &tex_);
}

void Fbo::bind() const {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
    glViewport(0, 0, w_, h_);
}

std::vector<uint32_t> Fbo::read_top_down() const {
    std::vector<uint32_t> bottom_up((size_t)w_ * h_), top_down((size_t)w_ * h_);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo_);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, w_, h_, GL_RGBA, GL_UNSIGNED_BYTE, bottom_up.data());
    for (int y = 0; y < h_; ++y)
        std::memcpy(&top_down[(size_t)y * w_], &bottom_up[(size_t)(h_ - 1 - y) * w_], (size_t)w_ * 4);
    return top_down;
}

size_t pattern_mismatches(const std::vector<uint32_t>& img, int w, int h, int frame) {
    size_t bad = 0;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            uint32_t got = img[(size_t)y * w + x], want = pattern_rgba(x, y, w, h, frame);
            if (got != want && bad++ < 3)
                std::fprintf(stderr, "  (%d,%d): got %08x want %08x\n", x, y, got, want);
        }
    return bad;
}
