// Copyright 2026 Tyler Casper
// SPDX-License-Identifier: Apache-2.0

#include "app.h"

#include <chrono>
#include <cstring>

#include "pattern.cuh"
#include "wsl_cuda_gl.h"

Options parse_options(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            CHECK(i + 1 < argc, "%s needs a value", a.c_str());
            return argv[++i];
        };
        if (a == "--size") {
            CHECK(sscanf(next(), "%dx%d", &o.width, &o.height) == 2, "--size expects WxH");
        } else if (a == "--frames") {
            o.frames = atoi(next());
        } else if (a == "--hidden") {
            o.hidden = true;
        } else if (a == "--verify") {
            o.verify = true;
        } else if (a == "--vsync") {
            o.vsync = true;
        } else if (a == "--dump") {
            o.dump = next();
        } else if (a == "--cuda-work-us") {
            o.cuda_work_us = atoi(next());
        } else if (a == "--display") {
            o.display = next();
            CHECK(o.display == "ssbo" || o.display == "texture", "--display expects ssbo or texture");
        } else {
            fprintf(stderr,
                    "usage: %s [--size WxH] [--frames N] [--hidden] [--verify] [--vsync] "
                    "[--dump out.ppm] [--display ssbo|texture] [--cuda-work-us N]\n",
                    argv[0]);
            exit(a == "--help" || a == "-h" ? 0 : 1);
        }
    }
    if (o.hidden && o.frames == 0) o.frames = 120; // nothing would ever close the window
    return o;
}

static void GLAPIENTRY gl_debug_cb(GLenum, GLenum type, GLuint, GLenum severity, GLsizei,
                                   const GLchar* msg, const void*) {
    if (type == GL_DEBUG_TYPE_ERROR || severity == GL_DEBUG_SEVERITY_HIGH)
        fprintf(stderr, "GL: %s\n", msg);
}

App app_init(const Options& opt, const char* title) {
    // Without this Mesa may choose llvmpipe (CPU) or another GPU (e.g. an iGPU), and no
    // zero-copy path to CUDA's GPU would be possible.
    wslcg::configure_mesa();

    glfwSetErrorCallback([](int code, const char* msg) { fprintf(stderr, "GLFW %d: %s\n", code, msg); });
    CHECK(glfwInit(), "glfwInit failed (is DISPLAY set?)");
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    glfwWindowHint(GLFW_VISIBLE, opt.hidden ? GLFW_FALSE : GLFW_TRUE);

    App app;
    app.opt = opt;
    app.window = glfwCreateWindow(opt.width, opt.height, title, nullptr, nullptr);
    CHECK(app.window, "glfwCreateWindow failed");
    glfwMakeContextCurrent(app.window);
    wslcg::set_gl_proc_loader(glfwGetProcAddress);
    glfwSwapInterval(opt.vsync ? 1 : 0);

    const char* renderer = (const char*)glGetString(GL_RENDERER);
    printf("GL renderer: %s\nGL version:  %s\n", renderer, (const char*)glGetString(GL_VERSION));
    if (!strstr(renderer, "D3D12"))
        fprintf(stderr, "warning: GL is not running on Mesa's D3D12 driver; it is probably on the CPU\n");

    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    glDebugMessageCallback(gl_debug_cb, nullptr);

    int fbw, fbh;
    glfwGetFramebufferSize(app.window, &fbw, &fbh);
    CHECK(fbw >= opt.width && fbh >= opt.height, "framebuffer %dx%d smaller than image", fbw, fbh);
    glViewport(0, 0, opt.width, opt.height);
    return app;
}

bool app_running(App& app, int frames_done) {
    glfwPollEvents();
    if (glfwWindowShouldClose(app.window)) return false;
    return app.opt.frames == 0 || frames_done < app.opt.frames;
}

void app_shutdown(App& app) {
    glfwDestroyWindow(app.window);
    glfwTerminate();
}

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
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
        fprintf(stderr, "shader compile failed:\n%s\n", log);
        exit(1);
    }
    return s;
}

GLuint compile_program(const char* fs_source) {
    // Fullscreen triangle generated from gl_VertexID; no vertex buffers needed.
    static const char* vs = R"(#version 450 core
        void main() {
            vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
            gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
        })";
    GLuint p = glCreateProgram();
    GLuint v = compile_shader(GL_VERTEX_SHADER, vs), f = compile_shader(GL_FRAGMENT_SHADER, fs_source);
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetProgramInfoLog(p, sizeof log, nullptr, log);
        fprintf(stderr, "program link failed:\n%s\n", log);
        exit(1);
    }
    glDeleteShader(v);
    glDeleteShader(f);
    // Core profile requires a bound VAO for any draw, even an attribute-less one.
    static GLuint vao = 0;
    if (!vao) {
        glCreateVertexArrays(1, &vao);
        glBindVertexArray(vao);
    }
    return p;
}

GLuint make_texture_blit_program() {
    return compile_program(R"(#version 450 core
        layout(binding = 0) uniform sampler2D u_tex;
        out vec4 o_color;
        void main() {
            ivec2 size = textureSize(u_tex, 0);
            ivec2 p = ivec2(gl_FragCoord.xy);
            o_color = texelFetch(u_tex, ivec2(p.x, size.y - 1 - p.y), 0);
        })");
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

bool verify_backbuffer(int w, int h, int frame, const std::string& dump_path) {
    std::vector<uint32_t> px((size_t)w * h);
    glReadBuffer(GL_BACK);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, px.data());

    // glReadPixels returns rows bottom-up; image row y is framebuffer row h-1-y.
    size_t bad = 0;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            uint32_t got = px[(size_t)(h - 1 - y) * w + x], want = pattern_rgba(x, y, w, h, frame);
            if (got != want && bad++ < 5)
                fprintf(stderr, "  mismatch at (%d,%d): got %08x want %08x\n", x, y, got, want);
        }
    }

    if (!dump_path.empty()) {
        FILE* f = fopen(dump_path.c_str(), "wb");
        CHECK(f, "cannot open %s", dump_path.c_str());
        fprintf(f, "P6\n%d %d\n255\n", w, h);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                uint32_t c = px[(size_t)(h - 1 - y) * w + x];
                uint8_t rgb[3] = {uint8_t(c), uint8_t(c >> 8), uint8_t(c >> 16)};
                fwrite(rgb, 1, 3, f);
            }
        fclose(f);
        printf("wrote %s\n", dump_path.c_str());
    }

    if (bad)
        printf("VERIFY FAILED: %zu of %d pixels differ from what CUDA rendered (frame %d)\n", bad, w * h, frame);
    else
        printf("VERIFY OK: all %d displayed pixels match what CUDA rendered (frame %d)\n", w * h, frame);
    return bad == 0;
}

void PhaseTimer::add(const char* name, double ms) {
    if (frames_ < warmup_) return;
    for (auto& t : totals_)
        if (t.first == name) {
            t.second += ms;
            return;
        }
    totals_.emplace_back(name, ms);
}

void PhaseTimer::report() const {
    int frames = frames_ - warmup_;
    if (frames <= 0) return;
    double sum = 0;
    printf("per-frame average over %d frames (after %d warm-up):\n", frames, warmup_);
    for (auto& t : totals_) {
        printf("  %-28s %8.3f ms\n", t.first.c_str(), t.second / frames);
        sum += t.second;
    }
    printf("  %-28s %8.3f ms  (%.0f fps)\n", "total", sum / frames, 1000.0 * frames / sum);
}
