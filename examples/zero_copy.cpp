// Copyright 2026 Tyler Casper
// SPDX-License-Identifier: Apache-2.0

// The simplest use of wsl-cuda-gl: one shared buffer, CUDA renders into it, GL draws
// from it, nothing goes through host memory.
//
//   --display ssbo     the fragment shader reads the buffer directly (no copy at all)
//   --display texture  one GPU-side copy into a GL texture (the buffer as a PBO), for
//                      code that needs a real sampler2D
//
// Each frame is fully serialized: map() waits for GL, unmap() waits for CUDA. See
// pipelined.cpp for overlapping the two.

#include "app.h"
#include "pattern.cuh"
#include "wsl_cuda_gl.h"

// Everything GL/CUDA-owned lives in here, so it is released before the context.
static bool run(App& app) {
    const Options& opt = app.opt;
    const int w = opt.width, h = opt.height;
    const bool use_texture = opt.display == "texture";

    wslcg::Buffer frame_buf((size_t)w * h * sizeof(uint32_t));

    GLuint prog, tex = 0;
    if (use_texture) {
        glCreateTextures(GL_TEXTURE_2D, 1, &tex);
        glTextureStorage2D(tex, 1, GL_RGBA8, w, h);
        prog = make_texture_blit_program();
    } else {
        prog = make_buffer_blit_program(w, h);
    }
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    PhaseTimer timer;
    bool ok = true;
    int frame = 0;
    for (; app_running(app, frame); ++frame) {
        double t0 = now_ms();
        auto* px = static_cast<uint32_t*>(frame_buf.map()); // waits for GL to finish with it
        double t1 = now_ms();
        launch_busy_wait(opt.cuda_work_us, stream);
        launch_pattern_linear(px, w, w, h, frame, stream);
        CUDA_CHECK(cudaGetLastError());
        frame_buf.unmap(stream); // waits for the kernels
        double t2 = now_ms();
        glUseProgram(prog);
        if (use_texture) {
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, frame_buf.gl_buffer());
            glTextureSubImage2D(tex, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
            glBindTextureUnit(0, tex);
        } else {
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, frame_buf.gl_buffer());
        }
        glDrawArrays(GL_TRIANGLES, 0, 3);
        double t3 = now_ms();

        if (opt.frames && frame == opt.frames - 1 && (opt.verify || !opt.dump.empty()))
            ok = verify_backbuffer(w, h, frame, opt.dump);

        double t4 = now_ms();
        glfwSwapBuffers(app.window);
        double t5 = now_ms();

        timer.add("map: wait for GL", t1 - t0);
        timer.add("CUDA render + unmap", t2 - t1);
        timer.add("GL draw (submit)", t3 - t2);
        timer.add("present (swap)", t5 - t4);
        timer.end_frame();
    }
    timer.report();

    if (tex) glDeleteTextures(1, &tex);
    glDeleteProgram(prog);
    CUDA_CHECK(cudaStreamDestroy(stream));
    return ok;
}

int main(int argc, char** argv) {
    App app = app_init(parse_options(argc, argv), "zero_copy: CUDA -> GL via wsl-cuda-gl");
    bool ok = run(app);
    app_shutdown(app);
    return ok ? 0 : 1;
}
