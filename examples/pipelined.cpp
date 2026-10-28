// Copyright 2026 Tyler Casper
// SPDX-License-Identifier: Apache-2.0

// Two wsl-cuda-gl buffers, so CUDA renders frame N while GL draws and presents N-1.
//
//   iteration N:  map(buf[N])          wait for GL's last use of buf[N] (frame N-2)
//                 kernel -> buf[N]     async; runs on the GPU while...
//                 draw buf[N-1]; swap  ...GL shows the previous frame
//                 unmap(buf[N])        wait for the kernel
//
// No frame data goes through host memory, and no glFinish or device-wide stall is left
// in the loop.

#include <memory>

#include "app.h"
#include "pattern.cuh"
#include "wsl_cuda_gl.h"

// Everything GL/CUDA-owned lives in here, so it is released before the context.
static bool run(App& app) {
    const Options& opt = app.opt;
    const int w = opt.width, h = opt.height;
    const size_t bytes = (size_t)w * h * sizeof(uint32_t);
    const bool use_texture = opt.display == "texture";

    wslcg::Buffer bufs[2] = {wslcg::Buffer(bytes), wslcg::Buffer(bytes)};

    GLuint prog, tex = 0;
    if (use_texture) {
        glCreateTextures(GL_TEXTURE_2D, 1, &tex);
        glTextureStorage2D(tex, 1, GL_RGBA8, w, h);
        prog = make_texture_blit_program();
    } else {
        prog = make_buffer_blit_program(w, h);
    }
    auto draw = [&](wslcg::Buffer& b) {
        glUseProgram(prog);
        if (use_texture) {
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, b.gl_buffer()); // GPU-side copy into the texture
            glTextureSubImage2D(tex, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
            glBindTextureUnit(0, tex);
        } else {
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, b.gl_buffer());
        }
        glDrawArrays(GL_TRIANGLES, 0, 3);
    };

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // With --verify, also check one frame mid-stream, displayed while CUDA is writing
    // the other buffer, as well as the last one.
    const int mid_check = opt.frames / 2;
    PhaseTimer timer;
    bool ok = true;
    int frame = 0;
    for (; app_running(app, frame); ++frame) {
        wslcg::Buffer& next = bufs[frame % 2];
        wslcg::Buffer& prev = bufs[(frame + 1) % 2]; // holds frame - 1

        double t0 = now_ms();
        auto* px = static_cast<uint32_t*>(next.map());
        double t1 = now_ms();
        launch_busy_wait(opt.cuda_work_us, stream);
        launch_pattern_linear(px, w, w, h, frame, stream);
        CUDA_CHECK(cudaGetLastError());
        bool verified = false;
        if (frame > 0) {
            draw(prev);
            if (opt.verify && frame - 1 == mid_check) {
                ok &= verify_backbuffer(w, h, frame - 1, "");
                verified = true;
            }
            glfwSwapBuffers(app.window);
        }
        double t2 = now_ms();
        next.unmap(stream);
        double t3 = now_ms();

        if (verified) continue; // keep the readback out of the timings
        timer.add("map: wait for GL", t1 - t0);
        timer.add("GL draw + present", t2 - t1);
        timer.add("unmap: wait for CUDA", t3 - t2);
        timer.end_frame();
    }
    // Drain the pipeline: show the last frame CUDA rendered.
    if (frame > 0) {
        draw(bufs[(frame - 1) % 2]);
        if (opt.verify || !opt.dump.empty()) ok &= verify_backbuffer(w, h, frame - 1, opt.dump);
        glfwSwapBuffers(app.window);
    }
    timer.report();

    if (tex) glDeleteTextures(1, &tex);
    glDeleteProgram(prog);
    CUDA_CHECK(cudaStreamDestroy(stream));
    return ok;
}

int main(int argc, char** argv) {
    App app = app_init(parse_options(argc, argv), "pipelined: CUDA -> GL via wsl-cuda-gl, double-buffered");
    bool ok = run(app);
    app_shutdown(app);
    return ok ? 0 : 1;
}
