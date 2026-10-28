// Copyright 2026 Tyler Casper
// SPDX-License-Identifier: Apache-2.0

// Baseline (no library): CUDA and OpenGL side by side, frame bounced through host memory.
//
//   CUDA kernel -> device buffer -> cudaMemcpy D2H (pinned host) -> glTexSubImage2D -> draw
//
// This works anywhere. zero_copy and pipelined remove the
// D2H copy and the host->GL upload using wsl-cuda-gl.

#include "app.h"
#include "pattern.cuh"

int main(int argc, char** argv) {
    Options opt = parse_options(argc, argv);
    App app = app_init(opt, "host_copy: CUDA -> host memory -> GL");
    const int w = opt.width, h = opt.height;
    const size_t bytes = (size_t)w * h * sizeof(uint32_t);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("CUDA device: %s\n", prop.name);

    uint32_t *d_frame, *h_frame;
    CUDA_CHECK(cudaMalloc(&d_frame, bytes));
    CUDA_CHECK(cudaMallocHost(&h_frame, bytes)); // pinned, so the D2H copy is a real DMA
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    GLuint tex;
    glCreateTextures(GL_TEXTURE_2D, 1, &tex);
    glTextureStorage2D(tex, 1, GL_RGBA8, w, h);
    GLuint prog = make_texture_blit_program();

    PhaseTimer timer;
    bool ok = true;
    int frame = 0;
    for (; app_running(app, frame); ++frame) {
        // Each phase is synchronized so its cost can be attributed.
        double t0 = now_ms();
        launch_busy_wait(opt.cuda_work_us, stream);
        launch_pattern_linear(d_frame, w, w, h, frame, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        double t1 = now_ms();
        CUDA_CHECK(cudaMemcpyAsync(h_frame, d_frame, bytes, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        double t2 = now_ms();
        glTextureSubImage2D(tex, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, h_frame);
        glFinish();
        double t3 = now_ms();
        glUseProgram(prog);
        glBindTextureUnit(0, tex);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();
        double t4 = now_ms();

        if (opt.frames && frame == opt.frames - 1 && (opt.verify || !opt.dump.empty()))
            ok = verify_backbuffer(w, h, frame, opt.dump);

        double t5 = now_ms();
        glfwSwapBuffers(app.window);
        double t6 = now_ms();

        timer.add("CUDA render", t1 - t0);
        timer.add("copy device -> host", t2 - t1);
        timer.add("upload host -> GL", t3 - t2);
        timer.add("GL draw", t4 - t3);
        timer.add("present (swap)", t6 - t5);
        timer.end_frame();
    }
    timer.report();

    glDeleteTextures(1, &tex);
    glDeleteProgram(prog);
    CUDA_CHECK(cudaFree(d_frame));
    CUDA_CHECK(cudaFreeHost(h_frame));
    CUDA_CHECK(cudaStreamDestroy(stream));
    app_shutdown(app);
    return ok ? 0 : 1;
}
