// Copyright 2026 Tyler Casper
// SPDX-License-Identifier: Apache-2.0

// CUDA writes, GL reads: every pixel GL draws (from the buffer as an SSBO) and every
// texel it copies into a texture (the buffer as a PBO) must be exactly what CUDA wrote.

#include <vector>

#include "gl_env.h"
#include "harness.h"
#include "pattern.cuh"

static void run_size(int w, int h, int frame) {
    std::printf("--- %dx%d\n", w, h);
    const size_t bytes = (size_t)w * h * 4;
    wslcg::Buffer buf(bytes);
    CHECK(buf.size() == bytes);
    CHECK(buf.gl_buffer() != 0);
    GLint64 gl_size = 0;
    glGetNamedBufferParameteri64v(buf.gl_buffer(), GL_BUFFER_SIZE, &gl_size);
    CHECK((size_t)gl_size == bytes);

    auto* px = static_cast<uint32_t*>(buf.map());
    REQUIRE(px != nullptr);
    launch_pattern_linear(px, w, w, h, frame, nullptr);
    CUDA_REQUIRE(cudaGetLastError());
    buf.unmap(nullptr);

    // Drawn straight from the buffer.
    Fbo fbo(w, h);
    fbo.bind();
    GLuint prog = make_buffer_blit_program(w, h);
    glUseProgram(prog);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buf.gl_buffer());
    glDrawArrays(GL_TRIANGLES, 0, 3);
    size_t bad = pattern_mismatches(fbo.read_top_down(), w, h, frame);
    std::printf("SSBO draw: %zu of %zu pixels differ\n", bad, (size_t)w * h);
    CHECK(bad == 0);
    glDeleteProgram(prog);

    // Copied into a texture on the GPU (texture row 0 = buffer row 0 = top).
    GLuint tex;
    glCreateTextures(GL_TEXTURE_2D, 1, &tex);
    glTextureStorage2D(tex, 1, GL_RGBA8, w, h);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buf.gl_buffer());
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTextureSubImage2D(tex, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    std::vector<uint32_t> texels((size_t)w * h);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glGetTextureImage(tex, 0, GL_RGBA, GL_UNSIGNED_BYTE, (GLsizei)bytes, texels.data());
    bad = pattern_mismatches(texels, w, h, frame);
    std::printf("PBO -> texture: %zu of %zu texels differ\n", bad, (size_t)w * h);
    CHECK(bad == 0);
    glDeleteTextures(1, &tex);
    CHECK(glGetError() == GL_NO_ERROR);
}

int main() {
    HeadlessGL gl;
    require_gpu_environment(gl);
    const int sizes[][2] = {{1, 1}, {333, 2}, {1001, 777}, {1920, 1080}, {3840, 2160}};
    int frame = 3;
    for (auto& s : sizes) run_size(s[0], s[1], frame += 7);
    return harness::finish();
}
