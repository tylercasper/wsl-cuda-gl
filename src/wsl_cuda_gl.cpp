// Copyright 2026 Tyler Casper
// SPDX-License-Identifier: Apache-2.0

#include "wsl_cuda_gl.h"

#include <cstdlib>
#include <mutex>
#include <set>

#include "cuda_api.h"
#include "d3d12_bridge.h"
#include "gl_api.h"
#include "mesa_version.h"

namespace wslcg {

using namespace detail;

void configure_mesa(int cuda_device) {
    // Mesa matches MESA_D3D12_DEFAULT_ADAPTER_NAME as a case-insensitive substring of
    // the adapter description, which on WSL is the same name CUDA reports.
    std::string adapter = "NVIDIA";
    if (const CudaDriver* cu = try_cuda_driver()) {
        CUdevice dev;
        char name[256];
        if (cu->DeviceGet(&dev, cuda_device) == CUDA_SUCCESS && cu->DeviceGetName(name, sizeof name, dev) == CUDA_SUCCESS)
            adapter = name;
    }
    setenv("GALLIUM_DRIVER", "d3d12", 0);
    setenv("MESA_D3D12_DEFAULT_ADAPTER_NAME", adapter.c_str(), 0);
}

void set_gl_proc_loader(GLProcLoader loader) { set_gl_loader(loader); }

struct detail::BufferState {
    const GLFunctions* gl = nullptr;
    const CudaDriver* cu = nullptr;
    CUcontext ctx = nullptr;
    bool retained_primary = false; // we retained device 0's primary context
    CUdevice primary_device = 0;
    std::unique_ptr<D3D12SharedBuffer> shared;
    GLuint memobj = 0, gl_buffer = 0;
    CUevent cuda_done = nullptr;
    std::unique_ptr<Kernels> kernels;
    bool mapped = false;

    CUdeviceptr device_address() const { return static_cast<CUdeviceptr>(shared->gpu_address()); }

    ~BufferState() {
        // Destructors must not throw; release what was created, in reverse order.
        if (gl) {
            if (gl_buffer) gl->DeleteBuffers(1, &gl_buffer);
            if (memobj) gl->DeleteMemoryObjectsEXT(1, &memobj);
        }
        if (cu && ctx && cu->CtxPushCurrent(ctx) == CUDA_SUCCESS) {
            kernels.reset();
            if (cuda_done) cu->EventDestroy(cuda_done);
            CUcontext ignored;
            cu->CtxPopCurrent(&ignored);
        }
        shared.reset();
        if (retained_primary) cu->DevicePrimaryCtxRelease(primary_device);
    }
};

namespace {

// Proves, once per CUDA context, that CUDA and GL really address the same memory on
// this driver. Sharing relies on undocumented behaviour (CUDA dereferencing a D3D12
// GPU virtual address), so a driver update could break it; better a clear Error here
// than silent corruption later.
void self_test(BufferState& b) {
    static std::mutex mutex;
    static std::set<CUcontext> passed;
    std::lock_guard<std::mutex> lock(mutex);
    if (passed.count(b.ctx)) return;

    const unsigned from_gl = 0x9E3779B9u, from_cuda = 0x7F4A7C15u;
    b.gl->ClearNamedBufferSubData(b.gl_buffer, gl::R32UI, 0, 4, gl::RED_INTEGER, gl::UNSIGNED_INT, &from_gl);
    b.gl->Finish();

    const CudaDriver& cu = *b.cu;
    CUdeviceptr seen = 0;
    cu.check(cu.MemAlloc(&seen, 4), "cuMemAlloc");
    unsigned cuda_saw = 0;
    CUresult r = CUDA_SUCCESS;
    try {
        b.kernels->selftest(b.device_address(), seen, from_cuda, nullptr);
        r = cu.CtxSynchronize();
        if (r == CUDA_SUCCESS) r = cu.MemcpyDtoH(&cuda_saw, seen, 4);
    } catch (...) {
        cu.MemFree(seen);
        throw;
    }
    cu.MemFree(seen);
    if (r != CUDA_SUCCESS)
        throw Error("wsl-cuda-gl: self-test failed: a CUDA kernel could not access the D3D12 buffer (" +
                    cu.error_string(r) + "). This driver may no longer share the GPU address space between "
                    "CUDA and D3D12; run probe_wsl_interop for details.");

    unsigned gl_saw = 0;
    b.gl->GetNamedBufferSubData(b.gl_buffer, 0, 4, &gl_saw);
    if (cuda_saw != from_gl || gl_saw != from_cuda) {
        char msg[256];
        std::snprintf(msg, sizeof msg,
                      "wsl-cuda-gl: self-test failed: CUDA read 0x%08x (GL wrote 0x%08x), GL read 0x%08x "
                      "(CUDA wrote 0x%08x). CUDA and GL are not seeing the same memory.",
                      cuda_saw, from_gl, gl_saw, from_cuda);
        throw Error(msg);
    }
    passed.insert(b.ctx);
}

} // namespace

Buffer::Buffer(std::size_t bytes) : impl_(std::make_unique<BufferState>()) {
    if (bytes == 0) throw Error("wsl-cuda-gl: Buffer size must be greater than 0");
    BufferState& b = *impl_;

    // GL: must be Mesa's d3d12 driver, with memory-object import.
    b.gl = &gl_functions();
    const GLFunctions& gl = *b.gl;
    const std::string renderer = gl.string(gl::RENDERER), version = gl.string(gl::VERSION);
    if (renderer.find("D3D12") == std::string::npos)
        throw Error("wsl-cuda-gl: the current GL context runs on \"" + renderer +
                    "\", not Mesa's d3d12 driver. Call wslcg::configure_mesa() before creating it (or set "
                    "GALLIUM_DRIVER=d3d12).");
    if (!gl.has_extension("GL_EXT_memory_object_fd"))
        throw Error("wsl-cuda-gl: the GL driver lacks GL_EXT_memory_object_fd (" + version + ")");

    // CUDA: the current context, or device 0's primary context like the runtime.
    b.cu = &cuda_driver();
    const CudaDriver& cu = *b.cu;
    cu.check(cu.CtxGetCurrent(&b.ctx), "cuCtxGetCurrent");
    if (!b.ctx) {
        cu.check(cu.DeviceGet(&b.primary_device, 0), "cuDeviceGet");
        cu.check(cu.DevicePrimaryCtxRetain(&b.ctx, b.primary_device), "cuDevicePrimaryCtxRetain");
        b.retained_primary = true;
    }
    ScopedContext scope(cu, b.ctx);
    CUdevice dev;
    char cuda_name[256] = {};
    cu.check(cu.CtxGetDevice(&dev), "cuCtxGetDevice");
    cu.check(cu.DeviceGetName(cuda_name, sizeof cuda_name, dev), "cuDeviceGetName");

    // The same GPU must be behind all three APIs.
    if (renderer.find(cuda_name) == std::string::npos)
        throw Error("wsl-cuda-gl: GL renders on \"" + renderer + "\" but CUDA uses \"" + cuda_name +
                    "\". Call wslcg::configure_mesa(<cuda device>) before creating the GL context.");
    b.shared = D3D12SharedBuffer::create(bytes, cuda_name);

    // GL imports the D3D12 buffer.
    gl.GetError(); // clear stale errors so the check below is about the import
    gl.CreateMemoryObjectsEXT(1, &b.memobj);
    const GLint dedicated = 1; // a committed D3D12 resource is a dedicated allocation
    gl.MemoryObjectParameterivEXT(b.memobj, gl::DEDICATED_MEMORY_OBJECT_EXT, &dedicated);
    {
        MesaMemobjImportFix fix(*b.shared, version.c_str());
        gl.ImportMemoryFdEXT(b.memobj, bytes, gl::HANDLE_TYPE_OPAQUE_FD_EXT, b.shared->export_fd()); // GL owns the fd
        gl.Finish(); // the import must have run before the fix is removed
        if (fix.active() && fix.redirected_calls() == 0)
            throw Error("wsl-cuda-gl: the Mesa memory-object workaround did not reach Mesa's D3D12 device");
    }
    gl.CreateBuffers(1, &b.gl_buffer);
    gl.NamedBufferStorageMemEXT(b.gl_buffer, (GLsizeiptr)bytes, b.memobj, 0);
    if (GLenum err = gl.GetError())
        throw Error("wsl-cuda-gl: GL failed to import the D3D12 buffer (GL error 0x" + [&] {
            char hex[16];
            std::snprintf(hex, sizeof hex, "%04x", err);
            return std::string(hex);
        }() + ")");

    cu.check(cu.EventCreate(&b.cuda_done, CU_EVENT_DISABLE_TIMING), "cuEventCreate");
    b.kernels = std::make_unique<Kernels>(cu);
    self_test(b);
}

Buffer::~Buffer() = default;
Buffer::Buffer(Buffer&&) noexcept = default;
Buffer& Buffer::operator=(Buffer&&) noexcept = default;

unsigned int Buffer::gl_buffer() const noexcept { return impl_ ? impl_->gl_buffer : 0; }
std::size_t Buffer::size() const noexcept { return impl_ ? impl_->shared->size() : 0; }
bool Buffer::is_mapped() const noexcept { return impl_ && impl_->mapped; }

void* Buffer::map() {
    if (!impl_) throw Error("wsl-cuda-gl: map() on a moved-from Buffer");
    BufferState& b = *impl_;
    if (b.mapped) throw Error("wsl-cuda-gl: Buffer is already mapped");
    // Wait for every GL command issued so far: any of them may still use the buffer.
    GLsync fence = b.gl->FenceSync(gl::SYNC_GPU_COMMANDS_COMPLETE, 0);
    GLenum r;
    do {
        r = b.gl->ClientWaitSync(fence, gl::SYNC_FLUSH_COMMANDS_BIT, 1000000000ull);
    } while (r == gl::TIMEOUT_EXPIRED);
    b.gl->DeleteSync(fence);
    if (r == gl::WAIT_FAILED) throw Error("wsl-cuda-gl: glClientWaitSync failed (is the GL context current?)");
    b.mapped = true;
    return reinterpret_cast<void*>(static_cast<uintptr_t>(b.device_address()));
}

void Buffer::unmap(CUstream_st* stream) {
    if (!impl_) throw Error("wsl-cuda-gl: unmap() on a moved-from Buffer");
    BufferState& b = *impl_;
    if (!b.mapped) throw Error("wsl-cuda-gl: Buffer is not mapped");
    ScopedContext scope(*b.cu, b.ctx);
    b.cu->check(b.cu->EventRecord(b.cuda_done, stream), "cuEventRecord");
    b.cu->check(b.cu->EventSynchronize(b.cuda_done), "waiting for CUDA work on the buffer");
    b.mapped = false;
}

static void check_copy(const std::unique_ptr<BufferState>& impl, std::size_t bytes, std::size_t offset) {
    if (!impl) throw Error("wsl-cuda-gl: copy on a moved-from Buffer");
    if (!impl->mapped) throw Error("wsl-cuda-gl: copies need the Buffer to be mapped");
    if (offset > impl->shared->size() || bytes > impl->shared->size() - offset)
        throw Error("wsl-cuda-gl: copy range exceeds the Buffer");
}

void Buffer::copy_from_device(const void* src, std::size_t bytes, std::size_t offset, CUstream_st* stream) {
    check_copy(impl_, bytes, offset);
    ScopedContext scope(*impl_->cu, impl_->ctx);
    impl_->kernels->copy(impl_->device_address() + offset, reinterpret_cast<CUdeviceptr>(src), bytes, stream);
}

void Buffer::copy_to_device(void* dst, std::size_t bytes, std::size_t offset, CUstream_st* stream) const {
    check_copy(impl_, bytes, offset);
    ScopedContext scope(*impl_->cu, impl_->ctx);
    impl_->kernels->copy(reinterpret_cast<CUdeviceptr>(dst), impl_->device_address() + offset, bytes, stream);
}

SupportReport check_support() {
    SupportReport report;
    try {
        const GLFunctions& gl = gl_functions();
        report.gl_renderer = gl.string(gl::RENDERER);
        report.gl_version = gl.string(gl::VERSION);
        report.mesa_workaround = mesa_has_memobj_import_bug(report.gl_version.c_str());
        const CudaDriver& cu = cuda_driver();
        CUcontext ctx = nullptr;
        CUdevice dev = 0;
        char name[256] = {};
        if (cu.CtxGetCurrent(&ctx) == CUDA_SUCCESS && ctx)
            cu.CtxGetDevice(&dev);
        else
            cu.DeviceGet(&dev, 0);
        if (cu.DeviceGetName(name, sizeof name, dev) == CUDA_SUCCESS) report.cuda_device = name;
        Buffer probe(4096); // creating one runs the self-test
        report.supported = true;
    } catch (const std::exception& e) {
        report.reason = e.what();
    }
    return report;
}

} // namespace wslcg
