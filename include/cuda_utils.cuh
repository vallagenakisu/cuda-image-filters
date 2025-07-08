#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace cif {

// Every CUDA runtime call returns a status and almost every bug in a beginner
// CUDA program is a status nobody looked at. Wrap the lot.
#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        const cudaError_t _cif_err = (call);                                              \
        if (_cif_err != cudaSuccess) {                                                    \
            std::fprintf(stderr, "CUDA error at %s:%d\n  %s\n  -> %s\n", __FILE__,        \
                         __LINE__, #call, cudaGetErrorString(_cif_err));                  \
            std::exit(EXIT_FAILURE);                                                      \
        }                                                                                 \
    } while (0)

// Kernel launches do not return a status. cudaGetLastError picks up launch
// configuration errors, the synchronise picks up faults raised while running.
#define CUDA_CHECK_KERNEL()                       \
    do {                                          \
        CUDA_CHECK(cudaGetLastError());           \
        CUDA_CHECK(cudaDeviceSynchronize());      \
    } while (0)

// Grid sizing: how many blocks of `b` threads are needed to cover `a` items.
// The kernels then guard with `if (x >= width || y >= height) return;` because
// the last block along each axis is usually partly outside the image.
__host__ __device__ inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// RAII device allocation, so an early return cannot leak GPU memory.
template <typename T>
class DeviceBuffer {
  public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(size_t count) { allocate(count); }

    ~DeviceBuffer() { release(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    void allocate(size_t count) {
        release();
        if (count == 0) return;
        CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
        count_ = count;
    }

    void release() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);  // destructor path: nothing useful to do with a failure
            ptr_ = nullptr;
        }
        count_ = 0;
    }

    void upload(const T* host, size_t count) {
        CUDA_CHECK(cudaMemcpy(ptr_, host, count * sizeof(T), cudaMemcpyHostToDevice));
    }

    void download(T* host, size_t count) const {
        CUDA_CHECK(cudaMemcpy(host, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
    }

    void zero() { CUDA_CHECK(cudaMemset(ptr_, 0, count_ * sizeof(T))); }

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t count() const { return count_; }
    size_t bytes() const { return count_ * sizeof(T); }

  private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

// Wall-clock timing around a kernel is misleading because launches are
// asynchronous. CUDA events are recorded in the stream itself, so they measure
// the kernel rather than the launch.
class GpuTimer {
  public:
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }

    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    void start(cudaStream_t stream = 0) { CUDA_CHECK(cudaEventRecord(start_, stream)); }

    // Milliseconds between the two events. Blocks until the stop event is
    // reached, so call it once per measurement rather than in a hot loop.
    float stop(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(stop_, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

  private:
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};

// Print every visible device. Exposed for --list-devices, and worth reading
// once because the shared-memory-per-block number is what caps the tile size.
void print_device_info();

// Abort with a readable message if no CUDA device is usable.
void require_cuda_device();

}  // namespace cif
