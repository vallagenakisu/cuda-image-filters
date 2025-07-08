#include "cuda_utils.cuh"

namespace cif {

void print_device_info() {
    int count = 0;
    const cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        std::printf("no CUDA devices: %s\n", cudaGetErrorString(err));
        return;
    }
    if (count == 0) {
        std::printf("no CUDA devices found\n");
        return;
    }

    int runtime_version = 0;
    int driver_version = 0;
    cudaRuntimeGetVersion(&runtime_version);
    cudaDriverGetVersion(&driver_version);
    std::printf("CUDA runtime %d.%d, driver %d.%d\n", runtime_version / 1000,
                (runtime_version % 1000) / 10, driver_version / 1000, (driver_version % 1000) / 10);

    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        std::printf("\ndevice %d: %s\n", i, prop.name);
        std::printf("  compute capability     %d.%d\n", prop.major, prop.minor);
        std::printf("  multiprocessors        %d\n", prop.multiProcessorCount);
        std::printf("  global memory          %.2f GiB\n",
                    static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));
        std::printf("  shared memory / block  %zu KiB\n", prop.sharedMemPerBlock / 1024);
        std::printf("  constant memory        %zu KiB\n", prop.totalConstMem / 1024);
        std::printf("  max threads / block    %d\n", prop.maxThreadsPerBlock);
        std::printf("  max block dims         %d x %d x %d\n", prop.maxThreadsDim[0],
                    prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        std::printf("  warp size              %d\n", prop.warpSize);
        std::printf("  memory bus width       %d bit\n", prop.memoryBusWidth);
        std::printf("  memory clock           %.0f MHz\n", prop.memoryClockRate / 1000.0);

        // The number the benchmark compares against: anything close to this is
        // memory bound, which every filter here is.
        const double peak_bw = 2.0 * prop.memoryClockRate * 1000.0 * (prop.memoryBusWidth / 8.0);
        std::printf("  theoretical bandwidth  %.1f GB/s\n", peak_bw / 1.0e9);
    }
}

void require_cuda_device() {
    int count = 0;
    const cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0) {
        std::fprintf(stderr,
                     "error: no usable CUDA device (%s)\n"
                     "       run with --backend cpu to use the reference implementation\n",
                     err == cudaSuccess ? "device count is zero" : cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaSetDevice(0));
}

}  // namespace cif
