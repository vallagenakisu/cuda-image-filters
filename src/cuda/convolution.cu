#include "conv_kernel.hpp"
#include "cuda_utils.cuh"
#include "gpu_filters.hpp"
#include "pixel_ops.hpp"

namespace cif {
namespace gpu {
namespace {

// Largest matrix we can hold: 65x65 for radius 32, which is 16.5 KiB of the
// 64 KiB constant bank.
constexpr int kMaxTaps = (2 * kMaxKernelRadius + 1) * (2 * kMaxKernelRadius + 1);

// The weights go in __constant__ rather than global memory on purpose. Every
// thread in a warp reads the same tap at the same time, which is exactly the
// broadcast access pattern the constant cache is built for - one fetch serves
// the whole warp instead of 32 separate global loads.
__constant__ float c_kernel[kMaxTaps];

void upload_kernel(const ConvKernel& kernel) {
    CUDA_CHECK(cudaMemcpyToSymbol(c_kernel, kernel.weights.data(),
                                  kernel.weights.size() * sizeof(float)));
}

// Naive version: every tap is a separate global memory read.
//
// A 3x3 kernel therefore reads each pixel up to 9 times, a 9x9 up to 81 times.
// The L2 and texture caches absorb a lot of that, but the redundancy is real
// and it is what the tiled version exists to remove.
__global__ void convolve_naive_kernel(const uint8_t* __restrict__ in,
                                      uint8_t* __restrict__ out,
                                      int width, int height, int channels,
                                      int radius, float bias) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int ksize = 2 * radius + 1;
    const int max_x = width - 1;
    const int max_y = height - 1;
    const size_t pixel = static_cast<size_t>(y) * width + x;

    for (int c = 0; c < channels; ++c) {
        if (is_alpha_channel(c, channels)) {
            out[pixel * channels + c] = in[pixel * channels + c];
            continue;
        }

        float acc = bias;
        for (int ky = -radius; ky <= radius; ++ky) {
            const int sy = clamp_coord(y + ky, max_y);
            for (int kx = -radius; kx <= radius; ++kx) {
                const int sx = clamp_coord(x + kx, max_x);
                const float w = c_kernel[(ky + radius) * ksize + (kx + radius)];
                const size_t src = (static_cast<size_t>(sy) * width + sx) * channels + c;
                acc += w * static_cast<float>(in[src]);
            }
        }
        out[pixel * channels + c] = clamp_to_byte(acc);
    }
}

}  // namespace

bool tiling_fits(const Options& opt, int radius, int channels) {
    // A block of block_x by block_y output pixels needs a tile of
    // (block_x + 2r) by (block_y + 2r) input pixels, including the halo it
    // shares with its neighbours.
    const size_t tile_w = static_cast<size_t>(opt.block_x) + 2 * radius;
    const size_t tile_h = static_cast<size_t>(opt.block_y) + 2 * radius;
    const size_t bytes = tile_w * tile_h * static_cast<size_t>(channels) * sizeof(float);

    int device = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&device) != cudaSuccess) return false;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return false;

    // Leave a little headroom rather than taking the whole budget: a block
    // that claims every byte of shared memory limits occupancy to one block
    // per SM, which usually costs more than the tiling saves.
    return bytes <= prop.sharedMemPerBlock / 2;
}

Timings convolve(const Image& in, Image& out, const ConvKernel& kernel, const Options& opt) {
    if (kernel.tap_count() > kMaxTaps) {
        std::fprintf(stderr, "error: kernel is %dx%d, larger than the %d tap limit\n",
                     kernel.size(), kernel.size(), kMaxTaps);
        std::exit(EXIT_FAILURE);
    }

    out = Image(in.width, in.height, in.channels);

    DeviceBuffer<uint8_t> d_in(in.byte_size());
    DeviceBuffer<uint8_t> d_out(out.byte_size());

    GpuTimer timer;
    Timings t;
    t.method = "naive";

    timer.start();
    d_in.upload(in.data.data(), in.byte_size());
    upload_kernel(kernel);
    t.upload_ms = timer.stop();

    const dim3 block(opt.block_x, opt.block_y);
    const dim3 grid(ceil_div(in.width, static_cast<int>(block.x)),
                    ceil_div(in.height, static_cast<int>(block.y)));

    timer.start();
    convolve_naive_kernel<<<grid, block>>>(d_in.get(), d_out.get(), in.width, in.height,
                                           in.channels, kernel.radius, kernel.bias);
    t.kernel_ms = timer.stop();
    CUDA_CHECK_KERNEL();

    timer.start();
    d_out.download(out.data.data(), out.byte_size());
    t.download_ms = timer.stop();

    return t;
}

}  // namespace gpu
}  // namespace cif
