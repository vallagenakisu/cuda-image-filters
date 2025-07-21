#include "cuda_utils.cuh"
#include "gpu_filters.hpp"
#include "pixel_ops.hpp"

namespace cif {
namespace gpu {
namespace {

// The Sobel operator is two fixed 3x3 matrices, so the taps are baked in
// rather than uploaded. They are __device__ const arrays, which the compiler
// is free to keep in registers or immediates.
__device__ const float kSobelX[9] = {-1.0f, 0.0f, 1.0f,
                                     -2.0f, 0.0f, 2.0f,
                                     -1.0f, 0.0f, 1.0f};
__device__ const float kSobelY[9] = {-1.0f, -2.0f, -1.0f,
                                     0.0f, 0.0f, 0.0f,
                                     1.0f, 2.0f, 1.0f};

// A gradient is only defined on a scalar field, so colour is reduced to luma
// at each tap and the output is one channel: the gradient magnitude.
//
// Nine taps is small enough that the redundant global reads are cheap and the
// L2 cache absorbs most of them - this one is left naive on purpose, as the
// contrast to the tiled convolution.
__global__ void sobel_kernel(const uint8_t* __restrict__ in, uint8_t* __restrict__ out,
                             int width, int height, int channels) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int max_x = width - 1;
    const int max_y = height - 1;

    float gx = 0.0f;
    float gy = 0.0f;

    for (int ky = -1; ky <= 1; ++ky) {
        const int sy = clamp_coord(y + ky, max_y);
        for (int kx = -1; kx <= 1; ++kx) {
            const int sx = clamp_coord(x + kx, max_x);
            const uint8_t* p = in + (static_cast<size_t>(sy) * width + sx) * channels;

            const float l = channels >= 3
                                ? luminance(static_cast<float>(p[0]), static_cast<float>(p[1]),
                                            static_cast<float>(p[2]))
                                : static_cast<float>(p[0]);

            const int tap = (ky + 1) * 3 + (kx + 1);
            gx += kSobelX[tap] * l;
            gy += kSobelY[tap] * l;
        }
    }

    out[static_cast<size_t>(y) * width + x] = clamp_to_byte(sqrtf(gx * gx + gy * gy));
}

}  // namespace

Timings sobel(const Image& in, Image& out, const Options& opt) {
    out = Image(in.width, in.height, 1);

    DeviceBuffer<uint8_t> d_in(in.byte_size());
    DeviceBuffer<uint8_t> d_out(out.byte_size());

    GpuTimer timer;
    Timings t;
    t.method = "sobel";

    timer.start();
    d_in.upload(in.data.data(), in.byte_size());
    t.upload_ms = timer.stop();

    const dim3 block(opt.block_x, opt.block_y);
    const dim3 grid(ceil_div(in.width, static_cast<int>(block.x)),
                    ceil_div(in.height, static_cast<int>(block.y)));

    timer.start();
    sobel_kernel<<<grid, block>>>(d_in.get(), d_out.get(), in.width, in.height, in.channels);
    t.kernel_ms = timer.stop();
    CUDA_CHECK_KERNEL();

    timer.start();
    d_out.download(out.data.data(), out.byte_size());
    t.download_ms = timer.stop();

    return t;
}

}  // namespace gpu
}  // namespace cif
