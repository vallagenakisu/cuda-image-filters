#include "cuda_utils.cuh"
#include "gpu_filters.hpp"
#include "pixel_ops.hpp"

namespace cif {
namespace gpu {
namespace {

// The first kernel of the project, and the template every other one follows.
//
// One thread per output pixel. The grid is 2D because the data is 2D; mapping
// a 2D image onto a 1D grid works but makes the coalescing story harder to
// reason about. Threads within a block that differ only in threadIdx.x read
// adjacent pixels, so a warp reads a contiguous run of bytes - which is what
// the memory controller wants.
__global__ void grayscale_kernel(const uint8_t* __restrict__ in,
                                 uint8_t* __restrict__ out,
                                 int width, int height, int channels) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // The last block on each axis usually hangs off the edge of the image,
    // because the image dimensions are rarely a multiple of the block size.
    // Without this guard those threads write out of bounds.
    if (x >= width || y >= height) return;

    const size_t pixel = static_cast<size_t>(y) * width + x;
    const uint8_t* src = in + pixel * channels;

    float value;
    if (channels >= 3) {
        value = luminance(static_cast<float>(src[0]), static_cast<float>(src[1]),
                          static_cast<float>(src[2]));
    } else {
        value = static_cast<float>(src[0]);
    }

    out[pixel] = clamp_to_byte(value);
}

}  // namespace

Timings grayscale(const Image& in, Image& out, const Options& opt) {
    out = Image(in.width, in.height, 1);

    DeviceBuffer<uint8_t> d_in(in.byte_size());
    DeviceBuffer<uint8_t> d_out(out.byte_size());

    GpuTimer timer;

    timer.start();
    d_in.upload(in.data.data(), in.byte_size());
    Timings t;
    t.upload_ms = timer.stop();

    const dim3 block(opt.block_x, opt.block_y);
    const dim3 grid(ceil_div(in.width, static_cast<int>(block.x)),
                    ceil_div(in.height, static_cast<int>(block.y)));

    timer.start();
    grayscale_kernel<<<grid, block>>>(d_in.get(), d_out.get(), in.width, in.height, in.channels);
    t.kernel_ms = timer.stop();
    CUDA_CHECK_KERNEL();

    timer.start();
    d_out.download(out.data.data(), out.byte_size());
    t.download_ms = timer.stop();

    return t;
}

}  // namespace gpu
}  // namespace cif
