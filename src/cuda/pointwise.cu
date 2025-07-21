#include "cuda_utils.cuh"
#include "gpu_filters.hpp"
#include "pixel_ops.hpp"

namespace cif {
namespace gpu {
namespace {

// Pointwise filters touch each pixel exactly once, so there is nothing to
// cache and nothing to tile - they are purely memory bound. The useful thing
// to notice in the benchmark is that these run at close to the device's
// theoretical bandwidth while the convolutions do not.

__global__ void invert_kernel(const uint8_t* __restrict__ in, uint8_t* __restrict__ out,
                              int width, int height, int channels) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const size_t base = (static_cast<size_t>(y) * width + x) * channels;
    for (int c = 0; c < channels; ++c) {
        out[base + c] = is_alpha_channel(c, channels)
                            ? in[base + c]
                            : static_cast<uint8_t>(255 - in[base + c]);
    }
}

__global__ void tonemap_kernel(const uint8_t* __restrict__ in, uint8_t* __restrict__ out,
                               int width, int height, int channels, float exposure, float white,
                               float gamma) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const size_t base = (static_cast<size_t>(y) * width + x) * channels;
    const float inv_gamma = 1.0f / gamma;
    const float white_sq = white * white;
    const int colour_channels = channels >= 3 ? 3 : 1;

    // Undo the display gamma to get roughly linear light, then expose.
    // Tone mapping in gamma space crushes the shadows.
    float linear[3] = {0.0f, 0.0f, 0.0f};
    for (int c = 0; c < colour_channels; ++c) {
        linear[c] = powf(static_cast<float>(in[base + c]) / 255.0f, gamma) * exposure;
    }

    const float l = colour_channels == 3 ? luminance(linear[0], linear[1], linear[2]) : linear[0];

    // Extended Reinhard: compresses highlights towards `white` and leaves the
    // shadows nearly untouched. Scaling all channels by one luminance ratio
    // keeps the hue; tone mapping each channel separately desaturates.
    float scale = 0.0f;
    if (l > 1e-6f) {
        const float mapped = l * (1.0f + l / white_sq) / (1.0f + l);
        scale = mapped / l;
    }

    for (int c = 0; c < colour_channels; ++c) {
        out[base + c] = clamp_to_byte(powf(linear[c] * scale, inv_gamma) * 255.0f);
    }
    if (channels == 4) out[base + 3] = in[base + 3];
}

// Both pointwise filters share the same setup, so it lives in one place.
template <typename LaunchFn>
Timings run_pointwise(const Image& in, Image& out, const Options& opt, const char* name,
                      LaunchFn launch) {
    out = Image(in.width, in.height, in.channels);

    DeviceBuffer<uint8_t> d_in(in.byte_size());
    DeviceBuffer<uint8_t> d_out(out.byte_size());

    GpuTimer timer;
    Timings t;
    t.method = name;

    timer.start();
    d_in.upload(in.data.data(), in.byte_size());
    t.upload_ms = timer.stop();

    const dim3 block(opt.block_x, opt.block_y);
    const dim3 grid(ceil_div(in.width, static_cast<int>(block.x)),
                    ceil_div(in.height, static_cast<int>(block.y)));

    timer.start();
    launch(grid, block, d_in.get(), d_out.get());
    t.kernel_ms = timer.stop();
    CUDA_CHECK_KERNEL();

    timer.start();
    d_out.download(out.data.data(), out.byte_size());
    t.download_ms = timer.stop();

    return t;
}

}  // namespace

Timings invert(const Image& in, Image& out, const Options& opt) {
    const int w = in.width, h = in.height, c = in.channels;
    return run_pointwise(in, out, opt, "invert",
                         [w, h, c](dim3 grid, dim3 block, const uint8_t* d_in, uint8_t* d_out) {
                             invert_kernel<<<grid, block>>>(d_in, d_out, w, h, c);
                         });
}

Timings tonemap(const Image& in, Image& out, const Options& opt) {
    const int w = in.width, h = in.height, c = in.channels;
    const float exposure = opt.exposure, white = opt.white, gamma = opt.gamma;
    return run_pointwise(in, out, opt, "tonemap",
                         [w, h, c, exposure, white, gamma](dim3 grid, dim3 block,
                                                           const uint8_t* d_in, uint8_t* d_out) {
                             tonemap_kernel<<<grid, block>>>(d_in, d_out, w, h, c, exposure, white,
                                                             gamma);
                         });
}

}  // namespace gpu
}  // namespace cif
