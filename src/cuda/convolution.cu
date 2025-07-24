#include <chrono>

#include "conv_kernel.hpp"
#include "cpu_reference.hpp"
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

// The 1D taps used by the separable path, at most 65 of them.
__constant__ float c_taps[2 * kMaxKernelRadius + 1];

void upload_kernel(const ConvKernel& kernel) {
    CUDA_CHECK(cudaMemcpyToSymbol(c_kernel, kernel.weights.data(),
                                  kernel.weights.size() * sizeof(float)));
    if (kernel.separable) {
        // horizontal and vertical are the same vector for every kernel here
        // (both blurs are symmetric), so one upload covers both passes.
        CUDA_CHECK(cudaMemcpyToSymbol(c_taps, kernel.horizontal.data(),
                                      kernel.horizontal.size() * sizeof(float)));
    }
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

// Tiled version: the block cooperatively stages its input into shared memory
// once, then every thread convolves out of that.
//
// The block writes blockDim.x by blockDim.y output pixels, but to do so it
// needs a (blockDim.x + 2r) by (blockDim.y + 2r) window of input - the extra
// ring is the "halo", the pixels the threads on the edge of the block reach
// into and which are shared with the neighbouring blocks.
//
// With a 16x16 block and radius 4 that is a 24x24 tile: 576 loads to serve
// 256 threads x 81 taps = 20736 reads. That ratio is the whole optimisation.
//
// The tile is float rather than uint8. It costs 4x the shared memory, but it
// keeps the byte-to-float conversion out of the inner loop and avoids the
// bank conflicts that byte-wide shared accesses produce.
__global__ void convolve_tiled_kernel(const uint8_t* __restrict__ in,
                                      uint8_t* __restrict__ out,
                                      int width, int height, int channels,
                                      int radius, float bias) {
    extern __shared__ float s_tile[];

    const int tile_w = static_cast<int>(blockDim.x) + 2 * radius;
    const int tile_h = static_cast<int>(blockDim.y) + 2 * radius;

    // Top-left corner of the tile in image space, halo included, so it starts
    // `radius` pixels above and to the left of the block's output region.
    const int origin_x = static_cast<int>(blockIdx.x * blockDim.x) - radius;
    const int origin_y = static_cast<int>(blockIdx.y * blockDim.y) - radius;

    const int max_x = width - 1;
    const int max_y = height - 1;

    // There are more tile cells than threads, so each thread loads several,
    // striding by the block dimensions. Striding (rather than giving each
    // thread a contiguous chunk) keeps consecutive threads on consecutive
    // addresses, so these loads coalesce too.
    for (int ty = threadIdx.y; ty < tile_h; ty += blockDim.y) {
        const int sy = clamp_coord(origin_y + ty, max_y);
        for (int tx = threadIdx.x; tx < tile_w; tx += blockDim.x) {
            const int sx = clamp_coord(origin_x + tx, max_x);
            const uint8_t* src = in + (static_cast<size_t>(sy) * width + sx) * channels;
            float* dst = s_tile + (static_cast<size_t>(ty) * tile_w + tx) * channels;
            for (int c = 0; c < channels; ++c) {
                dst[c] = static_cast<float>(src[c]);
            }
        }
    }

    // Every thread must finish loading before any thread starts reading its
    // neighbours' cells.
    __syncthreads();

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Note this guard comes *after* __syncthreads(), not before. Returning
    // early from the out-of-range threads would leave them sitting out a
    // barrier the rest of the block is waiting on, which is undefined
    // behaviour and in practice a hang. Those threads still had halo cells to
    // load, too.
    if (x >= width || y >= height) return;

    const int ksize = 2 * radius + 1;
    const size_t pixel = static_cast<size_t>(y) * width + x;

    for (int c = 0; c < channels; ++c) {
        if (is_alpha_channel(c, channels)) {
            out[pixel * channels + c] = in[pixel * channels + c];
            continue;
        }

        float acc = bias;
        for (int ky = 0; ky < ksize; ++ky) {
            // Tap (kx, ky) of the matrix corresponds to tile cell
            // (threadIdx + k): the thread's own centre sits at
            // threadIdx + radius, and the tap offset is k - radius.
            const size_t row = (static_cast<size_t>(threadIdx.y) + ky) * tile_w;
            for (int kx = 0; kx < ksize; ++kx) {
                const float w = c_kernel[ky * ksize + kx];
                acc += w * s_tile[(row + threadIdx.x + kx) * channels + c];
            }
        }
        out[pixel * channels + c] = clamp_to_byte(acc);
    }
}

// A separable kernel is the outer product of two 1D vectors, so convolving
// with the row vector and then the column vector gives the same result as the
// full matrix - but in 2*K taps per pixel instead of K*K. At radius 8 that is
// 34 taps instead of 289.
//
// The intermediate stays in float rather than being written back to bytes:
// quantising between the two passes would throw away most of the precision the
// second pass needs, and it shows up as banding in a strong blur.
__global__ void convolve_horizontal_kernel(const uint8_t* __restrict__ in,
                                           float* __restrict__ mid,
                                           int width, int height, int channels, int radius) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int ksize = 2 * radius + 1;
    const int max_x = width - 1;
    const size_t pixel = static_cast<size_t>(y) * width + x;
    const size_t row = static_cast<size_t>(y) * width;

    for (int c = 0; c < channels; ++c) {
        if (is_alpha_channel(c, channels)) {
            mid[pixel * channels + c] = static_cast<float>(in[pixel * channels + c]);
            continue;
        }
        float acc = 0.0f;
        for (int k = 0; k < ksize; ++k) {
            const int sx = clamp_coord(x + k - radius, max_x);
            acc += c_taps[k] * static_cast<float>(in[(row + sx) * channels + c]);
        }
        mid[pixel * channels + c] = acc;
    }
}

__global__ void convolve_vertical_kernel(const float* __restrict__ mid,
                                         uint8_t* __restrict__ out,
                                         int width, int height, int channels, int radius,
                                         float bias) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int ksize = 2 * radius + 1;
    const int max_y = height - 1;
    const size_t pixel = static_cast<size_t>(y) * width + x;

    for (int c = 0; c < channels; ++c) {
        if (is_alpha_channel(c, channels)) {
            out[pixel * channels + c] = clamp_to_byte(mid[pixel * channels + c]);
            continue;
        }
        // The bias belongs to the matrix as a whole, so it is added once here
        // rather than in both passes.
        float acc = bias;
        for (int k = 0; k < ksize; ++k) {
            const int sy = clamp_coord(y + k - radius, max_y);
            acc += c_taps[k] * mid[(static_cast<size_t>(sy) * width + x) * channels + c];
        }
        out[pixel * channels + c] = clamp_to_byte(acc);
    }
}

// Shared memory the tiled kernel needs for one block.
size_t tile_bytes(int block_x, int block_y, int radius, int channels) {
    const size_t tile_w = static_cast<size_t>(block_x) + 2 * radius;
    const size_t tile_h = static_cast<size_t>(block_y) + 2 * radius;
    return tile_w * tile_h * static_cast<size_t>(channels) * sizeof(float);
}

// One place that knows how to launch each variant, so convolve() and the
// benchmark cannot end up measuring different code from what they ship.
void launch(ConvMethod method, dim3 grid, dim3 block, const Options& opt, const ConvKernel& kernel,
            const uint8_t* d_in, uint8_t* d_out, float* d_mid, int width, int height,
            int channels) {
    switch (method) {
        case ConvMethod::Separable:
            convolve_horizontal_kernel<<<grid, block>>>(d_in, d_mid, width, height, channels,
                                                        kernel.radius);
            // No sync between the two: same stream, and a stream is ordered.
            convolve_vertical_kernel<<<grid, block>>>(d_mid, d_out, width, height, channels,
                                                      kernel.radius, kernel.bias);
            break;
        case ConvMethod::Tiled: {
            // Third launch parameter: dynamic shared memory, because the tile
            // size depends on a radius only known at runtime.
            const size_t shmem = tile_bytes(opt.block_x, opt.block_y, kernel.radius, channels);
            convolve_tiled_kernel<<<grid, block, shmem>>>(d_in, d_out, width, height, channels,
                                                          kernel.radius, kernel.bias);
            break;
        }
        case ConvMethod::Naive:
        case ConvMethod::Auto:
            convolve_naive_kernel<<<grid, block>>>(d_in, d_out, width, height, channels,
                                                   kernel.radius, kernel.bias);
            break;
    }
}

}  // namespace

bool tiling_fits(const Options& opt, int radius, int channels) {
    const size_t bytes = tile_bytes(opt.block_x, opt.block_y, radius, channels);

    int device = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&device) != cudaSuccess) return false;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return false;

    // Leave headroom rather than taking the whole budget: a block that claims
    // every byte of shared memory limits occupancy to one block per SM, which
    // usually costs more than the tiling saves.
    return bytes <= prop.sharedMemPerBlock / 2;
}

Timings convolve(const Image& in, Image& out, const ConvKernel& kernel, const Options& opt) {
    if (kernel.tap_count() > kMaxTaps) {
        std::fprintf(stderr, "error: kernel is %dx%d, larger than the %d tap limit\n",
                     kernel.size(), kernel.size(), kMaxTaps);
        std::exit(EXIT_FAILURE);
    }

    out = Image(in.width, in.height, in.channels);

    // Decide which implementation to run before touching the device.
    ConvMethod method = opt.method;
    const bool fits = tiling_fits(opt, kernel.radius, in.channels);
    if (method == ConvMethod::Auto) {
        // Cheapest first: 2K taps beats K*K taps by more than tiling saves,
        // so a separable matrix takes the two-pass path even though it does
        // not use shared memory at all.
        if (kernel.separable) {
            method = ConvMethod::Separable;
        } else {
            method = fits ? ConvMethod::Tiled : ConvMethod::Naive;
        }
    } else if (method == ConvMethod::Separable && !kernel.separable) {
        std::fprintf(stderr,
                     "warning: '%s' is not a separable matrix, so it cannot be split into two\n"
                     "         1D passes; falling back to %s\n",
                     kernel.name.c_str(), fits ? "tiled" : "naive");
        method = fits ? ConvMethod::Tiled : ConvMethod::Naive;
    } else if (method == ConvMethod::Tiled && !fits) {
        std::fprintf(stderr,
                     "warning: a %dx%d tile for radius %d needs %zu KiB of shared memory,\n"
                     "         which does not fit; falling back to the naive kernel\n",
                     opt.block_x + 2 * kernel.radius, opt.block_y + 2 * kernel.radius,
                     kernel.radius,
                     tile_bytes(opt.block_x, opt.block_y, kernel.radius, in.channels) / 1024);
        method = ConvMethod::Naive;
    }

    DeviceBuffer<uint8_t> d_in(in.byte_size());
    DeviceBuffer<uint8_t> d_out(out.byte_size());

    GpuTimer timer;
    Timings t;
    t.method = method_name(method);

    timer.start();
    d_in.upload(in.data.data(), in.byte_size());
    upload_kernel(kernel);
    t.upload_ms = timer.stop();

    const dim3 block(opt.block_x, opt.block_y);
    const dim3 grid(ceil_div(in.width, static_cast<int>(block.x)),
                    ceil_div(in.height, static_cast<int>(block.y)));

    // Only the separable path needs the float intermediate, so it is not
    // allocated for the other two.
    DeviceBuffer<float> d_mid;
    if (method == ConvMethod::Separable) {
        d_mid.allocate(in.pixel_count() * static_cast<size_t>(in.channels));
    }

    timer.start();
    launch(method, grid, block, opt, kernel, d_in.get(), d_out.get(), d_mid.get(), in.width,
           in.height, in.channels);
    t.kernel_ms = timer.stop();
    CUDA_CHECK_KERNEL();

    timer.start();
    d_out.download(out.data.data(), out.byte_size());
    t.download_ms = timer.stop();

    return t;
}

void benchmark_convolution(const Image& in, const ConvKernel& kernel, const Options& opt) {
    const bool fits = tiling_fits(opt, kernel.radius, in.channels);

    DeviceBuffer<uint8_t> d_in(in.byte_size());
    DeviceBuffer<uint8_t> d_out(in.byte_size());
    DeviceBuffer<float> d_mid(in.pixel_count() * static_cast<size_t>(in.channels));

    d_in.upload(in.data.data(), in.byte_size());
    upload_kernel(kernel);

    const dim3 block(opt.block_x, opt.block_y);
    const dim3 grid(ceil_div(in.width, static_cast<int>(block.x)),
                    ceil_div(in.height, static_cast<int>(block.y)));

    std::printf("\nbenchmark  %s, %s, block %dx%d, %d iterations\n", in.describe().c_str(),
                kernel.describe().c_str(), opt.block_x, opt.block_y, opt.bench);
    std::printf("           tile for radius %d needs %zu KiB of shared memory (%s)\n",
                kernel.radius, tile_bytes(opt.block_x, opt.block_y, kernel.radius, in.channels) / 1024,
                fits ? "fits" : "does not fit, tiled path skipped");

    std::printf("\n%-12s %10s %10s %12s %10s\n", "method", "ms/iter", "Mpixel/s", "GB/s eff.",
                "vs naive");
    std::printf("%-12s %10s %10s %12s %10s\n", "------", "-------", "--------", "---------",
                "--------");

    const ConvMethod methods[] = {ConvMethod::Naive, ConvMethod::Tiled, ConvMethod::Separable};
    float naive_ms = 0.0f;

    for (ConvMethod method : methods) {
        if (method == ConvMethod::Tiled && !fits) continue;
        if (method == ConvMethod::Separable && !kernel.separable) continue;

        // One untimed run first: the very first launch pays for module load
        // and JIT, which would otherwise land entirely on the first method
        // measured and make it look slow.
        launch(method, grid, block, opt, kernel, d_in.get(), d_out.get(), d_mid.get(), in.width,
               in.height, in.channels);
        CUDA_CHECK_KERNEL();

        GpuTimer timer;
        timer.start();
        for (int i = 0; i < opt.bench; ++i) {
            launch(method, grid, block, opt, kernel, d_in.get(), d_out.get(), d_mid.get(),
                   in.width, in.height, in.channels);
        }
        const float total_ms = timer.stop();
        CUDA_CHECK_KERNEL();

        const float ms = total_ms / static_cast<float>(opt.bench);
        if (method == ConvMethod::Naive) naive_ms = ms;

        const double seconds = static_cast<double>(ms) / 1000.0;
        const double mpixels = static_cast<double>(in.pixel_count()) / 1.0e6 / seconds;

        // "Effective" bandwidth: the traffic the algorithm actually needs,
        // one read and one write per pixel, regardless of how many times the
        // naive version re-reads the same byte. Comparing this against the
        // device's theoretical peak from --list-devices is the honest measure
        // of how close a kernel gets.
        const double bytes = 2.0 * static_cast<double>(in.byte_size());
        const double gbps = bytes / seconds / 1.0e9;

        std::printf("%-12s %10.4f %10.1f %12.2f %9.2fx\n", method_name(method), ms, mpixels, gbps,
                    naive_ms > 0.0f ? static_cast<double>(naive_ms / ms) : 1.0);
    }

    // One CPU pass for scale. Not iterated - at these sizes it is slow enough
    // that one run is plenty and the number is only there for perspective.
    Image cpu_out;
    const auto cpu_start = std::chrono::steady_clock::now();
    cpu::convolve(in, cpu_out, kernel);
    const auto cpu_end = std::chrono::steady_clock::now();
    const double cpu_ms =
        std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    std::printf("%-12s %10.4f %10.1f %12s %9.2fx\n", "cpu (1 run)", cpu_ms,
                static_cast<double>(in.pixel_count()) / 1.0e6 / (cpu_ms / 1000.0), "-",
                naive_ms > 0.0f ? cpu_ms / static_cast<double>(naive_ms) : 0.0);

    std::printf("\nnote: kernel time only - the host<->device copies are excluded, and on a\n"
                "      single small image they can easily cost more than the filter does.\n");
}

}  // namespace gpu
}  // namespace cif
