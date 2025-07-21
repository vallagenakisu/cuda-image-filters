#pragma once

#include <string>

#include "conv_kernel.hpp"
#include "image.hpp"
#include "options.hpp"

namespace cif {

// Where the time actually goes. On small images the PCIe copies dominate and
// the kernel is noise, which is itself worth seeing.
struct Timings {
    float upload_ms = 0.0f;
    float kernel_ms = 0.0f;
    float download_ms = 0.0f;

    // Which convolution path actually ran, since --method auto decides at
    // runtime based on whether the tile fits in shared memory.
    const char* method = "";

    float total_ms() const { return upload_ms + kernel_ms + download_ms; }
};

namespace gpu {

// Luminance conversion. Always writes a single-channel image.
Timings grayscale(const Image& in, Image& out, const Options& opt);

// Apply an arbitrary square matrix. `opt.method` selects the implementation;
// ConvMethod::Auto picks the best one that is valid for this kernel and image.
Timings convolve(const Image& in, Image& out, const ConvKernel& kernel, const Options& opt);

// True when a tile of (block + 2*radius) fits in the device's shared memory
// budget for a block. The host needs this to decide what Auto means.
bool tiling_fits(const Options& opt, int radius, int channels);

// Gradient magnitude. Always writes a single-channel image.
Timings sobel(const Image& in, Image& out, const Options& opt);

// Pointwise filters: one read and one write per pixel, nothing to cache.
Timings invert(const Image& in, Image& out, const Options& opt);
Timings tonemap(const Image& in, Image& out, const Options& opt);

}  // namespace gpu
}  // namespace cif
