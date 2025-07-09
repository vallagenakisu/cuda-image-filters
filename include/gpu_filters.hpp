#pragma once

#include <string>

#include "image.hpp"
#include "options.hpp"

namespace cif {

// Where the time actually goes. On small images the PCIe copies dominate and
// the kernel is noise, which is itself worth seeing.
struct Timings {
    float upload_ms = 0.0f;
    float kernel_ms = 0.0f;
    float download_ms = 0.0f;

    float total_ms() const { return upload_ms + kernel_ms + download_ms; }
};

namespace gpu {

// Luminance conversion. Always writes a single-channel image.
Timings grayscale(const Image& in, Image& out, const Options& opt);

}  // namespace gpu
}  // namespace cif
