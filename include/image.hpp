#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cif {

// A plain 8-bit-per-channel image held in host memory.
//
// Pixels are stored interleaved and row-major, which is exactly the layout
// stb_image hands us and exactly the layout we upload to the device. Pixel
// (x, y) channel c lives at data[(y * width + x) * channels + c].
struct Image {
    int width = 0;
    int height = 0;
    int channels = 0;  // 1 = gray, 3 = RGB, 4 = RGBA
    std::vector<uint8_t> data;

    Image() = default;
    Image(int w, int h, int c) : width(w), height(h), channels(c), data(static_cast<size_t>(w) * h * c, 0) {}

    bool empty() const { return data.empty() || width <= 0 || height <= 0; }

    // Number of pixels (not bytes).
    size_t pixel_count() const { return static_cast<size_t>(width) * static_cast<size_t>(height); }

    // Total bytes, i.e. what we pass to cudaMalloc / cudaMemcpy.
    size_t byte_size() const { return pixel_count() * static_cast<size_t>(channels); }

    // Bytes in one row. Our buffers are tightly packed, so there is no
    // separate pitch to worry about on the host side.
    size_t row_bytes() const { return static_cast<size_t>(width) * static_cast<size_t>(channels); }

    uint8_t* at(int x, int y) { return data.data() + (static_cast<size_t>(y) * width + x) * channels; }
    const uint8_t* at(int x, int y) const { return data.data() + (static_cast<size_t>(y) * width + x) * channels; }

    std::string describe() const {
        return std::to_string(width) + "x" + std::to_string(height) + "x" + std::to_string(channels);
    }
};

}  // namespace cif
