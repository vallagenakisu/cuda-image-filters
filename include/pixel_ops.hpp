#pragma once

#include <cstdint>

// This header is included by both .cpp and .cu translation units on purpose:
// the CPU reference backend and the CUDA kernels call the *same* inline
// functions, so --verify compares two implementations of the algorithm rather
// than two implementations of rounding.
#ifdef __CUDACC__
#define CIF_HD __host__ __device__
#else
#define CIF_HD
#endif

namespace cif {

// Convert an accumulated float back to an 8-bit channel value.
// Round-half-away-from-zero then saturate; both backends must agree exactly or
// --verify reports off-by-one noise that is not a real bug.
CIF_HD inline uint8_t clamp_to_byte(float v) {
    v = v < 0.0f ? 0.0f : v;
    v = v > 255.0f ? 255.0f : v;
    return static_cast<uint8_t>(v + 0.5f);
}

// Rec.601 luma, the same weights stb and most image tools use.
CIF_HD inline float luminance(float r, float g, float b) {
    return 0.299f * r + 0.587f * g + 0.114f * b;
}

// Clamp-to-edge addressing. Convolution taps near a border would otherwise
// read outside the image; repeating the edge pixel is the cheapest border mode
// and it keeps blurs from darkening at the frame.
CIF_HD inline int clamp_coord(int v, int hi) {
    v = v < 0 ? 0 : v;
    return v > hi ? hi : v;
}

// Alpha, when present, is channel 3 and is copied through untouched rather
// than filtered - blurring alpha against colour produces halos.
CIF_HD inline bool is_alpha_channel(int c, int channels) { return channels == 4 && c == 3; }

}  // namespace cif
