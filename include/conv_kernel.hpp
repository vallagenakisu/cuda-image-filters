#pragma once

#include <string>
#include <vector>

#include "options.hpp"

namespace cif {

// Largest radius any backend accepts. 32 gives a 65x65 matrix, which is
// 4225 floats = 16.5 KiB - still inside the 64 KiB of __constant__ memory the
// GPU path uploads the weights into.
inline constexpr int kMaxKernelRadius = 32;

// A square convolution matrix, stored row-major with the centre tap at
// index (radius, radius).
//
// `bias` is added after the weighted sum, which matters for kernels whose
// weights sum to zero: without it a Laplacian or a sum-zero emboss clips
// almost everything to black.
struct ConvKernel {
    std::string name = "custom";
    int radius = 1;
    float bias = 0.0f;
    std::vector<float> weights;  // (2*radius+1)^2

    // A separable kernel is the outer product of two 1D vectors, which turns
    // an O(K^2) convolution into two O(K) passes. Only the blurs qualify.
    bool separable = false;
    std::vector<float> horizontal;  // 2*radius+1
    std::vector<float> vertical;    // 2*radius+1

    int size() const { return 2 * radius + 1; }
    int tap_count() const { return size() * size(); }

    // kx and ky run over [-radius, radius].
    float at(int kx, int ky) const {
        return weights[static_cast<size_t>(ky + radius) * size() + (kx + radius)];
    }

    float sum() const;
    std::string describe() const;
};

// Builders for the named filters. Each returns a ready-to-use matrix so the
// CPU and GPU paths convolve with byte-identical weights.
ConvKernel make_box(int radius);
ConvKernel make_gaussian(int radius, float sigma);   // sigma <= 0 derives it from radius
ConvKernel make_sharpen(int radius, float sigma, float amount);  // unsharp mask
ConvKernel make_emboss();
ConvKernel make_laplacian();
ConvKernel make_identity();

// Pick and build the matrix implied by the options. Returns false with an
// explanation for filters that are not convolutions at all (grayscale, sobel,
// tonemap, invert).
bool build_kernel(const Options& opt, ConvKernel& out, std::string& error);

}  // namespace cif
