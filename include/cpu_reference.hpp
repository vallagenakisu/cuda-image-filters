#pragma once

#include <string>

#include "conv_kernel.hpp"
#include "image.hpp"
#include "options.hpp"

namespace cif {
namespace cpu {

// Single-threaded, deliberately straightforward implementations of every
// filter. This is the oracle: --verify runs the GPU path and this path over
// the same input and reports the difference, so a kernel indexing bug shows up
// as a number instead of a picture that looks vaguely wrong.
//
// Because these call the same inline helpers in pixel_ops.hpp that the kernels
// do, an exact match is the expected result, not an approximation.

void grayscale(const Image& in, Image& out);
void invert(const Image& in, Image& out);
void convolve(const Image& in, Image& out, const ConvKernel& kernel);
void sobel(const Image& in, Image& out);
void tonemap(const Image& in, Image& out, float exposure, float white, float gamma);

// Run whichever filter the options select. Returns false with an explanation
// if the filter is unknown here.
bool run(const Image& in, Image& out, const Options& opt, std::string& error);

}  // namespace cpu
}  // namespace cif
