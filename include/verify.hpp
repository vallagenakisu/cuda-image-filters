#pragma once

#include <cstddef>
#include <string>

#include "image.hpp"

namespace cif {

// How far apart two images are, channel by channel.
struct DiffStats {
    bool comparable = false;  // false when the dimensions or channel counts differ
    long max_abs = 0;
    double mean_abs = 0.0;
    size_t differing = 0;
    size_t total = 0;
    double psnr_db = 0.0;  // infinity when the images are identical

    double differing_percent() const {
        return total == 0 ? 0.0 : 100.0 * static_cast<double>(differing) / static_cast<double>(total);
    }
};

DiffStats compare_images(const Image& a, const Image& b);

// One line for the console. `tolerance` is the largest per-channel difference
// that still counts as a pass.
void print_diff(const DiffStats& stats, long tolerance);

// Whether the difference is small enough to call it a match.
bool diff_within(const DiffStats& stats, long tolerance);

}  // namespace cif
