#include "verify.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace cif {

DiffStats compare_images(const Image& a, const Image& b) {
    DiffStats stats;
    if (a.width != b.width || a.height != b.height || a.channels != b.channels) {
        return stats;  // comparable stays false
    }

    stats.comparable = true;
    stats.total = a.data.size();

    double sum_abs = 0.0;
    double sum_sq = 0.0;
    for (size_t i = 0; i < a.data.size(); ++i) {
        const long d = static_cast<long>(a.data[i]) - static_cast<long>(b.data[i]);
        const long ad = d < 0 ? -d : d;
        if (ad > stats.max_abs) stats.max_abs = ad;
        if (ad != 0) ++stats.differing;
        sum_abs += static_cast<double>(ad);
        sum_sq += static_cast<double>(d) * static_cast<double>(d);
    }

    stats.mean_abs = stats.total == 0 ? 0.0 : sum_abs / static_cast<double>(stats.total);

    // PSNR is the usual way to say "how different" for images. Identical
    // images give a zero mean-squared error, which would be a division by
    // zero, so that case is reported as infinite instead.
    const double mse = stats.total == 0 ? 0.0 : sum_sq / static_cast<double>(stats.total);
    stats.psnr_db = mse == 0.0 ? HUGE_VAL : 10.0 * std::log10((255.0 * 255.0) / mse);

    return stats;
}

bool diff_within(const DiffStats& stats, long tolerance) {
    return stats.comparable && stats.max_abs <= tolerance;
}

void print_diff(const DiffStats& stats, long tolerance) {
    if (!stats.comparable) {
        std::printf("verify  FAILED: the two results have different dimensions\n");
        return;
    }

    if (stats.max_abs == 0) {
        std::printf("verify  exact match: every one of %zu channel values is identical\n",
                    stats.total);
        return;
    }

    char psnr[32];
    if (std::isinf(stats.psnr_db)) {
        std::snprintf(psnr, sizeof(psnr), "inf");
    } else {
        std::snprintf(psnr, sizeof(psnr), "%.2f dB", stats.psnr_db);
    }

    std::printf("verify  %s: max|diff| %ld, mean|diff| %.5f, %zu of %zu differ (%.4f%%), PSNR %s\n",
                diff_within(stats, tolerance) ? "within tolerance" : "FAILED", stats.max_abs,
                stats.mean_abs, stats.differing, stats.total, stats.differing_percent(), psnr);

    if (!diff_within(stats, tolerance)) {
        std::printf("        max|diff| exceeds the tolerance of %ld - this is a real difference,\n"
                    "        not float rounding. Suspect the indexing or the halo.\n",
                    tolerance);
    }
}

}  // namespace cif
