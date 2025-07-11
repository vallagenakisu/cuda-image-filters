#include "conv_kernel.hpp"

#include <cmath>
#include <cstdio>

namespace cif {
namespace {

// Outer product of a 1D vector with itself, used by the separable blurs so the
// 2D matrix and the two 1D passes cannot drift apart.
void fill_from_1d(ConvKernel& k, const std::vector<float>& taps) {
    const int n = k.size();
    k.weights.assign(static_cast<size_t>(n) * n, 0.0f);
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            k.weights[static_cast<size_t>(y) * n + x] = taps[y] * taps[x];
        }
    }
    k.separable = true;
    k.horizontal = taps;
    k.vertical = taps;
}

ConvKernel make_3x3(const char* name, const float (&w)[9], float bias) {
    ConvKernel k;
    k.name = name;
    k.radius = 1;
    k.bias = bias;
    k.weights.assign(w, w + 9);
    return k;
}

}  // namespace

float ConvKernel::sum() const {
    float s = 0.0f;
    for (float w : weights) s += w;
    return s;
}

std::string ConvKernel::describe() const {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "%s %dx%d (sum %.3f%s)", name.c_str(), size(), size(), sum(),
                  separable ? ", separable" : "");
    return std::string(buf);
}

ConvKernel make_box(int radius) {
    ConvKernel k;
    k.name = "box";
    k.radius = radius;
    const int n = k.size();
    // A box blur is the outer product of two uniform vectors, so normalising
    // each 1D pass by n gives the 2D normalisation of n*n for free.
    const std::vector<float> taps(static_cast<size_t>(n), 1.0f / static_cast<float>(n));
    fill_from_1d(k, taps);
    return k;
}

ConvKernel make_gaussian(int radius, float sigma) {
    ConvKernel k;
    k.name = "gaussian";
    k.radius = radius;
    const int n = k.size();

    // A gaussian is numerically zero past about 3 sigma, so picking
    // sigma = radius/3 makes the truncation at the kernel edge harmless.
    if (sigma <= 0.0f) sigma = radius > 0 ? static_cast<float>(radius) / 3.0f : 1.0f;

    std::vector<float> taps(static_cast<size_t>(n), 0.0f);
    float total = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float d = static_cast<float>(i - radius);
        taps[i] = std::exp(-(d * d) / (2.0f * sigma * sigma));
        total += taps[i];
    }
    // Normalise the 1D taps; the 2D outer product is then normalised too.
    for (float& t : taps) t /= total;

    fill_from_1d(k, taps);
    return k;
}

ConvKernel make_sharpen(int radius, float sigma, float amount) {
    // Unsharp mask: result = original + amount * (original - blurred),
    // which folds into a single matrix as (1 + amount) * I - amount * G.
    ConvKernel k = make_gaussian(radius, sigma);
    k.name = "sharpen";

    for (float& w : k.weights) w = -amount * w;
    const size_t centre = static_cast<size_t>(radius) * k.size() + radius;
    k.weights[centre] += 1.0f + amount;

    // The combination is no longer a plain outer product.
    k.separable = false;
    k.horizontal.clear();
    k.vertical.clear();
    return k;
}

ConvKernel make_emboss() {
    // Sums to 1, so mid-grey stays mid-grey and no bias term is needed.
    const float w[9] = {-2.0f, -1.0f, 0.0f,
                        -1.0f, 1.0f, 1.0f,
                        0.0f, 1.0f, 2.0f};
    return make_3x3("emboss", w, 0.0f);
}

ConvKernel make_laplacian() {
    // Sums to 0: flat regions go to black and only edges survive.
    const float w[9] = {0.0f, -1.0f, 0.0f,
                        -1.0f, 4.0f, -1.0f,
                        0.0f, -1.0f, 0.0f};
    return make_3x3("laplacian", w, 0.0f);
}

ConvKernel make_identity() {
    const float w[9] = {0.0f, 0.0f, 0.0f,
                        0.0f, 1.0f, 0.0f,
                        0.0f, 0.0f, 0.0f};
    return make_3x3("identity", w, 0.0f);
}

bool build_kernel(const Options& opt, ConvKernel& out, std::string& error) {
    if (opt.radius < 0 || opt.radius > kMaxKernelRadius) {
        error = "radius must be between 0 and " + std::to_string(kMaxKernelRadius);
        return false;
    }

    switch (opt.filter) {
        case Filter::Box:
            out = make_box(opt.radius);
            return true;
        case Filter::Gaussian:
            out = make_gaussian(opt.radius, opt.sigma);
            return true;
        case Filter::Sharpen:
            out = make_sharpen(opt.radius, opt.sigma, opt.amount);
            return true;
        case Filter::Emboss:
            out = make_emboss();
            return true;
        case Filter::Laplacian:
            out = make_laplacian();
            return true;
        case Filter::Passthrough:
            out = make_identity();
            return true;
        default:
            error = std::string("filter '") + filter_name(opt.filter) + "' is not a convolution";
            return false;
    }
}

}  // namespace cif
