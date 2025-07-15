#include "conv_kernel.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

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

bool load_kernel_file(const std::string& path, ConvKernel& out, std::string& error) {
    std::ifstream file(path);
    if (!file) {
        error = "could not open kernel file '" + path + "'";
        return false;
    }

    ConvKernel k;
    k.name = "custom";
    bool normalize = false;
    std::vector<float> values;

    std::string line;
    int line_number = 0;
    while (std::getline(file, line)) {
        ++line_number;

        // Strip comments first so a '#' can follow real content on a line.
        const size_t hash = line.find('#');
        if (hash != std::string::npos) line.erase(hash);

        std::istringstream tokens(line);
        std::string token;
        while (tokens >> token) {
            if (token == "name") {
                if (!(tokens >> k.name)) {
                    error = path + ":" + std::to_string(line_number) + ": 'name' needs a value";
                    return false;
                }
            } else if (token == "bias") {
                if (!(tokens >> k.bias)) {
                    error = path + ":" + std::to_string(line_number) + ": 'bias' needs a number";
                    return false;
                }
            } else if (token == "normalize") {
                std::string flag;
                if (!(tokens >> flag)) {
                    error = path + ":" + std::to_string(line_number) + ": 'normalize' needs 0 or 1";
                    return false;
                }
                normalize = (flag == "1" || flag == "true" || flag == "yes");
            } else {
                // Anything else has to be a weight.
                char* end = nullptr;
                const float value = std::strtof(token.c_str(), &end);
                if (end == token.c_str() || *end != '\0') {
                    error = path + ":" + std::to_string(line_number) + ": '" + token +
                            "' is neither a directive nor a number";
                    return false;
                }
                values.push_back(value);
            }
        }
    }

    if (values.empty()) {
        error = path + ": no weights found";
        return false;
    }

    // The matrix has to be square with an odd side, otherwise there is no
    // single centre tap to align with the pixel being written.
    const int side = static_cast<int>(std::lround(std::sqrt(static_cast<double>(values.size()))));
    if (static_cast<size_t>(side) * side != values.size()) {
        error = path + ": " + std::to_string(values.size()) +
                " weights do not form a square matrix";
        return false;
    }
    if (side % 2 == 0) {
        error = path + ": matrix is " + std::to_string(side) + "x" + std::to_string(side) +
                ", but the side must be odd so there is a centre tap";
        return false;
    }

    k.radius = (side - 1) / 2;
    if (k.radius > kMaxKernelRadius) {
        error = path + ": radius " + std::to_string(k.radius) + " exceeds the limit of " +
                std::to_string(kMaxKernelRadius);
        return false;
    }
    k.weights = values;

    if (normalize) {
        const float total = k.sum();
        // A sum-zero matrix (an edge detector) cannot be normalised, and
        // dividing by ~0 would blow the weights up.
        if (std::fabs(total) < 1e-6f) {
            error = path + ": normalize requested but the weights sum to zero";
            return false;
        }
        for (float& w : k.weights) w /= total;
    }

    out = k;
    return true;
}

bool build_kernel(const Options& opt, ConvKernel& out, std::string& error) {
    if (opt.radius < 0 || opt.radius > kMaxKernelRadius) {
        error = "radius must be between 0 and " + std::to_string(kMaxKernelRadius);
        return false;
    }

    switch (opt.filter) {
        case Filter::Custom:
            return load_kernel_file(opt.kernel_path, out, error);
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
