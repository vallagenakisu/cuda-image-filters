#pragma once

#include <string>

namespace cif {

enum class Filter {
    Passthrough,  // decode + encode only, useful to sanity-check image I/O
    Grayscale,
    Invert,
    Box,
    Gaussian,
    Sharpen,
    Emboss,
    Laplacian,
    Sobel,
    Tonemap,
    Custom,  // NxN matrix read from a .kernel file
};

enum class Backend {
    Cuda,
    Cpu,
};

// How a convolution should be carried out on the GPU. This is the knob the
// whole optimisation exercise turns on.
enum class ConvMethod {
    Auto,       // tiled when the radius fits in shared memory, else naive
    Naive,      // every tap re-read straight from global memory
    Tiled,      // block loads a tile + halo into shared memory first
    Separable,  // two 1D passes; only valid for separable kernels (box, gaussian)
};

struct Options {
    std::string input;
    std::string output = "out.png";

    Filter filter = Filter::Passthrough;
    std::string kernel_path;  // --kernel, required for Filter::Custom

    Backend backend = Backend::Cuda;
    ConvMethod method = ConvMethod::Auto;

    // Convolution parameters.
    int radius = 2;       // kernel is (2*radius + 1) square
    float sigma = 0.0f;   // gaussian; 0 means "derive from radius"
    float amount = 1.0f;  // sharpen strength

    // Tone mapping parameters.
    float exposure = 1.0f;
    float white = 4.0f;
    float gamma = 2.2f;

    // Launch configuration.
    int block_x = 16;
    int block_y = 16;

    int force_channels = 0;  // 0 = keep the file's channel count

    bool verify = false;  // run the CPU reference too and report the difference
    int bench = 0;        // >0: time this many iterations of every method
    bool list_devices = false;
    bool quiet = false;
};

// Parse argv. Returns false and fills `error` on a bad argument.
// Sets `wants_help` when -h/--help was passed, in which case nothing else
// in `out` is meaningful.
bool parse_args(int argc, char** argv, Options& out, bool& wants_help, std::string& error);

const char* filter_name(Filter f);
const char* method_name(ConvMethod m);

void print_usage(const char* argv0);

}  // namespace cif
