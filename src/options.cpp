#include "options.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace cif {
namespace {

struct FilterEntry {
    const char* name;
    Filter value;
};

const FilterEntry kFilters[] = {
    {"passthrough", Filter::Passthrough},
    {"grayscale", Filter::Grayscale},
    {"gray", Filter::Grayscale},
    {"invert", Filter::Invert},
    {"box", Filter::Box},
    {"gaussian", Filter::Gaussian},
    {"blur", Filter::Gaussian},
    {"sharpen", Filter::Sharpen},
    {"emboss", Filter::Emboss},
    {"laplacian", Filter::Laplacian},
    {"sobel", Filter::Sobel},
    {"edges", Filter::Sobel},
    {"tonemap", Filter::Tonemap},
    {"custom", Filter::Custom},
};

bool parse_filter(const std::string& s, Filter& out) {
    for (const FilterEntry& e : kFilters) {
        if (s == e.name) {
            out = e.value;
            return true;
        }
    }
    return false;
}

bool parse_method(const std::string& s, ConvMethod& out) {
    if (s == "auto") { out = ConvMethod::Auto; return true; }
    if (s == "naive") { out = ConvMethod::Naive; return true; }
    if (s == "tiled" || s == "shared") { out = ConvMethod::Tiled; return true; }
    if (s == "separable") { out = ConvMethod::Separable; return true; }
    return false;
}

// Parse "16x16" or plain "16" (square) into a block size.
bool parse_block(const std::string& s, int& bx, int& by) {
    const size_t x = s.find('x');
    if (x == std::string::npos) {
        bx = by = std::atoi(s.c_str());
    } else {
        bx = std::atoi(s.substr(0, x).c_str());
        by = std::atoi(s.substr(x + 1).c_str());
    }
    return bx > 0 && by > 0 && bx * by <= 1024;
}

// Fetch the value that follows a flag, or report a missing argument.
bool take_value(int argc, char** argv, int& i, const char* flag, std::string& value, std::string& error) {
    if (i + 1 >= argc) {
        error = std::string("missing value after ") + flag;
        return false;
    }
    value = argv[++i];
    return true;
}

}  // namespace

const char* filter_name(Filter f) {
    switch (f) {
        case Filter::Passthrough: return "passthrough";
        case Filter::Grayscale: return "grayscale";
        case Filter::Invert: return "invert";
        case Filter::Box: return "box";
        case Filter::Gaussian: return "gaussian";
        case Filter::Sharpen: return "sharpen";
        case Filter::Emboss: return "emboss";
        case Filter::Laplacian: return "laplacian";
        case Filter::Sobel: return "sobel";
        case Filter::Tonemap: return "tonemap";
        case Filter::Custom: return "custom";
    }
    return "unknown";
}

const char* method_name(ConvMethod m) {
    switch (m) {
        case ConvMethod::Auto: return "auto";
        case ConvMethod::Naive: return "naive";
        case ConvMethod::Tiled: return "tiled";
        case ConvMethod::Separable: return "separable";
    }
    return "unknown";
}

bool parse_args(int argc, char** argv, Options& out, bool& wants_help, std::string& error) {
    wants_help = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        std::string value;

        if (arg == "-h" || arg == "--help") {
            wants_help = true;
            return true;
        } else if (arg == "--list-devices") {
            out.list_devices = true;
        } else if (arg == "-i" || arg == "--input") {
            if (!take_value(argc, argv, i, "--input", out.input, error)) return false;
        } else if (arg == "-o" || arg == "--output") {
            if (!take_value(argc, argv, i, "--output", out.output, error)) return false;
        } else if (arg == "-f" || arg == "--filter") {
            if (!take_value(argc, argv, i, "--filter", value, error)) return false;
            if (!parse_filter(value, out.filter)) {
                error = "unknown filter '" + value + "' (try --help)";
                return false;
            }
        } else if (arg == "-k" || arg == "--kernel") {
            if (!take_value(argc, argv, i, "--kernel", out.kernel_path, error)) return false;
            out.filter = Filter::Custom;  // passing a matrix implies a custom filter
        } else if (arg == "--backend") {
            if (!take_value(argc, argv, i, "--backend", value, error)) return false;
            if (value == "cuda" || value == "gpu") {
                out.backend = Backend::Cuda;
            } else if (value == "cpu") {
                out.backend = Backend::Cpu;
            } else {
                error = "unknown backend '" + value + "' (cuda or cpu)";
                return false;
            }
        } else if (arg == "--method") {
            if (!take_value(argc, argv, i, "--method", value, error)) return false;
            if (!parse_method(value, out.method)) {
                error = "unknown method '" + value + "' (auto, naive, tiled or separable)";
                return false;
            }
        } else if (arg == "--naive") {
            out.method = ConvMethod::Naive;
        } else if (arg == "--tiled") {
            out.method = ConvMethod::Tiled;
        } else if (arg == "-r" || arg == "--radius") {
            if (!take_value(argc, argv, i, "--radius", value, error)) return false;
            out.radius = std::atoi(value.c_str());
        } else if (arg == "-s" || arg == "--sigma") {
            if (!take_value(argc, argv, i, "--sigma", value, error)) return false;
            out.sigma = static_cast<float>(std::atof(value.c_str()));
        } else if (arg == "--amount") {
            if (!take_value(argc, argv, i, "--amount", value, error)) return false;
            out.amount = static_cast<float>(std::atof(value.c_str()));
        } else if (arg == "--exposure") {
            if (!take_value(argc, argv, i, "--exposure", value, error)) return false;
            out.exposure = static_cast<float>(std::atof(value.c_str()));
        } else if (arg == "--white") {
            if (!take_value(argc, argv, i, "--white", value, error)) return false;
            out.white = static_cast<float>(std::atof(value.c_str()));
        } else if (arg == "--gamma") {
            if (!take_value(argc, argv, i, "--gamma", value, error)) return false;
            out.gamma = static_cast<float>(std::atof(value.c_str()));
        } else if (arg == "--block") {
            if (!take_value(argc, argv, i, "--block", value, error)) return false;
            if (!parse_block(value, out.block_x, out.block_y)) {
                error = "bad --block '" + value + "' (expected NxM with N*M <= 1024)";
                return false;
            }
        } else if (arg == "--channels") {
            if (!take_value(argc, argv, i, "--channels", value, error)) return false;
            out.force_channels = std::atoi(value.c_str());
            if (out.force_channels != 0 && out.force_channels != 1 && out.force_channels != 3 &&
                out.force_channels != 4) {
                error = "--channels must be 0, 1, 3 or 4";
                return false;
            }
        } else if (arg == "--verify") {
            out.verify = true;
        } else if (arg == "--bench") {
            if (!take_value(argc, argv, i, "--bench", value, error)) return false;
            out.bench = std::atoi(value.c_str());
            if (out.bench <= 0) {
                error = "--bench needs a positive iteration count";
                return false;
            }
        } else if (arg == "-q" || arg == "--quiet") {
            out.quiet = true;
        } else if (!arg.empty() && arg[0] == '-') {
            error = "unknown option '" + arg + "' (try --help)";
            return false;
        } else if (out.input.empty()) {
            out.input = arg;  // allow the input path as a bare positional
        } else {
            error = "unexpected extra argument '" + arg + "'";
            return false;
        }
    }

    if (out.list_devices) return true;

    if (out.input.empty()) {
        error = "no input image (pass --input <path>)";
        return false;
    }
    if (out.filter == Filter::Custom && out.kernel_path.empty()) {
        error = "--filter custom needs --kernel <file>";
        return false;
    }
    if (out.radius < 0 || out.radius > 32) {
        error = "--radius must be between 0 and 32";
        return false;
    }
    return true;
}

void print_usage(const char* argv0) {
    std::printf(
        "cuda-image-filters - GPU image filtering\n"
        "\n"
        "usage: %s --input <image> [--output <image>] [--filter <name>] [options]\n"
        "\n"
        "filters:\n"
        "  passthrough        decode and re-encode only (checks image I/O)\n"
        "  grayscale          luminance conversion\n"
        "  invert             per-pixel negative\n"
        "  box                box blur, radius given by --radius\n"
        "  gaussian           gaussian blur, --radius and --sigma\n"
        "  sharpen            unsharp mask, strength given by --amount\n"
        "  emboss             directional emboss\n"
        "  laplacian          second-derivative edges\n"
        "  sobel              sobel gradient magnitude (writes 1 channel)\n"
        "  tonemap            reinhard tone mapping, --exposure/--white/--gamma\n"
        "  custom             arbitrary NxN matrix from --kernel <file>\n"
        "\n"
        "options:\n"
        "  -i, --input PATH     source image (png, jpg, bmp, tga, ...)\n"
        "  -o, --output PATH    destination image, default out.png\n"
        "  -f, --filter NAME    filter to apply, default passthrough\n"
        "  -k, --kernel PATH    convolution matrix file; implies --filter custom\n"
        "  -r, --radius N       convolution radius, kernel is (2N+1) square\n"
        "  -s, --sigma F        gaussian sigma; 0 derives it from the radius\n"
        "      --amount F       sharpen strength, default 1.0\n"
        "      --exposure F     tonemap exposure multiplier, default 1.0\n"
        "      --white F        tonemap white point, default 4.0\n"
        "      --gamma F        tonemap gamma, default 2.2\n"
        "      --backend NAME   cuda (default) or cpu\n"
        "      --method NAME    auto (default), naive, tiled or separable\n"
        "      --naive          shorthand for --method naive\n"
        "      --tiled          shorthand for --method tiled\n"
        "      --block NxM      CUDA block dimensions, default 16x16\n"
        "      --channels N     force 1, 3 or 4 channels on load\n"
        "      --verify         also run the CPU reference and report the diff\n"
        "      --bench N        time N iterations of every available method\n"
        "      --list-devices   print the CUDA devices found and exit\n"
        "  -q, --quiet          only print errors\n"
        "  -h, --help           this message\n"
        "\n"
        "examples:\n"
        "  %s -i photo.jpg -f gaussian -r 5 -s 2.0 -o blurred.png\n"
        "  %s -i photo.jpg -f sobel -o edges.png\n"
        "  %s -i photo.jpg -k kernels/emboss.kernel -o embossed.png\n"
        "  %s -i photo.jpg -f gaussian -r 7 --bench 100\n",
        argv0, argv0, argv0, argv0, argv0);
}

}  // namespace cif
