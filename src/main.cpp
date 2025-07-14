#include <cstdio>
#include <string>

#include "conv_kernel.hpp"
#include "cpu_reference.hpp"
#include "cuda_entry.hpp"
#include "gpu_filters.hpp"
#include "image.hpp"
#include "image_io.hpp"
#include "options.hpp"

namespace {

void report(const cif::Timings& t) {
    std::printf("gpu     %-9s upload %.3f ms | kernel %.3f ms | download %.3f ms | total %.3f ms\n",
                t.method, t.upload_ms, t.kernel_ms, t.download_ms, t.total_ms());
}

}  // namespace

int main(int argc, char** argv) {
    cif::Options opt;
    bool wants_help = false;
    std::string error;

    if (!cif::parse_args(argc, argv, opt, wants_help, error)) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 2;
    }
    if (wants_help || argc == 1) {
        cif::print_usage(argv[0]);
        return wants_help ? 0 : 2;
    }
    if (opt.list_devices) {
        cif::print_device_info();
        return 0;
    }

    cif::Image input = cif::load_image(opt.input, opt.force_channels, error);
    if (input.empty()) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }
    if (!opt.quiet) {
        std::printf("loaded  %s  (%s)\n", opt.input.c_str(), input.describe().c_str());
    }

    cif::Image output;
    if (opt.backend == cif::Backend::Cpu) {
        // The reference path needs no GPU at all, which is what makes it
        // usable as the oracle and usable on a machine without CUDA.
        if (!cif::cpu::run(input, output, opt, error)) {
            std::fprintf(stderr, "error: %s\n", error.c_str());
            return 3;
        }
        if (!opt.quiet) std::printf("cpu     reference backend\n");
    } else {
        cif::require_cuda_device();
        cif::Timings t;
        switch (opt.filter) {
            case cif::Filter::Grayscale:
                t = cif::gpu::grayscale(input, output, opt);
                break;
            case cif::Filter::Passthrough:
            case cif::Filter::Box:
            case cif::Filter::Gaussian:
            case cif::Filter::Sharpen:
            case cif::Filter::Emboss:
            case cif::Filter::Laplacian: {
                cif::ConvKernel kernel;
                if (!cif::build_kernel(opt, kernel, error)) {
                    std::fprintf(stderr, "error: %s\n", error.c_str());
                    return 3;
                }
                if (!opt.quiet) std::printf("kernel  %s\n", kernel.describe().c_str());
                t = cif::gpu::convolve(input, output, kernel, opt);
                break;
            }
            default:
                std::fprintf(stderr, "error: filter '%s' has no CUDA path yet (try --backend cpu)\n",
                             cif::filter_name(opt.filter));
                return 3;
        }
        if (!opt.quiet) report(t);
    }

    if (!cif::save_image(opt.output, output, error)) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }
    if (!opt.quiet) {
        std::printf("wrote   %s  (%s)\n", opt.output.c_str(), output.describe().c_str());
    }
    return 0;
}
