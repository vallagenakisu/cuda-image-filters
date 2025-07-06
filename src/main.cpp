#include <cstdio>
#include <string>

#include "image.hpp"
#include "image_io.hpp"
#include "options.hpp"

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

    cif::Image input = cif::load_image(opt.input, opt.force_channels, error);
    if (input.empty()) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }
    if (!opt.quiet) {
        std::printf("loaded  %s  (%s)\n", opt.input.c_str(), input.describe().c_str());
    }

    if (opt.filter != cif::Filter::Passthrough) {
        std::fprintf(stderr, "error: filter '%s' is not wired up yet\n", cif::filter_name(opt.filter));
        return 3;
    }

    if (!cif::save_image(opt.output, input, error)) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }
    if (!opt.quiet) {
        std::printf("wrote   %s\n", opt.output.c_str());
    }
    return 0;
}
