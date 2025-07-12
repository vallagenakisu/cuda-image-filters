#include "cpu_reference.hpp"

#include <cmath>

#include "pixel_ops.hpp"

namespace cif {
namespace cpu {

void grayscale(const Image& in, Image& out) {
    out = Image(in.width, in.height, 1);
    for (int y = 0; y < in.height; ++y) {
        for (int x = 0; x < in.width; ++x) {
            const uint8_t* src = in.at(x, y);
            const float value = in.channels >= 3
                                    ? luminance(static_cast<float>(src[0]), static_cast<float>(src[1]),
                                                static_cast<float>(src[2]))
                                    : static_cast<float>(src[0]);
            *out.at(x, y) = clamp_to_byte(value);
        }
    }
}

void invert(const Image& in, Image& out) {
    out = Image(in.width, in.height, in.channels);
    for (int y = 0; y < in.height; ++y) {
        for (int x = 0; x < in.width; ++x) {
            const uint8_t* src = in.at(x, y);
            uint8_t* dst = out.at(x, y);
            for (int c = 0; c < in.channels; ++c) {
                // Alpha is copied, not inverted: inverting it turns opaque
                // pixels transparent, which is never what anyone wants.
                dst[c] = is_alpha_channel(c, in.channels) ? src[c]
                                                          : static_cast<uint8_t>(255 - src[c]);
            }
        }
    }
}

void convolve(const Image& in, Image& out, const ConvKernel& kernel) {
    out = Image(in.width, in.height, in.channels);
    const int r = kernel.radius;
    const int max_x = in.width - 1;
    const int max_y = in.height - 1;

    for (int y = 0; y < in.height; ++y) {
        for (int x = 0; x < in.width; ++x) {
            uint8_t* dst = out.at(x, y);

            for (int c = 0; c < in.channels; ++c) {
                if (is_alpha_channel(c, in.channels)) {
                    dst[c] = in.at(x, y)[c];
                    continue;
                }

                float acc = kernel.bias;
                for (int ky = -r; ky <= r; ++ky) {
                    const int sy = clamp_coord(y + ky, max_y);
                    for (int kx = -r; kx <= r; ++kx) {
                        const int sx = clamp_coord(x + kx, max_x);
                        acc += kernel.at(kx, ky) * static_cast<float>(in.at(sx, sy)[c]);
                    }
                }
                dst[c] = clamp_to_byte(acc);
            }
        }
    }
}

void sobel(const Image& in, Image& out) {
    // Sobel is defined on a scalar field, so colour input is reduced to luma
    // first. The output is the gradient magnitude, one channel.
    out = Image(in.width, in.height, 1);

    static const float gx_taps[9] = {-1.0f, 0.0f, 1.0f,
                                     -2.0f, 0.0f, 2.0f,
                                     -1.0f, 0.0f, 1.0f};
    static const float gy_taps[9] = {-1.0f, -2.0f, -1.0f,
                                     0.0f, 0.0f, 0.0f,
                                     1.0f, 2.0f, 1.0f};

    const int max_x = in.width - 1;
    const int max_y = in.height - 1;

    for (int y = 0; y < in.height; ++y) {
        for (int x = 0; x < in.width; ++x) {
            float gx = 0.0f;
            float gy = 0.0f;

            for (int ky = -1; ky <= 1; ++ky) {
                const int sy = clamp_coord(y + ky, max_y);
                for (int kx = -1; kx <= 1; ++kx) {
                    const int sx = clamp_coord(x + kx, max_x);
                    const uint8_t* p = in.at(sx, sy);
                    const float l = in.channels >= 3
                                        ? luminance(static_cast<float>(p[0]), static_cast<float>(p[1]),
                                                    static_cast<float>(p[2]))
                                        : static_cast<float>(p[0]);
                    const int tap = (ky + 1) * 3 + (kx + 1);
                    gx += gx_taps[tap] * l;
                    gy += gy_taps[tap] * l;
                }
            }

            *out.at(x, y) = clamp_to_byte(std::sqrt(gx * gx + gy * gy));
        }
    }
}

void tonemap(const Image& in, Image& out, float exposure, float white, float gamma) {
    out = Image(in.width, in.height, in.channels);

    const float inv_gamma = 1.0f / gamma;
    const float white_sq = white * white;
    const int colour_channels = in.channels >= 3 ? 3 : 1;

    for (int y = 0; y < in.height; ++y) {
        for (int x = 0; x < in.width; ++x) {
            const uint8_t* src = in.at(x, y);
            uint8_t* dst = out.at(x, y);

            // Undo the display gamma to get roughly linear light, then expose.
            float linear[3] = {0.0f, 0.0f, 0.0f};
            for (int c = 0; c < colour_channels; ++c) {
                linear[c] = std::pow(static_cast<float>(src[c]) / 255.0f, gamma) * exposure;
            }

            const float l = colour_channels == 3 ? luminance(linear[0], linear[1], linear[2])
                                                 : linear[0];

            // Extended Reinhard: compresses the highlights towards `white`
            // while leaving the shadows nearly untouched.
            float scale = 0.0f;
            if (l > 1e-6f) {
                const float mapped = l * (1.0f + l / white_sq) / (1.0f + l);
                scale = mapped / l;
            }

            for (int c = 0; c < colour_channels; ++c) {
                const float v = std::pow(linear[c] * scale, inv_gamma);
                dst[c] = clamp_to_byte(v * 255.0f);
            }
            if (in.channels == 4) dst[3] = src[3];
        }
    }
}

bool run(const Image& in, Image& out, const Options& opt, std::string& error) {
    switch (opt.filter) {
        case Filter::Passthrough:
            out = in;
            return true;
        case Filter::Grayscale:
            grayscale(in, out);
            return true;
        case Filter::Invert:
            invert(in, out);
            return true;
        case Filter::Sobel:
            sobel(in, out);
            return true;
        case Filter::Tonemap:
            tonemap(in, out, opt.exposure, opt.white, opt.gamma);
            return true;
        default: {
            ConvKernel kernel;
            if (!build_kernel(opt, kernel, error)) return false;
            convolve(in, out, kernel);
            return true;
        }
    }
}

}  // namespace cpu
}  // namespace cif
