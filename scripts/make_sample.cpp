// Generates assets/sample.png, a test pattern chosen to make each filter's
// behaviour obvious rather than to look pretty:
//
//   - hard checkerboard edges      sobel and laplacian have something to find
//   - concentric rings             a blur that is not isotropic shows up here
//   - one-pixel lines              sharpen and blur act most visibly on these
//   - a smooth horizontal ramp     banding from lost precision is visible
//   - a few blown-out highlights   gives tonemap something to compress
//
// Build:  c++ -std=c++17 -Iinclude -isystem third_party/stb \
//             scripts/make_sample.cpp src/image_io.cpp -o make_sample

#include <cmath>
#include <cstdio>
#include <string>

#include "image_io.hpp"

namespace {

float clamp01(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

}  // namespace

int main(int argc, char** argv) {
    const std::string out = argc > 1 ? argv[1] : "assets/sample.png";
    const int width = argc > 2 ? std::atoi(argv[2]) : 1024;
    const int height = argc > 3 ? std::atoi(argv[3]) : 768;

    cif::Image img(width, height, 3);

    const float cx = width * 0.5f;
    const float cy = height * 0.5f;
    const float max_r = std::sqrt(cx * cx + cy * cy);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float fx = static_cast<float>(x) / static_cast<float>(width);
            const float dx = static_cast<float>(x) - cx;
            const float dy = static_cast<float>(y) - cy;
            const float r = std::sqrt(dx * dx + dy * dy);

            // Smooth horizontal ramp: any quantisation shows as vertical bands.
            float red = fx;
            float green = 0.35f + 0.35f * std::sin(r * 0.06f);  // concentric rings
            float blue = 1.0f - fx;

            // Hard-edged colour checkerboard over the left half.
            if (x < width / 2) {
                const bool dark = ((x / 64) + (y / 64)) % 2 == 0;
                if (dark) {
                    red *= 0.15f;
                    green *= 0.15f;
                    blue *= 0.15f;
                }
            }

            // One-pixel grid lines: the finest detail a filter can destroy.
            if (x % 37 == 0 || y % 41 == 0) {
                red = green = blue = 0.95f;
            }

            // A couple of blown-out highlights for the tone mapper to pull back.
            const float h1x = static_cast<float>(x) - width * 0.78f;
            const float h1y = static_cast<float>(y) - height * 0.28f;
            const float h1 = std::exp(-(h1x * h1x + h1y * h1y) / 2400.0f);
            const float h2x = static_cast<float>(x) - width * 0.62f;
            const float h2y = static_cast<float>(y) - height * 0.72f;
            const float h2 = std::exp(-(h2x * h2x + h2y * h2y) / 900.0f);
            const float glow = h1 + h2;
            red += glow;
            green += glow * 0.95f;
            blue += glow * 0.8f;

            // Gentle vignette so the frame is not uniform.
            const float vignette = 1.0f - 0.35f * (r / max_r);

            uint8_t* p = img.at(x, y);
            p[0] = static_cast<uint8_t>(clamp01(red * vignette) * 255.0f + 0.5f);
            p[1] = static_cast<uint8_t>(clamp01(green * vignette) * 255.0f + 0.5f);
            p[2] = static_cast<uint8_t>(clamp01(blue * vignette) * 255.0f + 0.5f);
        }
    }

    std::string error;
    if (!cif::save_image(out, img, error)) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }
    std::printf("wrote %s (%dx%d)\n", out.c_str(), width, height);
    return 0;
}
