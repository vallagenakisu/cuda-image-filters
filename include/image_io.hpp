#pragma once

#include <string>

#include "image.hpp"

namespace cif {

// Decode an image from disk. Supports everything stb_image supports: PNG, JPG,
// BMP, TGA, GIF, PSD, PNM.
//
// desired_channels: 0 keeps the file's own channel count, or force 1/3/4.
// On failure returns an empty Image and fills `error`.
Image load_image(const std::string& path, int desired_channels, std::string& error);

// Encode to disk. The format is chosen from the file extension:
// .png, .jpg/.jpeg, .bmp, .tga. jpg_quality is only used for JPEG.
bool save_image(const std::string& path, const Image& img, std::string& error, int jpg_quality = 95);

}  // namespace cif
