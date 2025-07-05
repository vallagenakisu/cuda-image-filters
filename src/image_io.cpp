#include "image_io.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>

// stb wants exactly one translation unit to pull in the implementations.
#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace cif {
namespace {

std::string lower_extension(const std::string& path) {
    const size_t dot = path.find_last_of('.');
    if (dot == std::string::npos) return "";
    std::string ext = path.substr(dot + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext;
}

}  // namespace

Image load_image(const std::string& path, int desired_channels, std::string& error) {
    int w = 0, h = 0, file_channels = 0;
    uint8_t* pixels = stbi_load(path.c_str(), &w, &h, &file_channels, desired_channels);
    if (pixels == nullptr) {
        error = "could not read '" + path + "': " + (stbi_failure_reason() ? stbi_failure_reason() : "unknown error");
        return Image{};
    }

    // When desired_channels is non-zero stb converts for us, so the buffer
    // really has that many channels even though file_channels says otherwise.
    const int channels = desired_channels > 0 ? desired_channels : file_channels;

    Image img(w, h, channels);
    std::memcpy(img.data.data(), pixels, img.byte_size());
    stbi_image_free(pixels);
    return img;
}

bool save_image(const std::string& path, const Image& img, std::string& error, int jpg_quality) {
    if (img.empty()) {
        error = "refusing to write an empty image to '" + path + "'";
        return false;
    }

    const std::string ext = lower_extension(path);
    const int stride = static_cast<int>(img.row_bytes());
    int ok = 0;

    if (ext == "png" || ext.empty()) {
        ok = stbi_write_png(path.c_str(), img.width, img.height, img.channels, img.data.data(), stride);
    } else if (ext == "jpg" || ext == "jpeg") {
        ok = stbi_write_jpg(path.c_str(), img.width, img.height, img.channels, img.data.data(), jpg_quality);
    } else if (ext == "bmp") {
        ok = stbi_write_bmp(path.c_str(), img.width, img.height, img.channels, img.data.data());
    } else if (ext == "tga") {
        ok = stbi_write_tga(path.c_str(), img.width, img.height, img.channels, img.data.data());
    } else {
        error = "unsupported output extension '." + ext + "' (use png, jpg, bmp or tga)";
        return false;
    }

    if (!ok) {
        error = "could not write '" + path + "'";
        return false;
    }
    return true;
}

}  // namespace cif
