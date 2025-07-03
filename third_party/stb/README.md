# third_party/stb

Vendored single-header image libraries by Sean Barrett (public domain / MIT).

- `stb_image.h`       - decodes JPG, PNG, BMP, TGA, GIF, PSD, HDR, PIC, PNM
- `stb_image_write.h` - encodes PNG, BMP, TGA, JPG, HDR

Source: https://github.com/nothings/stb

They are vendored so the project builds with nothing but a CUDA toolkit
installed - no package manager, no `find_package`, no network at build time.
Exactly one translation unit (`src/image_io.cpp`) defines
`STB_IMAGE_IMPLEMENTATION` / `STB_IMAGE_WRITE_IMPLEMENTATION`.
