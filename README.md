# cuda-image-filters

Real-time image processing filters written in CUDA C++.

Load an image, run a parallel filter on the GPU, write the result back out.
The point of the project is to learn how GPU memory and thread indexing
actually work, so every filter is implemented twice: once the naive way
(straight global-memory reads) and once optimised (shared-memory tiling).

## Planned filters

- Grayscale conversion
- Box blur / Gaussian blur
- Sobel edge detection
- Sharpen, emboss, Laplacian
- Arbitrary user-supplied convolution kernels
- Reinhard tone mapping

## Status

Work in progress.
