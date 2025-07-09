#pragma once

// Host-side declarations of the few CUDA helpers main.cpp needs.
//
// Deliberately free of <cuda_runtime.h>: main.cpp, options.cpp and
// image_io.cpp are plain C++ translation units compiled by the host compiler,
// and only the .cu files are handed to nvcc. Keeping the boundary here means
// the CPU backend can be built and tested on a machine with no CUDA at all.
namespace cif {

void print_device_info();
void require_cuda_device();

}  // namespace cif
