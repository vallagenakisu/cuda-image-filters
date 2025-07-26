cmake_minimum_required(VERSION 3.18)
project(cuda-image-filters LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  # A debug build of a GPU filter is slow enough to be misleading when you
  # look at the benchmark numbers, so default to Release.
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Which GPU architectures to generate code for.
#
# 'native' asks nvcc to detect the card in this machine, which is what you
# want for a build you are going to run here. It needs CUDA 11.5+ and
# CMake 3.24+; older setups get an explicit list covering Pascal through Ada.
# Override with -DCMAKE_CUDA_ARCHITECTURES=86 for a specific card.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
  else()
    set(CMAKE_CUDA_ARCHITECTURES 61 70 75 80 86 89)
  endif()
endif()

set(HOST_SOURCES
  src/main.cpp
  src/options.cpp
  src/image_io.cpp
  src/conv_kernel.cpp
  src/cpu_reference.cpp
  src/verify.cpp
)

set(CUDA_SOURCES
  src/cuda/device_info.cu
  src/cuda/grayscale.cu
  src/cuda/convolution.cu
  src/cuda/sobel.cu
  src/cuda/pointwise.cu
)

add_executable(cuda-filters ${HOST_SOURCES} ${CUDA_SOURCES})

target_include_directories(cuda-filters PRIVATE include)
# SYSTEM: stb is third-party and warns a lot under -Wall; those warnings are
# not ours to fix and would drown out the ones that are.
target_include_directories(cuda-filters SYSTEM PRIVATE third_party/stb)

target_compile_options(cuda-filters PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# --- tests -------------------------------------------------------------------
# Host-only, so they build and run on a machine with no GPU.
enable_testing()

add_executable(test_kernels tests/test_kernels.cpp src/conv_kernel.cpp src/options.cpp)
target_include_directories(test_kernels PRIVATE include)
add_test(NAME kernels COMMAND test_kernels)

# --- convenience -------------------------------------------------------------
add_custom_target(sample
  COMMAND cuda-filters --help
  COMMENT "Build finished. Run ./cuda-filters --help for usage."
)
