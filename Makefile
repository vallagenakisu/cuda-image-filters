# Plain make fallback for when you would rather not install CMake.
# CMake is the supported path; this exists because `make` is one command.
#
#   make                 build ./cuda-filters
#   make test            build and run the host-only unit tests
#   make CUDA_ARCH=86    target one specific architecture
#   make clean

NVCC      ?= nvcc
CXX       ?= g++
PYTHON    ?= python3
CUDA_ARCH ?= native

BIN     := cuda-filters
BUILD   := build

INCLUDES  := -Iinclude -isystem third_party/stb
CXXFLAGS  := -std=c++17 -O3 -Wall -Wextra $(INCLUDES)

# -arch=native detects the card in this machine and needs CUDA 11.5 or newer.
# On an older toolkit, pass an explicit number: make CUDA_ARCH=75
NVCCFLAGS := -std=c++17 -O3 -arch=$(CUDA_ARCH) --expt-relaxed-constexpr $(INCLUDES)

HOST_SRC := src/main.cpp src/options.cpp src/image_io.cpp \
            src/conv_kernel.cpp src/cpu_reference.cpp src/verify.cpp
CUDA_SRC := src/cuda/device_info.cu src/cuda/grayscale.cu src/cuda/convolution.cu \
            src/cuda/sobel.cu src/cuda/pointwise.cu

HOST_OBJ := $(patsubst %.cpp,$(BUILD)/%.o,$(HOST_SRC))
CUDA_OBJ := $(patsubst %.cu,$(BUILD)/%.o,$(CUDA_SRC))

.PHONY: all test sample demo ui clean

all: $(BIN)

# nvcc does the link so the CUDA runtime is pulled in without naming it.
$(BIN): $(HOST_OBJ) $(CUDA_OBJ)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(BUILD)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/%.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# The tests are host-only on purpose: they run on a machine with no GPU.
test: $(BUILD)/test_kernels
	./$(BUILD)/test_kernels
	$(PYTHON) tests/test_gui.py

$(BUILD)/test_kernels: tests/test_kernels.cpp src/conv_kernel.cpp src/options.cpp
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Regenerate the test pattern in assets/.
sample: $(BUILD)/make_sample
	./$(BUILD)/make_sample assets/sample.png 1024 768

$(BUILD)/make_sample: scripts/make_sample.cpp src/image_io.cpp
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Render every filter into out/.
demo: $(BIN)
	./scripts/demo.sh

# Local web UI. Standard library only, so there is nothing to install.
ui: $(BIN)
	$(PYTHON) gui/server.py

clean:
	rm -rf $(BUILD) $(BIN)
