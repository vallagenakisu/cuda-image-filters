// Unit tests for the convolution matrix builders.
//
// No framework: a check helper and an exit code is enough, and it keeps the
// project buildable with nothing but nvcc and a C++ compiler.
//
//   c++ -std=c++17 -Iinclude tests/test_kernels.cpp src/conv_kernel.cpp \
//       src/options.cpp -o test_kernels && ./test_kernels

#include "conv_kernel.hpp"
#include <cstdio>
#include <cmath>
using namespace cif;
static int fails = 0;
static void chk(bool c, const char* m){ std::printf("%s %s\n", c?"pass":"FAIL", m); if(!c) fails++; }
int main(){
  auto b = make_box(2);
  chk(b.size()==5 && b.tap_count()==25, "box radius 2 is 5x5");
  chk(std::fabs(b.sum()-1.0f)<1e-5f, "box normalised to 1");
  chk(b.separable, "box separable");
  chk(std::fabs(b.at(0,0)-1.0f/25.0f)<1e-6f, "box centre tap = 1/25");

  auto g = make_gaussian(3, 0.0f);
  chk(std::fabs(g.sum()-1.0f)<1e-5f, "gaussian normalised to 1");
  chk(g.at(0,0) > g.at(1,0) && g.at(1,0) > g.at(3,0), "gaussian decays from centre");
  chk(std::fabs(g.at(1,2)-g.at(-1,-2))<1e-7f, "gaussian symmetric");
  // outer product must match the 1D taps exactly
  bool sep_ok=true;
  for(int y=-3;y<=3;y++) for(int x=-3;x<=3;x++)
    if(std::fabs(g.at(x,y)-g.horizontal[x+3]*g.vertical[y+3])>1e-7f) sep_ok=false;
  chk(sep_ok, "gaussian 2D == outer product of 1D taps");

  auto s = make_sharpen(2, 0.0f, 1.0f);
  chk(std::fabs(s.sum()-1.0f)<1e-4f, "unsharp mask preserves brightness (sum 1)");
  chk(s.at(0,0) > 1.0f, "unsharp centre tap boosted");
  chk(s.at(1,0) < 0.0f, "unsharp surround negative");
  chk(!s.separable, "unsharp not separable");

  chk(std::fabs(make_emboss().sum()-1.0f)<1e-6f, "emboss sums to 1");
  chk(std::fabs(make_laplacian().sum())<1e-6f, "laplacian sums to 0");
  chk(std::fabs(make_identity().at(0,0)-1.0f)<1e-6f, "identity centre = 1");

  // amount 0 must reduce to identity
  auto s0 = make_sharpen(2, 0.0f, 0.0f);
  bool id_ok = std::fabs(s0.at(0,0)-1.0f)<1e-6f && std::fabs(s0.at(1,1))<1e-6f;
  chk(id_ok, "sharpen amount 0 == identity");

  Options o; o.filter=Filter::Sobel; ConvKernel k; std::string e;
  chk(!build_kernel(o,k,e) && !e.empty(), "build_kernel rejects sobel");
  std::printf("\n%s\n", fails? "FAILURES" : "all kernel tests pass");
  return fails?1:0;
}
