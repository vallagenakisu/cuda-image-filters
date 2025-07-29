# Notes on the optimisation work

Working notes kept while building this. They are the "why", which does not fit
in code comments without drowning the code.

## The problem with the naive convolution

One thread per output pixel, every tap read straight from global memory:

```
for ky in -r..r:
  for kx in -r..r:
    acc += weight[ky][kx] * input[y + ky][x + kx]
```

For a K x K kernel that is K² global loads per pixel. At radius 4 (9x9) that is
81 loads to produce one output byte.

The redundancy across threads is the real issue. The thread at (x, y) and the
thread at (x+1, y) read 72 of the same 81 pixels. Every pixel in the image gets
read up to K² times by different threads.

L1 and L2 absorb a good share of that, which is why the naive version is not a
disaster. But the traffic is real, and the kernel is memory bound, so traffic
is the thing to cut.

## Shared memory tiling

Shared memory is per-block, on-chip and roughly an order of magnitude lower
latency than global. The idea:

1. The block cooperatively copies the input region it needs into shared memory.
2. `__syncthreads()`.
3. Every thread convolves out of shared memory.

The region is bigger than the block's output region. A 16x16 block whose
threads each reach `r` pixels in every direction needs a `(16 + 2r) x (16 + 2r)`
window. The extra ring is the **halo**.

```
        <-------- 16 + 2r -------->
        +-------------------------+
        |  halo                   |
        |    +-----------------+  |   ^
        |    |                 |  |   |
        |    |   16x16 output  |  |  16 + 2r
        |    |                 |  |   |
        |    +-----------------+  |   v
        |                         |
        +-------------------------+
```

Loads per block, radius 4, 16x16 block:

| | loads |
|---|---|
| naive | 256 threads x 81 taps = 20736 global reads |
| tiled | 24 x 24 = 576 global reads, then 20736 *shared* reads |

36x less global traffic. The arithmetic is unchanged.

### Three things that bite

**The bounds guard must come after `__syncthreads()`.** The obvious thing is:

```cuda
if (x >= width || y >= height) return;   // WRONG here
... load tile ...
__syncthreads();
```

Threads that return early never reach the barrier, while the rest of the block
waits for them. `__syncthreads()` requires every thread in the block to arrive.
This is undefined behaviour and in practice it hangs. Those threads also had
halo cells to load. Load first, sync, *then* guard the output write.

**The halo is bigger than it looks.** With 16x16 and r=7, the tile is 30x30 =
900 cells for 256 threads. More than half the loaded data is halo. The
efficiency of tiling falls as the radius grows relative to the block, which is
why `--method auto` stops using it when the tile no longer fits in half the
shared memory budget.

**Striding the cooperative load.** There are more tile cells than threads, so
each thread loads several. Do it like this:

```cuda
for (int ty = threadIdx.y; ty < tile_h; ty += blockDim.y)
  for (int tx = threadIdx.x; tx < tile_w; tx += blockDim.x)
```

and consecutive threads stay on consecutive addresses, so the loads coalesce.
Giving each thread a contiguous chunk instead scatters the warp across the
image and throws away the bandwidth tiling was supposed to save.

### Why the tile is float and not uint8

It costs 4x the shared memory. In exchange the byte-to-float conversion leaves
the inner loop, and shared memory is banked in 4-byte words — byte-wide
accesses make threads in a warp collide on the same bank and serialise.

## Separability beats tiling

A separable kernel is the outer product of two 1D vectors, so convolving by the
row vector and then the column vector equals convolving by the full matrix:

| radius | 2D taps | separable taps |
|---:|---:|---:|
| 1 | 9 | 6 |
| 4 | 81 | 18 |
| 8 | 289 | 34 |
| 16 | 1089 | 66 |

That is an algorithmic win and it grows with the radius, so it beats a constant
factor of memory optimisation. Gaussian and box blurs are separable. Unsharp
mask, emboss and Laplacian are not.

The intermediate buffer between the two passes is float. Rounding back to bytes
in between throws away the precision the second pass needs and shows up as
banding in a strong blur.

The honest conclusion from having both: **the best memory optimisation loses to
a better algorithm.** Tiling is still worth understanding — it is the technique
that applies when there is no algorithmic shortcut.

## Constant memory for the weights

Every thread in a warp reads the same tap at the same instant. That is a
broadcast, which is exactly what `__constant__` is for: one fetch serves the
whole warp. Putting the weights in global memory instead means 32 loads where
one would do.

The bank is 64 KiB, which caps the matrix at 65x65 here.

## What "memory bound" looks like when you measure it

`--list-devices` prints the theoretical peak bandwidth
(`2 x memory clock x bus width / 8`). Compare it to the effective GB/s from
`--bench`.

The pointwise filters (invert, grayscale, tonemap) get reasonably close to
peak: one read, one write, almost no arithmetic. The convolutions do not, and
the gap is the redundant traffic. That gap is the thing all of this is about.

## Where the time actually goes

On one 1024x768 image, the `cudaMemcpy` in each direction can cost more than
the kernel. A GPU filter is not automatically faster than a CPU one — it is
faster when the work per byte transferred is high enough, or when the data is
already on the device and stays there.

This is why `--bench` reports kernel time separately from the copies, and why
the obvious next step for this project would be to keep a video frame resident
on the device across filters rather than round-tripping each one.

## Still to try

- shared-memory tiling for the Sobel kernel (currently naive)
- a log-average luminance reduction so tone mapping picks its own exposure
  instead of taking it as a flag — a real parallel reduction, which this
  project does not otherwise have
- `__ldg` / texture objects for the naive path, to see how much of the gap the
  read-only cache closes on its own
- processing 4 pixels per thread with `uchar4` loads, so each thread moves 32
  bits instead of 8
