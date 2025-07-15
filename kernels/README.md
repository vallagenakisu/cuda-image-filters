# Custom convolution kernels

Any file here can be fed to the tool:

```
cuda-filters -i photo.jpg -k kernels/outline.kernel -o outlined.png
```

Passing `--kernel` implies `--filter custom`, so there is nothing else to set.

## Format

- `#` starts a comment, to end of line.
- `name <text>` — shown in the log, purely cosmetic.
- `bias <number>` — added after the weighted sum. Sum-zero matrices need it if
  you want a mid-grey background instead of black.
- `normalize 0|1` — divide every weight by the sum of the weights. Use it for
  blurs so the image keeps its brightness. It is an error on a sum-zero matrix.
- Everything else is a weight. Line breaks are only for readability; the
  weights just have to add up to an odd perfect square (9, 25, 49, ...).

Borders clamp to the edge pixel and alpha is copied through untouched.

## Writing your own

Rules of thumb:

- weights summing to **1** preserve brightness (blur, sharpen, emboss)
- weights summing to **0** keep only edges and make flat areas black
  (Laplacian, outline, Sobel) — add `bias 128` to see negative values
- a large positive centre with a negative surround sharpens
- an asymmetric matrix gives a directional effect (emboss, motion blur)
