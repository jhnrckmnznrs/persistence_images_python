# Persistence Images in Python

A standalone Python package for converting persistence diagrams into fixed-length numerical features. It supports NumPy computation, shared Gaussian bases for quantized data, and optional error-controlled algorithms for large datasets. It uses the same mathematical conventions and CSV format as the companion [Rust implementation](https://github.com/jhnrckmnznrs/persistence_images).

## Install

Requires Python 3.10 or newer.

```bash
python -m pip install .
```

To use the optional fast and FFT backends and to run the tests:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

Only NumPy is needed for the standard implementation. The optional `fast` extra installs Numba; the `grid` extra installs Numba and SciPy.

## Command line

Each input CSV is headerless, with at least two columns containing **birth and death** values:

```text
0.12,1.45
0.30,2.10
```

The spacing CSV has a header, the source image filename in its first column, and positive voxel spacing in its **third** column:

```text
filename,notes,spacing_micro
image1.tif,example,0.86
image2.tif,example,1.02
```

A diagram file named `image1.csv` looks up `image1.tif` in the spacing table. Missing spacing entries default to `1.0`. CSV files are processed recursively in sorted path order.

```bash
persistence-images \
  --input ./data/diagrams \
  --spacing ./data/spacing.csv \
  --output ./output/images.csv \
  --nx 128 --ny 128 \
  --sigma-x 0.2 --sigma-y 0.2 \
  --engine auto
```

You can also run `python -m persistence_images`. The output directory must already exist. The output CSV contains no header; each row starts with the input filename stem, followed by the flattened persistence-image values. The flag `--remove-h0` removes one dominant component according to the Rust project's selection rule. Use `--diagram-suffix` to select diagram files and `--stream` for a two-pass input workflow that reduces memory use.

For large, continuous-valued diagrams, the one-FFT interpolation algorithm is available with an **explicit approximation tolerance**:

```bash
persistence-images \
  --input ./data/diagrams --spacing ./data/spacing.csv \
  --output ./output/images.csv \
  --nx 256 --ny 256 \
  --sigma-x 0.2 --sigma-y 0.2 \
  --engine interp_fft --absolute-error 1e-8
```

This engine requires the `grid` extra. If the requested interpolation tolerance cannot be certified with the supported stencil widths, it raises an error rather than silently using an uncertified approximation.

## Python API

```python
import numpy as np
from persistence_images import Bounds, compute_bounds, persistence_image

# API coordinates are (birth, persistence), where persistence = death - birth.
diagram = np.array([[0.1, 0.8], [0.4, 0.3], [0.6, 0.5]])
bounds = compute_bounds([diagram])

vector = persistence_image(
    diagram,
    bounds,
    nx=64,
    ny=48,
    sigma_x=0.2,
    sigma_y=0.25,
    engine="auto",
)
image = vector.reshape(64, 48)
```

The library also exposes `read_diagram`, `load_dataset`, `batch_persistence_images`, `SharedGaussianBasis`, and `persistence_image_reference` for testing.

## Mathematical convention

For a point with birth `b` and death `d`, define persistence `p = d - b`. The point contribution at pixel center `(x_i, y_j)` is

\[
I_{ij} = \sum_k \exp\left[-\frac{(x_i-b_k)^2}{2\sigma_x^2}
                         -\frac{(y_j-p_k)^2}{2\sigma_y^2}\right].
\]

The Gaussian kernel is **unnormalized and unweighted**. The library evaluates the sum at pixel centers; it does not integrate Gaussian mass over pixels or normalize the final feature vector. Voxel spacing multiplies both diagram coordinates and Gaussian bandwidths. Pixel bounds are computed globally from spacing-scaled diagrams. Images flatten in x-major order: pixel `(i, j)` has index `i * ny + j`.

## Computation engines

| Engine | Suitable cases | Precision |
| --- | --- | --- |
| `auto` | General use; automatically chooses a suitable engine | No intentional approximation unless `absolute_error` is supplied |
| `dense` | Small or continuous-valued diagrams | Original Gaussian sum in exact arithmetic, with different floating-point summation order |
| `factorized` | Repeated birth/persistence coordinates | Uses coordinate multiplicities, exact in real arithmetic |
| `shared` | Batches sharing coordinates and Gaussian settings | Reuses Gaussian bases across diagrams |
| `sparse` | Narrow Gaussian kernels | Error-controlled local truncation; requires Numba |
| `grid_fft` | Large diagrams with well-resolved Gaussians | Error-controlled Gaussian moment gridding; requires Numba and SciPy |
| `interp_fft` | Large continuous diagrams on sufficiently fine grids | Error-controlled lattice interpolation and one Gaussian FFT; requires Numba and SciPy |

`shared` is a batch/CLI engine rather than a single-diagram `persistence_image` engine. The `grid_fft` and `interp_fft` methods require an explicit positive `absolute_error`. The precision certificates bound **truncation or interpolation error in exact arithmetic**; they do not include floating-point FFT roundoff or signed-deposition roundoff. The output should be compared with a direct calculation using numerical tolerances rather than bit-for-bit equality.

The `interp_fft` method deposits each source point to a local tensor-product interpolation stencil on an extended regular lattice, then evaluates one zero-padded Gaussian convolution. Its work scales approximately as `O(N * wx * wy + P * log P)`, where `N` is the point count, `wx * wy` is stencil size, and `P` is the FFT workspace size. See [the mathematical error-bound derivation](docs/ONE_FFT_ERROR_BOUNDS.md).

## Stable weighted, pixel-integrated persistence images

Choose `--representation stable` for the traditional weighted, integrated
Gaussian persistence image: normalized Gaussian **probability densities**,
a bounded Lipschitz ramp `min(p/weight_scale, 1)` vanishing at zero persistence,
and **exact pixel integrals** using Gaussian CDF differences.

```bash
persistence-images --input ./data/diagrams --spacing ./data/spacing.csv \\
  --output ./output/stable.csv --nx 128 --ny 128 \\
  --sigma-x 0.2 --sigma-y 0.2 \\
  --representation stable --weight-scale 1.0 --bounds 0 3 0 3
```

The Python API also offers `stable_persistence_image(diagram, bounds, ...)`
and `stable_batch_images`, taking finite (birth,persistence) coordinates.
Stable mode supports `auto`, `dense`, and `factorized` (all compute the
exact integral formula in real arithmetic), with no optional truncation
tolerance. The original sampled mode remains unchanged.

The Gaussian is normalized over the **whole plane**, not within the image
bounds. The output is not re-normalized to sum to one. To invoke the
Wasserstein-1 stability theorem, use the same **fixed bounds, bandwidths
and weight scale** across diagrams; default dataset-fitted bounds and
per-sample voxel spacing can otherwise change the mathematical map.
Essential/infinite classes must be handled upstream.

See the [self-contained definition and proof](docs/STABLE_PERSISTENCE_IMAGES.md).

## Reusable CDF bases for stable persistence images

For batches of quantized diagrams, `stable_batch_images(..., engine="shared")`
reuses weighted Gaussian CDF-difference bases. If the unique birth values
are `beta_u` and positive persistence values are `pi_v`, construct:

```text
A[i,u] = Phi((x[i+1]-beta[u])/sigma_x) - Phi((x[i]-beta[u])/sigma_x)
D[v,j] = min(1, pi[v]/weight_scale) *
         (Phi((y[j+1]-pi[v])/sigma_y)-Phi((y[j]-pi[v])/sigma_y))
C[u,v] = number of diagram points with (beta[u], pi[v])
PI = A @ C @ D
```

This retains the **same exact pixel integrals** in real arithmetic, even
for non-square grids and anisotropic bandwidths. It does not normalize
the final image by its sum. Equal voxel-spacing groups reuse the CDF
bases; different spacings are processed separately. The automatic
stable batch engine applies a conservative reuse cost heuristic and
falls back to direct integration when beneficial. The explicit `shared`
engine enforces a memory cap and errors if exceeded; it is not available
for the streaming CLI. Individual diagrams still use `dense`,
`factorized`, or `auto`.

Example:

```python
from persistence_images import StableCdfBasis, stable_batch_images
images = stable_batch_images(
    diagrams, bounds, 128, 128, 0.2, 0.2,
    spacings=[1.0] * len(diagrams), weight_scale=1.0, engine="shared"
)
```

This factorization is standard separability plus a multiplicity matrix;
it does not change the assumptions of the mathematical Wasserstein
stability bound. Floating-point summation order may differ from direct
integration.

## Testing and benchmarking

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python benchmarks/benchmark_one_fft.py
```

The benchmark script measures warmed computation kernels (not initial JIT compilation or CSV input/output). Performance varies considerably with diagram size, grid resolution, point multiplicity, bandwidth, and hardware. Measure your own workloads before selecting an engine.

To compare the Python and Rust implementations on the same synthetic diagrams, install Rust and Cargo, clone the [Rust repository](https://github.com/jhnrckmnznrs/persistence_images), and run:

```bash
python scripts/compare_with_rust.py /path/to/persistence_images
```

The package is maintained independently from Rust and does not require a Rust compiler for ordinary installation or use.

## License

MIT License. Copyright (c) 2026 John Rick Manzanares.

## Reproducible benchmarks

The [shared benchmark suite](benchmarks/README.md) includes deterministic
small-to-large quantized/continuous diagrams, anisotropic and narrow kernels,
independent reference errors, CPU metadata and peak process RSS. Run both
repositories with `python benchmarks/run_suite.py --rust-repo ../persistence_images`.
