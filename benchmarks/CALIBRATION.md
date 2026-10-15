# Stable persistence-image backend calibration

These benchmark fixtures are controlled synthetic diagrams, **not real
scientific datasets**. They exercise continuous and quantized source
coordinates under identical Gaussian integrals in both implementations.
The timings below come from a single GitHub Actions workflow run on
15 October 2026, with one warmed measured repetition per case and
threads limited to one; **do not treat them as statistically robust
hardware benchmarks**.

Run and raw machine-readable reports:
[Cross-language GitHub Actions run](https://github.com/jhnrckmnznrs/persistence_images_python/actions/runs/38718065258).

Parameters in both cases: 2,500 persistence points; 96 × 80 pixels;
normalized, ramp-weighted Gaussian *pixel integrals*;
fixed bounds [0,1]², sigma_x=0.16, sigma_y=0.10,
weight_scale=0.7, requested interpolation tolerance 1e-8.

## Continuous sources

Generator FNV64 fingerprint: `fc2c829e048f1ae6`.

| Language | Direct (ms) | Integrated one-FFT (ms) | Auto (ms) |
| --- | ---: | ---: | ---: |
| Rust | 19.601 | 1.401 | 1.350 |
| Python | 25.177 | 1.443 | 1.922 |

Observed maximum difference at 64 deterministic pixels:
approximately 1.23e-12 for the integrated FFT against direct stable
integration. This is an empirical sample, not a whole-grid guarantee;
the integrated interpolation selector independently certifies a
per-pixel bound in exact arithmetic.

## Quantized sources

Generator FNV64 fingerprint: `bfc4ef72eb4dab5b`.

| Language | Direct (ms) | Shared CDF bases (ms) | Auto (ms) |
| --- | ---: | ---: | ---: |
| Rust | 16.547 | 0.253 | 0.256 |
| Python | 17.507 | 0.587 | 0.673 |

The exact real-arithmetic CDF factorization agrees with direct
integration to observed floating-point discrepancies below 6e-15 in
the sampled pixels.

## Interpretation and calibration policy

For the continuous case, automatic execution should admit the
interpolation-certified integrated FFT. For repeated coordinates,
it should examine exact weighted CDF bases before FFT.
For small diagrams, non-reusable source coordinates, unresolvable
narrow Gaussians, out-of-grid sources or missing approximation tolerance,
use the exact stable CDF fallback.

The dispatch uses deliberately simple empirical work estimates and
is **not** proven globally optimal. The FFT interpolation certificates
exclude signed-deposition roundoff and FFT roundoff. The Linux peak
RSS metric includes process and runtime startup, so it is not the
incremental algorithm allocation.

Repeat the complete suite on the same physical machine with more
repetitions, preferably also on actual persistence diagrams before
using these estimates in a manuscript:

```bash
python benchmarks/run_suite.py \
  --rust-repo ../persistence_images \
  --representation stable --profile full --repeats 5 \
  --output-dir benchmarks/results/stable
```

See [README.md](README.md) for environment variables, error
reference scope, and report formats.


## Larger 20,000-point validation

A separate [successful benchmark run](https://github.com/jhnrckmnznrs/persistence_images_python/actions/runs/38718397901)
used the manifest's `large_continuous` and `large_quantized`
cases, 128 × 128 pixels, five different implementations or dispatches,
and the same fixed error tolerance. Each result still uses a **single
warmed timed repetition**; errors are checked at 64 fixed pixels.

| Dataset | Language | Direct ms | Preferred method ms | Auto ms |
| --- | --- | ---: | ---: | ---: |
| Continuous | Rust | 277.904 | FFT 5.921 | 5.954 |
| Continuous | Python | 286.848 | FFT 5.683 | 7.166 |
| Quantized | Rust | 268.058 | shared CDF 1.168 | 1.176 |
| Quantized | Python | 263.016 | shared CDF 2.681 | 3.085 |

Maximum sampled absolute error was 6.15e-13 for the approximate
integrated FFT. The exact shared CDF computation agreed with the direct
integrals at floating-point precision. The Rust and Python results were
compared using matching generator fingerprints and sampled pixel indices.

**Limitations:** Single-repetition hosted benchmark times do not establish
portable speedups or a provably optimal selector. The large-case input
coordinates are controlled synthetic examples, not real measurements.
Real persistence-diagram datasets and repeated local hardware benchmarks
remain essential before publication.
