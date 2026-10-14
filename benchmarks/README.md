# Reproducible persistence-image benchmarking

The `workloads.json` manifest is identical to the one in the
[companion Rust repository](https://github.com/jhnrckmnznrs/persistence_images).
Both languages use a deterministic 32-bit generator and verify an FNV-1a
fingerprint of each dataset. Inputs are **birth–persistence** points.

The standard suite covers small and medium diagrams; the full suite adds
20,000- and 100,000-point diagrams, quantized and continuous coordinates,
anisotropic and narrow bandwidths. Samples use fixed global bounds [0,1]².

```bash
python -m pip install -e ".[dev]"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  python benchmarks/benchmark_suite.py --case medium_continuous --engine dense \
    --repeats 5 --warmup 1 --output results/python_medium.json

python benchmarks/run_suite.py --rust-repo ../persistence_images \
    --profile smoke --repeats 3 --output-dir results/smoke

# Full suite may require substantial runtime/memory.
python benchmarks/run_suite.py --rust-repo ../persistence_images \
    --profile full --repeats 5 --output-dir results/full
```

Each run launches one case and backend in its own process. Reports include
median kernel runtime, independent direct-reference error, and Linux
process-wide **peak RSS** (VmHWM, including runtime initialization/warmup,
not incremental algorithm memory). Warmup and initial Numba compilation are
excluded from timed loops. The reported time excludes file I/O and process
launch. For <=2048 pixels the direct sum checks every pixel; larger grids
check **64 deterministic pixels** only. These sampled errors are **not**
uniform image error certificates; analytical Gaussian approximation bounds
exclude floating-point FFT/deposition error.

The orchestrator stores one JSON per run plus `summary.csv` and
`summary.json`. If an engine does not support a workload, a failure record
is kept rather than a fabricated time. Select specific backends with
`--engines python:dense,rust:separable,python:interp_fft,rust:interp-fft`.
Use `--strict` to reject any unsupported case. A cross-language image-sum
check and fixed fingerprints detect drift but do not constitute full pixelwise
parity. Existing end-to-end Rust/Python CLI parity tests remain independent.

Pin compiler/Python and library versions, run on the **same physical machine**,
and limit BLAS/Numba threads for controlled comparisons. Hosted CI benchmarks
verify correctness, not credible speedup estimates. Synthetic workloads
should be supplemented by genuine diagrams.
