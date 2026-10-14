"""Reproducible stable CDF basis benchmark (times exclude dataset creation).

OPENBLAS_NUM_THREADS=1 python benchmarks/benchmark_stable_shared.py
"""
import os
import time
from statistics import median
import numpy as np
from persistence_images import Bounds, stable_batch_images

def main():
    b = Bounds(0, 1, 0, 1)
    diagrams = [
        np.array([(((k * 7 + frame) % 24) / 24.0,
                   (((k * 11 + frame) % 16) + 1) / 17.0)
                  for k in range(1500)], dtype=float)
        for frame in range(12)
    ]
    kwargs = dict(nx=80, ny=64, sigma_x=0.14, sigma_y=0.11,
                  weight_scale=0.7, spacings=[1.0] * len(diagrams))
    def measure(engine):
        elapsed=[]
        output=None
        for _ in range(4):
            t0=time.perf_counter()
            output=stable_batch_images(diagrams,b,engine=engine,**kwargs)
            elapsed.append((time.perf_counter()-t0)*1000)
        return median(elapsed[1:]),output
    direct,ref=measure("dense")
    shared,opt=measure("shared")
    error=max(float(np.max(np.abs(a-b))) for a,b in zip(ref,opt))
    print(f"NumPy={np.__version__} OPENBLAS_NUM_THREADS={os.getenv('OPENBLAS_NUM_THREADS')}")
    print(f"diagrams={len(diagrams)}, points/diagram={len(diagrams[0])}, grid=80x64")
    print(f"stable direct {direct:.3f} ms, stable shared {shared:.3f} ms,"
          f" speedup {direct/shared:.2f}x; max abs difference={error:.3e}")
    assert error<1e-9

if __name__=="__main__":
    main()
