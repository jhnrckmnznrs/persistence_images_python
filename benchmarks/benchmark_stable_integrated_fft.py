"""Stable integrated FFT benchmark; kernel cache and JIT warmed.

OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python benchmarks/benchmark_stable_integrated_fft.py
"""
import json
from statistics import median
from time import perf_counter
import numpy as np
from persistence_images import Bounds,stable_persistence_image,stable_interp_fft_image

def main():
    rng=np.random.default_rng(47)
    bounds=Bounds(0,1,0,1)
    for n,nx,ny,sx,sy in [(500,64,64,.18,.18),(5000,96,80,.18,.20),
                            (20000,128,128,.18,.18),(20000,128,128,.08,.24)]:
        pts=rng.uniform(.01,.99,(n,2))
        opts=dict(nx=nx,ny=ny,sigma_x=sx,sigma_y=sy,weight_scale=.4)
        fft,cert=stable_interp_fft_image(pts,bounds,absolute_error=1e-8,
            return_certificate=True,**opts)
        exact=stable_persistence_image(pts,bounds,engine="dense",**opts)
        times={}
        for name,fn in [
            ("direct",lambda:stable_persistence_image(pts,bounds,engine="dense",**opts)),
            ("fft",lambda:stable_interp_fft_image(pts,bounds,absolute_error=1e-8,**opts)),
        ]:
            samples=[]
            for _ in range(4):
                t=perf_counter()
                fn()
                samples.append(1000*(perf_counter()-t))
            times[name]=median(samples[1:])
        err=np.abs(exact-fft)
        report=dict(n=n,nx=nx,ny=ny,sx=sx,sy=sy,fft_ms=times["fft"],
            direct_ms=times["direct"],speedup=times["direct"]/times["fft"],
            max_error=float(err.max()),l1_error=float(err.sum()),
            pixel_bound=cert.pixel_error,l1_bound=cert.vector_l1_error,
            stencil=[cert.width_x,cert.width_y])
        assert report["max_error"] <= cert.pixel_error+2e-9
        print(json.dumps(report),flush=True)
if __name__=="__main__": main()
