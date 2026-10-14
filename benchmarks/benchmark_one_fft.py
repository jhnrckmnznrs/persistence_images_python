"""Reproducible comparison; from source checkout use PYTHONPATH=src.
OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python benchmarks/benchmark_one_fft.py
"""
import json,sys,time,os,platform
from pathlib import Path
import numpy as np
from persistence_images import Bounds,persistence_image
from persistence_images._interp_fft import select_uniform_width,InterpolationFFT
from persistence_images._grid_fft import AdaptiveGridFFT

def bench(fn,reps):
    fn() # warm JIT, memory allocator and kernels
    t=[]
    for _ in range(reps):
        t0=time.perf_counter(); result=fn();t.append((time.perf_counter()-t0)*1000)
    return result,float(np.median(t))

rows=[]
for n,g,s in [(2000,128,.2),(10000,128,.2),(100000,128,.2),(100000,256,.2),(100000,256,.05),(100000,512,.2)]:
    arr=np.random.default_rng(110+g).uniform(size=(n,2))
    b=(0.,1.,0.,1.);tol=1e-8
    width=select_uniform_width(n,b,g,g,s,s,tol)
    if width is None:continue
    _,wx,wy,cert=width
    plan=InterpolationFFT(b,g,g,s,s,wx,wy)
    result,interp_ms=bench(lambda: plan.evaluate(arr),5 if n<=10000 else 3)
    moment=AdaptiveGridFFT(b,g,g,s,s)
    (_,_,q),moment_ms=bench(lambda: moment.evaluate(arr,tol),3)
    if n*g*g <= 7e9:
        dense,dense_ms=bench(lambda:persistence_image(arr,Bounds(*b),g,g,s,s,engine='dense').reshape(g,g),3)
        error=float(np.max(np.abs(result-dense)))
    else:dense_ms=None;error=None
    item=dict(points=n,nx=g,ny=g,sigma_x=s,sigma_y=s,wx=wx,wy=wy,
              certificate=cert,interp_ms=interp_ms,moment_ms=moment_ms,
              moment_degree=q,dense_ms=dense_ms,measured_max_error=error,
              moment_speedup=moment_ms/interp_ms,
              dense_speedup=None if dense_ms is None else dense_ms/interp_ms)
    rows.append(item); print(json.dumps(item),flush=True)
meta=dict(runtime='warm kernels/JIT; no CSV; single-thread requested', python=sys.version,
          numpy=np.__version__,platform=platform.platform(),
          openblas_threads=os.environ.get('OPENBLAS_NUM_THREADS'))
out=Path(__file__).with_name('benchmark_one_fft_results.json')
out.write_text(json.dumps(dict(metadata=meta,results=rows),indent=2)+'\n')
print('Saved',out)
