"""Controlled numerical comparison: persim exact pixel masses; GUDHI sampled.

Run:
  OPENBLAS_NUM_THREADS=1 python benchmarks/compare_libraries.py --n 250 --pixels 24 --repeats 5

Every library is configured with the *same* birth/persistence coordinate data,
fixed bounds, normalized Gaussian variance and bounded linear-ramp weights.
persim outputs pixel INTEGRALS and is numerically comparable to the stable
backend. GUDHI outputs point-SAMPLED Gaussian heights; its numerical parity
is tested against an independent sampled reference, NOT stable image integrals.

Benchmark times exclude generator construction, process startup and fitting.
Times are from repeated calls, with the first call excluded as warmup.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
from persistence_images import Bounds, stable_persistence_image, stable_batch_images, flexible_persistence_image


def images_from_persim(points, pixels, sigma, scale):
    from persim import PersistenceImager
    imager=PersistenceImager(
        birth_range=(0.0,1.0),
        pers_range=(0.0,1.0),
        pixel_size=1.0/pixels,
        weight="linear_ramp",
        weight_params={"low":0.0,"high":1.0,"start":0.0,"end":scale},
        kernel="gaussian",
        kernel_params={"sigma":np.diag([sigma**2,sigma**2])}
    )
    if tuple(imager.resolution)!=(pixels,pixels):
        raise RuntimeError(f"persim resolution does not match: {imager.resolution}")
    return imager


def sampled_reference(points, pixels, sigma, scale):
    # GUDHI evaluates on linspace endpoints, not at centers/pixel integrals.
    grid=np.linspace(0.0,1.0,pixels)
    img=np.zeros((pixels,pixels),dtype=np.float64)  # y-major, x-fast
    denom=2*np.pi*sigma*sigma
    for b,p in points:
        dx=(grid-b)/sigma
        dy=(grid-p)/sigma
        img+=min(p/scale,1.0)*(
            np.exp(-0.5*dy*dy)[:,None]*
            np.exp(-0.5*dx*dx)[None,:])/denom
    return img.ravel(order="C")


def timed(fn,repeats):
    fn()  # warmed; not charged to timed operations
    times=[]
    output=None
    for _ in range(repeats):
        begin=perf_counter()
        output=fn()
        times.append(1e3*(perf_counter()-begin))
    return output,median(times),times


def run(n=250,pixels=24,repeats=5,sigma=0.2,scale=0.6,seed=93,kind='continuous'):
    if n<1 or pixels<2 or repeats<1 or sigma<=0 or scale<=0:
        raise ValueError("Invalid benchmark settings")
    if kind not in {"continuous","quantized"}:
        raise ValueError("kind must be continuous or quantized")
    rng=np.random.default_rng(seed)
    if kind == "quantized":
        points=np.column_stack((
            rng.integers(1,17,size=n)/20.0,
            rng.integers(1,17,size=n)/20.0
        ))
    else:
        points=np.column_stack((
            rng.uniform(0.05,0.9,n),rng.uniform(0.05,0.95,n)
        ))
    bounds=Bounds(0,1,0,1)
    # End-to-end fixed-bound stable mode and the independently exposed
    # flexible analytic CDF route should compute the same image.
    ref,ref_time,_=timed(lambda:stable_persistence_image(
        points,bounds,pixels,pixels,sigma,sigma,
        weight_scale=scale,engine="dense"),repeats)
    flex,flex_time,_=timed(lambda:flexible_persistence_image(
        points,bounds,pixels,pixels,
        weight="linear_ramp",weight_params={"start":0.0,"end":scale},
        sigma_x=sigma,sigma_y=sigma),repeats)
    fast,fast_time,_=timed(lambda:stable_batch_images(
        [points],bounds,pixels,pixels,sigma,sigma,
        weight_scale=scale,engine="auto",absolute_error=1e-8)[0],repeats)
    fast_err=float(np.max(np.abs(ref-fast)))
    if fast_err>1.02e-8:
        raise AssertionError(f"Adaptive stable numerical tolerance exceeded: {fast_err}")
    numerical=float(np.max(np.abs(ref-flex)))
    if numerical>1e-10:
        raise AssertionError(f"Flexible CDF parity failed: {numerical}")
    result={
        "schema_version":1,
        "dataset":{"generator":"PCG64","seed":seed,"n":n,"kind":kind,"pixels":pixels,
                   "sigma_std":sigma,"weight_scale":scale,"bounds":[0,1,0,1]},
        "representations":{"stable":"Gaussian mass integrated over rectangles",
                           "gudhi":"normalized Gaussian density at grid samples"},
        "runtime":{"python":platform.python_version(),
                   "numpy":np.__version__,"openblas":os.getenv("OPENBLAS_NUM_THREADS")},
        "results":{
            "our_stable_direct":{"median_ms":ref_time},
            "our_stable_auto":{"median_ms":fast_time,"max_abs_error_against_stable":fast_err},
            "our_flexible_diagonal_cdf":{"median_ms":flex_time,
                                          "max_abs_error_against_stable":numerical}
        },
    }
    try:
        import persim
        imager=images_from_persim(points,pixels,sigma,scale)
        def compute_persim():
            return np.asarray(imager.transform(points,skew=False),dtype=np.float64)
        img,elapsed,_=timed(compute_persim,repeats)
        if img.shape!=(pixels,pixels):
            raise AssertionError(f"persim image shape {img.shape} != {(pixels,pixels)}")
        err=float(np.max(np.abs(img.ravel(order="C")-ref)))
        result["results"]["persim_integrated"]={
            "median_ms":elapsed,"max_abs_error_against_stable":err,
            "version":getattr(persim,"__version__",None)}
        if err>2e-8:
            raise AssertionError(f"persim integrated pixel parity failed: {err}")
    except ImportError:
        result["results"]["persim_integrated"]={"status":"not_installed"}
    try:
        import gudhi
        from gudhi.representations import PersistenceImage
        bd=np.column_stack((points[:,0],points[:,0]+points[:,1]))
        g=PersistenceImage(
            bandwidth=sigma,weight=lambda point:min(point[1]/scale,1.0),
            resolution=[pixels,pixels],im_range=[0.,1.,0.,1.])
        g.fit([bd])
        out,elapsed,_=timed(lambda:np.asarray(g.transform([bd])[0]),repeats)
        expected=sampled_reference(points,pixels,sigma,scale)
        error=float(np.max(np.abs(out-expected)))
        result["results"]["gudhi_sampled"]={
            "median_ms":elapsed,"max_abs_error_against_sampled_reference":error,
            "version":getattr(gudhi,"__version__",None)}
        if error>1e-9:
            raise AssertionError(f"GUDHI sampled parity failed: {error}")
    except ImportError:
        result["results"]["gudhi_sampled"]={"status":"not_installed"}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n",type=int,default=250)
    p.add_argument("--pixels",type=int,default=24)
    p.add_argument("--sigma",type=float,default=0.2)
    p.add_argument("--weight-scale",type=float,default=0.6)
    p.add_argument("--seed",type=int,default=93)
    p.add_argument("--kind",choices=("continuous","quantized"),default="continuous")
    p.add_argument("--repeats",type=int,default=5)
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    report=run(args.n,args.pixels,args.repeats,args.sigma,
               args.weight_scale,args.seed,args.kind)
    s=json.dumps(report,indent=2,sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(s+"\n")
    print(s)


if __name__=="__main__":
    main()
