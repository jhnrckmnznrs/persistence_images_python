"""Fixed cross-language numerical-kernel benchmarks, one case per process."""
from __future__ import annotations
import argparse,json,math,os,platform,statistics,time
from pathlib import Path
import numpy as np
from persistence_images import Bounds,persistence_image,stable_persistence_image,stable_batch_images

MANIFEST=Path(__file__).with_name("workloads.json")
FNV_OFFSET=14695981039346656037
FNV_PRIME=1099511628211

def generate(case,levels):
    state=int(case["seed"])
    points=np.empty((case["n"],2),dtype=np.float64)
    digest=FNV_OFFSET
    for row in range(case["n"]):
        for axis in range(2):
            state=(state*1664525+1013904223)&0xffffffff
            v=state>>8
            for shift in (0,8,16,24):
                digest=((digest^((v>>shift)&255))*FNV_PRIME)&0xffffffffffffffff
            if case["kind"]=="continuous":
                points[row,axis]=(v+0.5)/(2**24)
            else:
                points[row,axis]=(v%levels[axis])/levels[axis]
    return points,f"{digest:016x}"

def sample_indices(nx,ny):
    m=nx*ny
    return list(range(m)) if m<=2048 else [(i*(m-1))//63 for i in range(64)]

def direct_reference(points,case,indices):
    ref=[]
    for index in indices:
        x=(index//case["ny"]+0.5)/case["nx"]
        y=(index%case["ny"]+0.5)/case["ny"]
        dx=(x-points[:,0])/case["sigma_x"]
        dy=(y-points[:,1])/case["sigma_y"]
        ref.append(math.fsum(np.exp(-0.5*(dx*dx+dy*dy)).tolist()))
    return np.array(ref)

def stable_direct_reference(points,case,indices,weight_scale):
    """Independent direct stable reference at specified pixel rectangles."""
    from persistence_images.stable import _cdf_intervals
    ref=[]
    weights=np.minimum(points[:,1]/weight_scale,1.)
    for index in indices:
        i,j=divmod(index,case["ny"])
        xe=np.array([i/case["nx"],(i+1)/case["nx"]])
        ye=np.array([j/case["ny"],(j+1)/case["ny"]])
        gx=_cdf_intervals(xe,points[:,0],case["sigma_x"])[0]
        gy=_cdf_intervals(ye,points[:,1],case["sigma_y"])[0]
        ref.append(math.fsum((weights*gx*gy).tolist()))
    return np.asarray(ref,dtype=np.float64)

def peak_rss():
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1])*1024
    except OSError:
        pass
    return None

def cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":",1)[1].strip()
    except OSError:
        pass
    return None

def run(case,suite,engine,repeats=5,warmup=1,epsilon=1e-8,representation='sampled',weight_scale=0.7):
    if representation not in ("sampled","stable") or weight_scale<=0 or not math.isfinite(weight_scale):
        raise ValueError("Invalid representation or persistence weight scale")
    if representation=="stable" and engine not in ("auto","dense","factorized","shared","interp_fft"):
        raise ValueError("Unsupported stable engine")
    if repeats<1 or warmup<0 or epsilon<=0 or not math.isfinite(epsilon):
        raise ValueError("Invalid repeats, warmup, or tolerance")
    points,digest=generate(case,suite["quantized_levels"])
    if digest!=case["fingerprint_fnv64"]:
        raise AssertionError(f"Dataset fingerprint mismatch: {digest}")
    bounds=Bounds(*suite["bounds"])
    approx=engine in ("sparse","interp_fft","grid_fft","auto")
    def compute():
        kwargs=dict(nx=case["nx"],ny=case["ny"],sigma_x=case["sigma_x"],
                    sigma_y=case["sigma_y"],engine=engine,
                    absolute_error=epsilon if approx else None)
        if representation=="stable":
            kwargs["weight_scale"]=weight_scale
            # Both CLIs use batch dispatch, even for one diagram.
            return stable_batch_images([points],bounds,**kwargs)[0]
        return persistence_image(points,bounds,**kwargs)
    for _ in range(warmup):compute()
    samples=[]
    image=None
    for _ in range(repeats):
        start=time.perf_counter_ns()
        image=compute()
        samples.append((time.perf_counter_ns()-start)/1e6)
    indices=sample_indices(case["nx"],case["ny"])
    exact=(stable_direct_reference(points,case,indices,weight_scale)
           if representation=="stable" else direct_reference(points,case,indices))
    error=float(np.max(np.abs(image[indices]-exact)))
    return {
      "schema_version":1,"suite":"persistence-images-cross-language-v1",
      "case":case["name"],"profile":case["profile"],"kind":case["kind"],
      "language":"python","engine":engine,"representation":representation,
      "weight_scale":weight_scale if representation=="stable" else None,
      "generator":suite["generator"],
      "source_fingerprint_fnv64":digest,
      "n_points":case["n"],"nx":case["nx"],"ny":case["ny"],
      "sigma_x":case["sigma_x"],"sigma_y":case["sigma_y"],
      "absolute_error_requested":epsilon if approx else None,
      "warmup":warmup,"repeats":repeats,
      "times_ms":samples,"median_ms":statistics.median(samples),
      "peak_rss_bytes":peak_rss(),
      "memory_metric":"Linux VmHWM process peak (includes runtime and warmup)",
      "reference_scope":"full" if len(indices)==case["nx"]*case["ny"] else "64 sampled pixels",
      "reference_pixel_count":len(indices),
      "max_abs_error":error,
      "max_scaled_error":error/max(1.0,float(np.max(np.abs(exact)))),
      "sampled_pixel_indices":indices,
      "sampled_pixel_values":[float(v) for v in image[indices]],
      "image_sum":float(np.sum(image)),
      "image_linf":float(np.max(np.abs(image))),
      "python_version":platform.python_version(),
      "numpy_version":np.__version__,"platform":platform.platform(),
      "cpu_model":cpu_model(),"logical_cpus":os.cpu_count(),
      "thread_env":{k:os.environ.get(k) for k in
         ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMBA_NUM_THREADS")},
      "precision_note":"Reference checks selected pixels only on large grids; FFT/deposition roundoff excluded from certificates"
    }

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case",default="small_continuous")
    p.add_argument("--representation",choices=("sampled","stable"),default="sampled")
    p.add_argument("--weight-scale",type=float,default=0.7)
    p.add_argument("--engine",default="dense",choices=[
      "auto","dense","factorized","shared","sparse","grid_fft","interp_fft"])
    p.add_argument("--warmup",type=int,default=1)
    p.add_argument("--repeats",type=int,default=5)
    p.add_argument("--absolute-error",type=float,default=1e-8)
    p.add_argument("--output",type=Path)
    p.add_argument("--list",action="store_true")
    a=p.parse_args()
    suite=json.loads(MANIFEST.read_text())
    if a.list:
        for c in suite["cases"]:
            print(f'{c["name"]:24s} [{c["profile"]}] points={c["n"]} grid={c["nx"]}x{c["ny"]}')
        return
    lookup={c["name"]:c for c in suite["cases"]}
    if a.case not in lookup:p.error("Unknown case; use --list")
    result=run(lookup[a.case],suite,a.engine,a.repeats,a.warmup,a.absolute_error,
               a.representation,a.weight_scale)
    text=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(text+"\n")
    print(text)
if __name__=="__main__": main()
