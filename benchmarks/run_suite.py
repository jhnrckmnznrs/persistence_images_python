"""Cross-language benchmark orchestration: isolated processes and JSON/CSV reports."""
import argparse,csv,json,os,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
LEVEL={"smoke":0,"standard":1,"full":2}
ALLOWED={
 "python":{"auto","dense","factorized","sparse","interp_fft","grid_fft"},
 "rust":{"auto","direct","separable","shared","sparse","interp-fft"}}

def recommended(case):
    name=case["name"]
    if name.startswith("small"):
        return ([("python","dense"),("rust","separable"),
                 ("python","auto"),("rust","auto")]
                + ([("python","factorized"),("rust","shared")] if case["kind"]=="quantized" else []))
    if name=="large_anisotropic":
        return [("python","dense"),("rust","separable"),
                ("python","sparse"),("rust","sparse"),
                ("python","auto"),("rust","auto")]
    if name=="large_narrow":
        return [("python","dense"),("python","sparse"),("rust","sparse"),
                ("python","auto"),("rust","auto")]
    if name.startswith("huge"):
        base=[("python","dense"),("python","auto"),("rust","auto"),
              ("python","interp_fft"),("rust","interp-fft")]
    else:
        base=[("python","dense"),("rust","separable"),("python","auto"),
              ("rust","auto"),("python","interp_fft"),("rust","interp-fft")]
    if case["kind"]=="quantized":
        base.extend([("python","factorized"),("rust","shared")])
    return base

def verify(records):
    ok=[r for r in records if r["status"]=="ok"]
    checks=0
    for i,a in enumerate(ok):
        for b in ok[i+1:]:
            if a["case"]!=b["case"] or a["language"]==b["language"]:continue
            checks+=1
            if a["source_fingerprint_fnv64"]!=b["source_fingerprint_fnv64"]:
                raise AssertionError("Dataset generator mismatch: "+a["case"])
            if ("sampled_pixel_indices" in a and "sampled_pixel_indices" in b):
                if a["sampled_pixel_indices"]!=b["sampled_pixel_indices"]:
                    raise AssertionError("Different sampled pixel positions: "+a["case"])
                av=a["sampled_pixel_values"]
                bv=b["sampled_pixel_values"]
                if len(av)!=len(bv):
                    raise AssertionError("Different sampled pixel counts")
                tolerance=(a.get("absolute_error_requested") or 0.0)+(b.get("absolute_error_requested") or 0.0)
                tolerance+=2e-10*max([1.0]+[abs(v) for v in av]+[abs(v) for v in bv])
                if max(abs(x-y) for x,y in zip(av,bv))>tolerance:
                    raise AssertionError("Cross-language sampled pixels differ: "+a["case"])
            pixels=a["nx"]*a["ny"]
            allowed=pixels*2e-8+2e-10*max(1,abs(a["image_sum"]),abs(b["image_sum"]))
            if abs(a["image_sum"]-b["image_sum"])>allowed:
                raise AssertionError("Cross-language image sum mismatch: "+a["case"])
    return checks

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile",choices=list(LEVEL),default="smoke")
    p.add_argument("--rust-repo",type=Path)
    p.add_argument("--engines",help="Comma-separated language:engine pairs")
    p.add_argument("--output-dir",type=Path,default=Path("benchmarks/results"))
    p.add_argument("--repeats",type=int,default=5)
    p.add_argument("--warmup",type=int,default=1)
    p.add_argument("--absolute-error",type=float,default=1e-8)
    p.add_argument("--timeout",type=int,default=900)
    p.add_argument("--strict",action="store_true")
    args=p.parse_args()
    suite=json.loads((HERE/"benchmarks"/"workloads.json").read_text())
    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    rust=args.rust_repo.resolve() if args.rust_repo else None
    if rust and not (rust/"Cargo.toml").is_file():
        p.error("--rust-repo has no Cargo.toml")
    custom=None
    if args.engines:
        custom=[]
        for name in args.engines.split(","):
            lang,sep,engine=name.strip().partition(":")
            if not sep or engine not in ALLOWED.get(lang,set()):
                p.error("Unknown language:engine pair: "+name)
            custom.append((lang,engine))
    env=os.environ.copy()
    for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMBA_NUM_THREADS"):
        env[k]="1"
    records=[]
    for case in suite["cases"]:
        if LEVEL[case["profile"]]>LEVEL[args.profile]:continue
        for lang,engine in custom if custom is not None else recommended(case):
            if lang=="rust" and rust is None:continue
            tag=case["name"]+"__"+lang+"__"+engine.replace("-","_")
            path=out/(tag+".json")
            if lang=="python":
                cmd=[sys.executable,str(HERE/"benchmarks"/"benchmark_suite.py")]
            else:
                cmd=["cargo","run","--quiet","--release","--locked",
                     "--manifest-path",str(rust/"Cargo.toml"),"--example","benchmark_suite","--"]
            cmd+=["--case",case["name"],"--engine",engine,
                   "--repeats",str(args.repeats),"--warmup",str(args.warmup),
                   "--absolute-error",str(args.absolute_error),"--output",str(path)]
            print(f"{case['name']}: {lang} {engine}",file=sys.stderr,flush=True)
            try:
                result=subprocess.run(cmd,capture_output=True,text=True,env=env,timeout=args.timeout)
                if result.returncode:
                    raise RuntimeError((result.stderr or result.stdout)[-1600:])
                data=json.loads(path.read_text())
                data["status"]="ok"
            except (RuntimeError,subprocess.TimeoutExpired,OSError) as exc:
                if args.strict:raise RuntimeError(tag) from exc
                data={"status":"failed","case":case["name"],
                      "language":lang,"engine":engine,"detail":str(exc)[-1600:]}
            path.write_text(json.dumps(data,indent=2,sort_keys=True)+"\n")
            records.append(data)
    pairs=verify(records)
    report={"schema_version":1,"suite":"persistence-images-cross-language-v1",
            "profile":args.profile,"records":records,
            "validation":{"cross_language_pairs_compared":pairs,
              "note":"Image-sum agreement is not pixelwise parity; reference pixels are checked separately."}}
    (out/"summary.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    fields=["status","case","language","engine","kind","n_points","nx","ny",
            "sigma_x","sigma_y","median_ms","peak_rss_bytes","max_abs_error",
            "max_scaled_error","reference_scope","source_fingerprint_fnv64","detail"]
    with (out/"summary.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for rec in records:writer.writerow({k:rec.get(k) for k in fields})
    print(json.dumps({"output_dir":str(out),"records":len(records),
          "ok":sum(r["status"]=="ok" for r in records),"failed":sum(r["status"]!="ok" for r in records),
          "cross_language_pairs_compared":pairs},indent=2))
if __name__=="__main__":main()
