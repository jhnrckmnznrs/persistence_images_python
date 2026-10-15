"""Check Python and Rust stable integrated FFT against Rust exact CDF pixels."""
import csv
from pathlib import Path
import subprocess
import sys
import tempfile

rust=Path(sys.argv[1]).resolve()
with tempfile.TemporaryDirectory() as folder:
    base=Path(folder)
    diagrams=base/"diagrams"
    diagrams.mkdir()
    for sample in range(2):
        rows=[]
        for k in range(550+sample*130):
            birth=.05+((k*71+sample*17)%997)/1100.
            persistence=.05+((k*37+sample*23)%991)/1200.
            rows.append(f"{birth:.15f},{birth+persistence:.15f}\n")
        (diagrams/f"{sample}.csv").write_text("".join(rows))
    spacing=base/"spacing.csv"
    spacing.write_text("filename,note,spacing\n0.tif,test,1.0\n1.tif,test,1.15\n")
    common=["--input",str(diagrams),"--spacing",str(spacing),
        "--nx","96","--ny","80","--sigma-x","0.23","--sigma-y","0.19",
        "--weight-scale","0.7","--bounds","0","2","0","2",
        "--representation","stable"]
    def run_python(engine):
        path=base/f"python_{engine}.csv"
        cmd=[sys.executable,"-m","persistence_images",*common,
             "--engine",engine,"--output",str(path)]
        if engine=="interp_fft":cmd+=["--absolute-error","1e-8"]
        subprocess.run(cmd,check=True)
        with path.open(newline="") as f:return list(csv.reader(f))
    def run_rust(engine):
        path=base/f"rust_{engine}.csv"
        cmd=["cargo","run","--release","--locked","--",*common,
             "--engine",engine,"--output",str(path)]
        if engine in ("interp-fft","auto"):cmd+=["--absolute-error","1e-8"]
        subprocess.run(cmd,cwd=rust,check=True)
        with path.open(newline="") as f:return list(csv.reader(f))
    truth=run_rust("direct")
    outputs={"python integrated FFT":run_python("interp_fft"),
             "rust integrated FFT":run_rust("interp-fft"),
             "rust tolerance-aware auto":run_rust("auto")}
    for label,output in outputs.items():
        assert [a[0] for a in output]==[a[0] for a in truth]
        error=max(abs(float(x)-float(y))
            for a,b in zip(output,truth) for x,y in zip(a[1:],b[1:]))
        assert error<=1.1e-8,(label,error)
        print(label,"versus Rust direct stable CDF max difference:",error)

