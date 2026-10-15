"""Compare Python integrated FFT against exact Rust CDF pixel integrals."""
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
    p=base/"python.csv"
    r=base/"rust.csv"
    subprocess.run([sys.executable,"-m","persistence_images",*common,
        "--engine","interp_fft","--absolute-error","1e-8","--output",str(p)],check=True)
    subprocess.run(["cargo","run","--release","--locked","--",*common,
        "--engine","direct","--output",str(r)],cwd=rust,check=True)
    with p.open(newline="") as f: pp=list(csv.reader(f))
    with r.open(newline="") as f: rr=list(csv.reader(f))
    assert [a[0] for a in pp]==[b[0] for b in rr]
    error=max(abs(float(x)-float(y))
        for a,b in zip(pp,rr) for x,y in zip(a[1:],b[1:]))
    assert error<=1.1e-8,error
    print("Stable integrated FFT vs Rust exact CDF maximum difference:",error)
