import csv
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np

rust = Path(sys.argv[1]).resolve()
with tempfile.TemporaryDirectory() as folder:
    base = Path(folder)
    diagrams = base / "diagrams"
    diagrams.mkdir()
    (diagrams / "a.csv").write_text("0.1,0.5\n0.7,1.7\n0.9,0.9\n")
    (diagrams / "b.csv").write_text("0.2,0.9\n0.3,0.3000001\n0.8,1.9\n")
    (diagrams / "empty.csv").write_text("")
    spacing = base / "spacing.csv"
    spacing.write_text("filename,note,spacing\na.tif,test,1.0\nb.tif,test,1.7\nempty.tif,test,1.0\n")
    common = ["--input", str(diagrams), "--spacing", str(spacing),
              "--nx", "17", "--ny", "13", "--sigma-x", "0.35",
              "--sigma-y", "0.42", "--weight-scale", "0.8",
              "--bounds", "0", "4", "0", "3", "--representation", "stable"]
    p = base / "python.csv"
    r = base / "rust.csv"
    subprocess.run([sys.executable, "-m", "persistence_images",
                    *common, "--output", str(p)], check=True)
    subprocess.run(["cargo", "run", "--release", "--locked", "--",
                    *common, "--output", str(r)], cwd=rust, check=True)
    def read(path):
        with path.open(newline="") as f:
            return list(csv.reader(f))
    pp, rr = read(p), read(r)
    assert [row[0] for row in pp] == [row[0] for row in rr]
    difference = max(abs(float(x)-float(y))
                     for a,b in zip(pp,rr) for x,y in zip(a[1:],b[1:]))
    assert difference <= 1e-11, difference
    print("stable Rust/Python parity max error:", difference)
