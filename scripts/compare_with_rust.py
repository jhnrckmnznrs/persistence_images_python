"""Optional cross-language integration test, requires Rust/Cargo and the Rust source.

Usage:
    python scripts/compare_with_rust.py /path/to/persistence_images
"""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def read_features(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as f:
        return {row[0]: np.asarray(row[1:], dtype=float) for row in csv.reader(f)}


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/compare_with_rust.py /path/to/rust/repository")
        return 2
    repo = Path(sys.argv[1]).resolve()
    if not (repo / "Cargo.toml").is_file():
        raise FileNotFoundError(f"Cargo.toml not found in {repo}")

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        diagrams = work / "diagrams"
        diagrams.mkdir()
        (diagrams / "sample_a.csv").write_text(
            "0.01,1.2\n0.03,2.1\n0.03,0.8\n", encoding="utf-8"
        )
        (diagrams / "sample_b.csv").write_text(
            "0.5,0.6\n1.4,2.7\n", encoding="utf-8"
        )
        spacing = work / "spacing.csv"
        spacing.write_text(
            "filename,info,spacing_micro\nsample_a.tif,,0.86\nsample_b.tif,,1.02\n",
            encoding="utf-8",
        )
        rust_out = work / "rust.csv"
        python_out = work / "python.csv"
        flags = ["--input", str(diagrams), "--spacing", str(spacing),
                 "--nx", "10", "--ny", "8", "--sigma-x", "0.86",
                 "--sigma-y", "0.63984", "--remove-h0"]
        subprocess.run(["cargo", "run", "--quiet", "--release", "--manifest-path",
                        str(repo / "Cargo.toml"), "--", *flags,
                        "--output", str(rust_out)], check=True)
        subprocess.run([sys.executable, "-m", "persistence_images", *flags,
                        "--output", str(python_out)], check=True)
        rust = read_features(rust_out)
        python = read_features(python_out)
        assert rust.keys() == python.keys(), "Different filenames"
        for name in rust:
            np.testing.assert_allclose(rust[name], python[name], rtol=1e-12,
                                       atol=1e-13, err_msg=name)
        print("Rust/Python parity passed for synthetic diagrams.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
