"""Check deterministic fixture generation and independent reference."""
import importlib.util,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("benchmark_suite",HERE/"benchmarks"/"benchmark_suite.py")
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
def test_fingerprints():
    s=json.loads((HERE/"benchmarks"/"workloads.json").read_text())
    for c in s["cases"]:
        p,digest=mod.generate(c,s["quantized_levels"])
        assert p.shape==(c["n"],2)
        assert digest==c["fingerprint_fnv64"]
        assert np.all((p>=0)&(p<=1))
def test_reference():
    s=json.loads((HERE/"benchmarks"/"workloads.json").read_text())
    c=s["cases"][0]
    r=mod.run(c,s,"dense",1,0,1e-8)
    assert r["reference_scope"]=="full"
    assert r["max_abs_error"]<1e-10
    assert len(r["times_ms"])==1

def test_stable_reference_and_approximate_backend():
    from persistence_images import stable_persistence_image, Bounds
    s=json.loads((HERE/"benchmarks"/"workloads.json").read_text())
    c=s["cases"][0]
    report=mod.run(c,s,"dense",1,0,1e-8,"stable",0.7)
    assert report["representation"]=="stable"
    assert report["reference_scope"]=="full"
    assert report["max_abs_error"]<1e-10
    assert report["image_sum"]>=0

def test_stable_fft_benchmark_on_medium():
    import pytest
    pytest.importorskip("numba")
    pytest.importorskip("scipy")
    s=json.loads((HERE/"benchmarks"/"workloads.json").read_text())
    c=next(x for x in s["cases"] if x["name"]=="medium_continuous")
    r=mod.run(c,s,"interp_fft",1,0,1e-8,"stable",0.7)
    assert r["max_abs_error"]<=1.02e-8
    assert r["reference_pixel_count"]==64
