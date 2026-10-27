"""Explain a saved, frozen linear persistence-image fold without refitting.

CSV has headers 'birth,persistence'. NPZ must contain:
birth_edges, persistence_edges, sigma, tau, coef, center,
scale, intercept. Optionally active, other_blocks_offset, expected_prediction.
Only UNIFORM rectangular grids are supported by this API.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from persistence_images import Bounds, LinearPIModel, explain_linear_pi


def read_diagram(path):
    with Path(path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or not {"birth", "persistence"}.issubset(reader.fieldnames):
            raise ValueError("diagram CSV must have birth,persistence column headers")
        values = [(float(row["birth"]), float(row["persistence"])) for row in reader]
    return np.asarray(values, dtype=float).reshape(-1, 2)


def load_fold(path):
    with np.load(path, allow_pickle=False) as d:
        required = {
            "birth_edges", "persistence_edges", "sigma", "tau",
            "coef", "center", "scale", "intercept",
        }
        missing = required - set(d.files)
        if missing:
            raise ValueError(f"missing saved fold keys: {sorted(missing)}")
        u, v = d["birth_edges"], d["persistence_edges"]
        if any(x.ndim != 1 or len(x) < 2 or not np.all(np.isfinite(x))
               or np.any(np.diff(x) <= 0) for x in (u, v)):
            raise ValueError("grid edges must be finite strictly increasing arrays")
        if not all(np.allclose(np.diff(x), np.diff(x)[0], rtol=1e-12, atol=1e-12)
                   for x in (u, v)):
            raise ValueError("nonuniform grid not supported by the current API")
        bounds = Bounds(float(u[0]), float(u[-1]), float(v[0]), float(v[-1]))
        coef = np.asarray(d["coef"], dtype=float).copy()
        center = np.asarray(d["center"], dtype=float).copy()
        if "active" in d.files:
            active = np.asarray(d["active"], dtype=bool)
            if active.shape != coef.shape or not np.all(coef[~active] == 0):
                raise ValueError("inactive coefficients must be zero on full grid")
        model = LinearPIModel(
            coef=coef, intercept=float(d["intercept"]),
            center=center, scale=np.asarray(d["scale"], dtype=float).copy(),
        )
        offset = float(d["other_blocks_offset"]) if "other_blocks_offset" in d.files else 0.
        expected = float(d["expected_prediction"]) if "expected_prediction" in d.files else None
        sigma = float(d["sigma"])
        tau = float(d["tau"])
    return bounds, len(u)-1, len(v)-1, sigma, tau, model, offset, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=Path, required=True)
    parser.add_argument("--diagram", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="output file stem for _intervals.csv and _summary.json")
    args = parser.parse_args()
    bounds, nx, ny, sigma, tau, model, offset, expected = load_fold(args.fold)
    result = explain_linear_pi(read_diagram(args.diagram), bounds, nx, ny,
                               sigma, sigma, model,
                               weight_scale=tau, other_blocks=offset)
    if expected is not None and not np.isclose(result.prediction, expected, atol=1e-7, rtol=1e-7):
        raise ArithmeticError("saved original fold prediction not reproduced")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_name(args.output.name + "_intervals.csv").open(
        "w", newline="", encoding="utf-8"
    ) as out:
        writer = csv.writer(out)
        writer.writerow(["interval_id", "birth", "persistence", "contribution"])
        for i, (point, val) in enumerate(zip(result.diagram, result.contributions)):
            writer.writerow([i, float(point[0]), float(point[1]), float(val)])
    summary = {
        "prediction": result.prediction,
        "reconstructed_prediction": result.reconstructed_prediction,
        "reconstruction_abs_error": result.reconstruction_error,
        "base": result.base,
        "other_blocks": result.other_blocks,
        "n_intervals": len(result.diagram),
        "n_pixels": len(result.image),
    }
    if expected is not None:
        summary["saved_expected_prediction"] = expected
        summary["saved_expected_abs_error"] = abs(result.prediction - expected)
    args.output.with_name(args.output.name + "_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
