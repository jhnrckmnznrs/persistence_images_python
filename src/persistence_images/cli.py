"""Command-line interface mirroring the Rust CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from .core import compute_bounds, persistence_image
from .batch import batch_persistence_images
from .dataset import load_dataset, to_scaled_diagrams
from .io import collect_csv_files, load_spacing, read_diagram
from .io import save_images_to_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="persistence-images",
        description="Compute Rust-compatible persistence images from CSV diagrams",
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="Diagram directory")
    parser.add_argument("--spacing", "-s", type=Path, required=True, help="Spacing CSV")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output CSV")
    parser.add_argument("--nx", type=int, default=10)
    parser.add_argument("--ny", type=int, default=8)
    parser.add_argument("--sigma-x", type=float, default=0.86)
    parser.add_argument("--sigma-y", type=float, default=0.63984)
    parser.add_argument("--remove-h0", action="store_true")
    parser.add_argument("--diagram-suffix", type=str)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--engine", choices=("auto", "dense", "factorized", "shared", "sparse", "grid_fft", "interp_fft"), default="auto")
    parser.add_argument("--stream", action="store_true", help="Two-pass input scan; avoids keeping diagrams in memory")
    parser.add_argument("--absolute-error", type=float, default=None, help="Optional per-pixel truncation bound; auto stays exact if omitted")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.engine in ("sparse", "grid_fft", "interp_fft") and args.absolute_error is None:
        parser.error("--engine sparse, grid_fft or interp_fft requires --absolute-error")
    if args.stream:
        if args.engine == "shared":
            raise ValueError("The shared engine requires a materialized dataset; omit --stream")
        spacing = load_spacing(args.spacing)
        paths = collect_csv_files(
            args.input, diagram_suffix=args.diagram_suffix,
            exclude=(args.spacing, args.output),
        )
        if not paths:
            print("No data found.")
            return 0

        def scaled(path: Path):
            s = spacing.get(f"{path.stem}.tif", 1.0)
            return read_diagram(path, remove_h0=args.remove_h0) * s

        bounds = compute_bounds(scaled(path) for path in paths)
        if bounds is None:
            print("No persistence points remain after preprocessing.")
            return 0
        print(f"Bounds: {bounds}")
        images = (
            persistence_image(
                scaled(path), bounds,
                nx=args.nx, ny=args.ny,
                sigma_x=args.sigma_x * spacing.get(f"{path.stem}.tif", 1.0),
                sigma_y=args.sigma_y * spacing.get(f"{path.stem}.tif", 1.0),
                batch_size=args.batch_size, engine=args.engine,
                absolute_error=args.absolute_error,
            )
            for path in paths
        )
        save_images_to_csv((path.stem for path in paths), images, args.output)
        print(f"Saved to {args.output}")
        return 0

    dataset = load_dataset(
        args.input, args.spacing, args.output, args.diagram_suffix, args.remove_h0
    )
    if not dataset:
        print("No data found.")
        return 0

    diagrams = to_scaled_diagrams(dataset)
    bounds = compute_bounds(diagrams)
    if bounds is None:
        print("No persistence points remain after preprocessing.")
        return 0
    print(f"Bounds: {bounds}")

    images = batch_persistence_images(
        diagrams, bounds, nx=args.nx, ny=args.ny,
        sigma_x=args.sigma_x, sigma_y=args.sigma_y,
        spacings=[sample.voxel_spacing for sample in dataset],
        batch_size=args.batch_size, engine=args.engine,
        absolute_error=args.absolute_error,
    )
    save_images_to_csv((sample.path.stem for sample in dataset), images, args.output)
    print(f"Saved to {args.output}")
    return 0
