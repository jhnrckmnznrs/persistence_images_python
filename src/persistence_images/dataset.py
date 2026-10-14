"""Dataset loading and voxel-spacing normalization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .io import collect_csv_files, load_spacing, read_diagram


@dataclass
class SpacingDiagram:
    path: Path
    diagram: NDArray[np.float64]  # unscaled (birth, persistence)
    voxel_spacing: float


def load_dataset(
    root: str | Path,
    spacing_csv: str | Path,
    output_csv: str | Path | None = None,
    diagram_suffix: str | None = None,
    remove_h0: bool = False,
) -> list[SpacingDiagram]:
    """Match the Rust pipeline's source selection and spacing-key convention.

    For a diagram at `dir/foo.csv`, spacing is found under key `foo.tif`;
    missing keys fall back to 1.0. Matching is case-sensitive.
    """
    spacing_map = load_spacing(spacing_csv)
    exclude = [spacing_csv]
    if output_csv is not None:
        exclude.append(output_csv)
    dataset: list[SpacingDiagram] = []
    for path in collect_csv_files(root, diagram_suffix=diagram_suffix, exclude=exclude):
        dataset.append(SpacingDiagram(
            path=path,
            diagram=read_diagram(path, remove_h0=remove_h0),
            voxel_spacing=spacing_map.get(f"{path.stem}.tif", 1.0),
        ))
    return dataset


def apply_spacing(sample: SpacingDiagram) -> NDArray[np.float64]:
    """Scale both birth and persistence coordinates by voxel spacing."""
    return sample.diagram * sample.voxel_spacing


def to_scaled_diagrams(dataset: list[SpacingDiagram]) -> list[NDArray[np.float64]]:
    return [apply_spacing(sample) for sample in dataset]
