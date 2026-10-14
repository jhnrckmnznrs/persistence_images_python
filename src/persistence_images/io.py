"""CSV input/output conventions of the Rust persistence_images project."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import _points


def remove_infinite_class(birth_death: ArrayLike) -> NDArray[np.float64]:
    """Remove exactly one point with minimum birth; break ties by max death.

    This mirrors Rust's `remove_infinite_class`. It does NOT identify an
    arbitrary row whose death is infinite, nor remove the longest-lived row.
    """
    points = _points(birth_death)
    if len(points) == 0:
        return points
    best = 0
    best_birth, best_death = points[0]
    for i in range(1, len(points)):
        birth, death = points[i]
        if birth < best_birth or (birth == best_birth and death > best_death):
            best, best_birth, best_death = i, birth, death
    return np.delete(points, best, axis=0)


def read_diagram(path: str | Path, remove_h0: bool = False) -> NDArray[np.float64]:
    """Read a no-header CSV of (birth, death), return (birth, persistence)."""
    rows: list[tuple[float, float]] = []
    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row_number, row in enumerate(reader, start=1):
            if len(row) < 2:
                raise ValueError(f"{path}: row {row_number} needs at least two columns")
            try:
                birth, death = float(row[0]), float(row[1])
            except ValueError as exc:
                raise ValueError(f"{path}: invalid number in row {row_number}") from exc
            if not np.isfinite(birth) or not np.isfinite(death):
                raise ValueError(f"{path}: non-finite value in row {row_number}")
            rows.append((birth, death))
    data = _points(rows)
    if remove_h0:
        data = remove_infinite_class(data)
    data[:, 1] -= data[:, 0]
    return data


def load_spacing(path: str | Path) -> dict[str, float]:
    """Read headered CSV: filename in column 0, spacing in column 2."""
    spacing: dict[str, float] = {}
    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader, None)  # Rust's csv::Reader defaults to has_headers(true)
        for row_number, row in enumerate(reader, start=2):
            if len(row) < 3:
                raise ValueError(f"{path}: row {row_number} needs at least three columns")
            try:
                value = float(row[2])
            except ValueError as exc:
                raise ValueError(f"{path}: invalid spacing in row {row_number}") from exc
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{path}: spacing must be finite and positive in row {row_number}")
            spacing[row[0]] = value
    return spacing


def collect_csv_files(
    root: str | Path,
    diagram_suffix: str | None = None,
    exclude: Iterable[str | Path] = (),
) -> list[Path]:
    """Recursively collect CSV paths, sorted, excluding spacing/output paths."""
    root = Path(root)
    excluded = {Path(path).resolve() for path in exclude}
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    paths = [
        p for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() == ".csv"
        and (diagram_suffix is None or p.name.endswith(diagram_suffix))
        and p.resolve() not in excluded
    ]
    return sorted(paths)


def save_images_to_csv(
    filenames: Iterable[str], images: Iterable[ArrayLike], output_path: str | Path
) -> None:
    """Write rows [filename, pixel_1, ...] with no header, like the Rust CLI."""
    with Path(output_path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for filename, image in zip(filenames, images, strict=True):
            writer.writerow([filename, *(repr(float(v)) for v in np.ravel(image))])
