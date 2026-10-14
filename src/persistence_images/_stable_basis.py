"""Reusable exact weighted Gaussian CDF-difference bases for stable images.

For a fixed pixel grid, bandwidths and weight scale:
    image = A @ C @ D
where C counts persistence points at distinct (birth, persistence) pairs,
A integrates the birth-axis Gaussian per pixel, and D integrates the
persistence-axis Gaussian per pixel and includes the persistence ramp weight.
The factorization is exact in real arithmetic; roundoff depends on summation
order. Pixel values are not renormalized.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Sequence
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import Bounds, _points, _validate_params
from .stable import _axes, _cdf_intervals, stable_persistence_image

MAX_COUNT_CELLS = 2_000_000
MAX_BASIS_ELEMENTS = 16_000_000


@dataclass(frozen=True)
class StableCdfBasis:
    """CDF bases shared only when bounds, bandwidths, and weight scale match."""

    births: NDArray[np.float64]
    persistences: NDArray[np.float64]
    a: NDArray[np.float64]
    d: NDArray[np.float64]
    max_cells: int = MAX_COUNT_CELLS

    @classmethod
    def from_diagrams(
        cls, diagrams: Sequence[ArrayLike], bounds: Bounds,
        nx: int, ny: int, sigma_x: float, sigma_y: float,
        *, weight_scale: float = 1.0,
        max_cells: int = MAX_COUNT_CELLS,
        max_basis_elements: int = MAX_BASIS_ELEMENTS,
    ) -> "StableCdfBasis":
        _validate_params(nx, ny, sigma_x, sigma_y, 1)
        if not np.isfinite(weight_scale) or weight_scale <= 0:
            raise ValueError("weight_scale must be finite and positive")
        if max_cells < 1 or max_basis_elements < 1:
            raise ValueError("Memory caps must be positive")
        xe, ye = _axes(bounds, nx, ny)
        points = [_points(diagram) for diagram in diagrams]
        if any(np.any(p[:, 1] < 0) for p in points):
            raise ValueError("Stable images require nonnegative persistence")
        nonempty = [p[p[:, 1] > 0] for p in points if len(p)]
        nonempty = [p for p in nonempty if len(p)]
        if nonempty:
            joined = np.concatenate(nonempty)
            births = np.unique(joined[:, 0])
            persistences = np.unique(joined[:, 1])
        else:
            births = np.empty(0, dtype=np.float64)
            persistences = np.empty(0, dtype=np.float64)
        nb, npers = len(births), len(persistences)
        if nb * npers > max_cells or nx * nb + ny * npers > max_basis_elements:
            raise ValueError("Stable CDF basis exceeds memory limits")
        a = _cdf_intervals(xe, births, sigma_x)
        d = (_cdf_intervals(ye, persistences, sigma_y)
             * np.minimum(persistences / weight_scale, 1.0)[None, :]).T
        return cls(births, persistences, a, d, max_cells)

    def coordinate_counts(self) -> tuple[int, int]:
        return len(self.births), len(self.persistences)

    def transform(self, diagram: ArrayLike) -> NDArray[np.float64]:
        points = _points(diagram)
        if np.any(points[:, 1] < 0):
            raise ValueError("Stable images require nonnegative persistence")
        nx, nb = self.a.shape
        npers, ny = self.d.shape
        points = points[points[:, 1] > 0]
        if len(points) == 0:
            return np.zeros(nx * ny, dtype=np.float64)
        if nb == 0 or npers == 0:
            raise ValueError("Point absent from CDF basis")
        bi = np.searchsorted(self.births, points[:, 0])
        pi = np.searchsorted(self.persistences, points[:, 1])
        if (np.any(bi >= nb) or np.any(pi >= npers)
                or not np.array_equal(self.births[bi], points[:, 0])
                or not np.array_equal(self.persistences[pi], points[:, 1])):
            raise ValueError("Diagram contains coordinates not in CDF basis")
        counts = np.bincount(
            bi * npers + pi, minlength=nb * npers
        ).reshape(nb, npers)
        lc = nx * nb * npers + nx * npers * ny
        rc = nb * npers * ny + nx * nb * ny
        if lc <= rc:
            return ((self.a @ counts) @ self.d).ravel(order="C")
        return (self.a @ (counts @ self.d)).ravel(order="C")


def _candidate(ds: list[NDArray[np.float64]], nx: int, ny: int) -> bool:
    if len(ds) < 2 or nx * ny < 256 or sum(len(d) for d in ds) < 512:
        return False
    segments = [d[d[:, 1] > 0][:512] for d in ds if len(d)]
    segments = [seg for seg in segments if len(seg)]
    if not segments:
        return False
    sample = np.concatenate(segments)[:2048]
    nb = len(np.unique(sample[:, 0]))
    npers = len(np.unique(sample[:, 1]))
    return nb * npers <= 8 * len(sample)


def _economical(
    basis: StableCdfBasis, ds: list[NDArray[np.float64]], nx: int, ny: int
) -> bool:
    nb, npers = basis.coordinate_counts()
    n = sum(len(d) for d in ds)
    shared = (12 * (nx * nb + ny * npers) + 8 * n
              + len(ds) * (nb * npers * min(nx, ny) + nx * ny * min(nb, npers)))
    dense = n * (nx * ny + 4 * (nx + ny))
    return 10 * shared < 6 * dense


def stable_batch_images_with_bases(
    diagrams: Sequence[ArrayLike], bounds: Bounds,
    nx: int = 10, ny: int = 8, sigma_x: float = 0.86, sigma_y: float = 0.63984,
    *, spacings: Sequence[float] | None = None,
    weight_scale: float = 1.0, batch_size: int = 4096,
    engine: str = "auto",
) -> list[NDArray[np.float64]]:
    """Stable batch with optional reusable CDF bases; input already spacing-scaled."""
    _validate_params(nx, ny, sigma_x, sigma_y, batch_size)
    if not np.isfinite(weight_scale) or weight_scale <= 0:
        raise ValueError("weight_scale must be finite and positive")
    if engine not in {"auto", "dense", "factorized", "shared"}:
        raise ValueError("Stable images support auto, dense, factorized or shared")
    points = [_points(d) for d in diagrams]
    if any(np.any(d[:, 1] < 0) for d in points):
        raise ValueError("Stable images require nonnegative persistence")
    spacing = [1.0] * len(points) if spacings is None else list(spacings)
    if len(spacing) != len(points):
        raise ValueError("spacings length mismatch")
    if any(not np.isfinite(s) or s <= 0 or not np.isfinite(sigma_x * s)
           or not np.isfinite(sigma_y * s) or sigma_x * s <= 0
           or sigma_y * s <= 0 for s in spacing):
        raise ValueError("Scaled bandwidths must be finite and positive")
    groups: dict[float, list[int]] = defaultdict(list)
    for i, s in enumerate(spacing):
        groups[float(s)].append(i)
    outputs: list[NDArray[np.float64] | None] = [None] * len(points)
    for s, idx in groups.items():
        ds = [points[i] for i in idx]
        sx, sy = sigma_x * s, sigma_y * s
        use_shared = engine == "shared" or (engine == "auto" and _candidate(ds, nx, ny))
        if use_shared:
            try:
                basis = StableCdfBasis.from_diagrams(
                    ds, bounds, nx, ny, sx, sy, weight_scale=weight_scale
                )
                if engine == "shared" or _economical(basis, ds, nx, ny):
                    for i in idx:
                        outputs[i] = basis.transform(points[i])
                    continue
            except ValueError:
                if engine == "shared":
                    raise
        for i in idx:
            outputs[i] = stable_persistence_image(
                points[i], bounds, nx, ny, sx, sy,
                weight_scale=weight_scale, batch_size=batch_size,
                engine="dense" if engine == "shared" else engine,
            )
    return [out for out in outputs if out is not None]
