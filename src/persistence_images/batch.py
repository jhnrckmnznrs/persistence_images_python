"""Reusable Gaussian bases for batches of diagrams with shared pixel grids.

The algebra is I = A @ C @ D, where C counts exact coordinate pairs.
For multiple diagrams, A and D are constructed once for each distinct
bandwidth (i.e. voxel spacing) and reused across samples. No weighting,
normalization, or pixel-area integration is introduced.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import Bounds, _pixel_centers, _points, _validate_params, persistence_image


@dataclass(frozen=True)
class SharedGaussianBasis:
    """Immutable Gaussian basis reusable for diagrams at fixed bounds/bandwidths.

    ``births`` and ``persistences`` are the sorted union of coordinates in the
    training batch. ``transform`` raises if a new diagram contains a coordinate
    absent from that union; use ``from_diagrams`` to build a new basis.
    """
    births: NDArray[np.float64]
    persistences: NDArray[np.float64]
    a: NDArray[np.float64]   # (nx, n_births)
    d: NDArray[np.float64]   # (n_persistences, ny)
    max_cells: int = 2_000_000

    @classmethod
    def from_diagrams(
        cls, diagrams: Sequence[ArrayLike], bounds: Bounds,
        nx: int, ny: int, sigma_x: float, sigma_y: float, *,
        max_cells: int = 2_000_000, max_basis_elements: int = 16_000_000,
    ) -> "SharedGaussianBasis":
        _validate_params(nx, ny, sigma_x, sigma_y, 1)
        if max_cells < 1 or max_basis_elements < 1:
            raise ValueError("Memory caps must be positive")
        points = [_points(d) for d in diagrams]
        nonempty = [p for p in points if len(p)]
        births = np.unique(np.concatenate([p[:, 0] for p in nonempty])) if nonempty else np.empty(0)
        persistences = np.unique(np.concatenate([p[:, 1] for p in nonempty])) if nonempty else np.empty(0)
        nb, npers = len(births), len(persistences)
        if nb * npers > max_cells or nx * nb + ny * npers > max_basis_elements:
            raise ValueError("Shared basis exceeds the memory cap")
        xs, ys = _pixel_centers(bounds, nx, ny)
        a = np.exp(-0.5 * ((xs[:, None] - births[None, :]) / sigma_x) ** 2)
        d = np.exp(-0.5 * ((persistences[:, None] - ys[None, :]) / sigma_y) ** 2)
        return cls(births, persistences, a, d, max_cells)

    def transform(self, diagram: ArrayLike) -> NDArray[np.float64]:
        points = _points(diagram)
        nx, nb = self.a.shape
        npers, ny = self.d.shape
        if len(points) == 0:
            return np.zeros(nx * ny, dtype=np.float64)
        bi = np.searchsorted(self.births, points[:, 0])
        pi = np.searchsorted(self.persistences, points[:, 1])
        if (np.any(bi >= nb) or np.any(pi >= npers)
                or not np.array_equal(self.births[bi], points[:, 0])
                or not np.array_equal(self.persistences[pi], points[:, 1])):
            raise ValueError("Diagram contains coordinates outside the fitted shared basis")
        counts = np.bincount(bi * npers + pi, minlength=nb * npers).reshape(nb, npers)
        if nx * nb * npers + nx * npers * ny <= nb * npers * ny + nx * nb * ny:
            return ((self.a @ counts) @ self.d).ravel(order="C")
        return (self.a @ (counts @ self.d)).ravel(order="C")


def batch_persistence_images(
    diagrams: Sequence[ArrayLike], bounds: Bounds,
    nx: int = 10, ny: int = 8, sigma_x: float = 0.86, sigma_y: float = 0.63984,
    *, spacings: Sequence[float] | None = None,
    engine: str = "auto", absolute_error: float | None = None,
    batch_size: int = 4096,
) -> list[NDArray[np.float64]]:
    """Compute images while reusing one basis per distinct spacing when useful.

    Diagrams are already in (scaled birth, scaled persistence) coordinates.
    ``spacings`` changes only bandwidths, as in the Rust CLI. With ``auto``
    and no tolerance the result is exact in real arithmetic; setting
    ``absolute_error`` allows local truncation with the advertised error bound.
    ``shared`` forces basis reuse and rejects batches exceeding memory limits.
    """
    _validate_params(nx, ny, sigma_x, sigma_y, batch_size)
    if engine not in {"auto", "dense", "factorized", "shared", "sparse", "grid_fft", "interp_fft"}:
        raise ValueError("Invalid batch engine")
    points = [_points(d) for d in diagrams]
    spacing = [1.0] * len(points) if spacings is None else list(spacings)
    if len(points) != len(spacing):
        raise ValueError("spacings must have one element per diagram")
    if any(not np.isfinite(s) or s <= 0 for s in spacing):
        raise ValueError("spacings must be finite and positive")
    groups: dict[float, list[int]] = defaultdict(list)
    for i, value in enumerate(spacing):
        groups[float(value)].append(i)
    result: list[NDArray[np.float64] | None] = [None] * len(points)
    for scale, indices in groups.items():
        subset = [points[i] for i in indices]
        sx, sy = sigma_x * scale, sigma_y * scale
        # Conservative screening avoids expensive global uniques on continuous
        # data. The final memory cap and operation-count check remain mandatory.
        use_shared = engine == "shared"
        if engine == "auto" and len(indices) >= 2 and nx * ny >= 256:
            samples = [p[:min(512, len(p))] for p in subset]
            nonempty = [p for p in samples if len(p)]
            if nonempty:
                sample = np.concatenate(nonempty, axis=0)[:2048]
                n = sum(len(p) for p in subset)
                ub = len(np.unique(sample[:, 0]))
                up = len(np.unique(sample[:, 1]))
                use_shared = (n >= 512 and ub * up <= 8 * len(sample))
        if use_shared:
            try:
                basis = SharedGaussianBasis.from_diagrams(subset, bounds, nx, ny, sx, sy)
                nb, npers = len(basis.births), len(basis.persistences)
                total_n = sum(len(p) for p in subset)
                # Integer-operation proxy; building bases and histograms costs
                # more than multiply-adds, so use a conservative threshold.
                shared_cost = (12 * (nx * nb + ny * npers)
                               + 8 * total_n
                               + len(indices) * (nb * npers * min(nx, ny)
                                                 + nx * ny * min(nb, npers)))
                dense_cost = total_n * (nx * ny + 4 * (nx + ny))
                if engine == "shared" or shared_cost < 0.6 * dense_cost:
                    for i in indices:
                        result[i] = basis.transform(points[i])
                    continue
            except ValueError:
                if engine == "shared":
                    raise
        for i in indices:
            local_engine = "auto" if engine == "shared" else engine
            result[i] = persistence_image(
                points[i], bounds, nx, ny, sx, sy,
                engine=local_engine, absolute_error=absolute_error,
                batch_size=batch_size,
            )
    return [v for v in result if v is not None]
