"""Flexible probability-normalized, pixel-integrated persistence images.

Default stability-oriented ramp is unchanged in the optimized stable API.
Arbitrary weights and correlated covariances are separate *reference* options.
Only the diagonal covariance is integrated analytically by Gaussian CDF
differences. Off-diagonal covariance uses deterministic Gauss-Legendre
quadrature and is NOT certified by the fast backend's error theorem.
"""
from __future__ import annotations

from math import pi, sqrt
from typing import Callable, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import Bounds, _points, _validate_params
from .stable import _axes, _cdf_intervals

WeightFunction = Callable[[NDArray[np.float64]], ArrayLike]


def persistence_weights(
    persistence: ArrayLike, weight: str | WeightFunction = "ramp",
    *, weight_scale: float = 1.0, weight_params: Mapping[str, float] | None = None
) -> NDArray[np.float64]:
    """Evaluate nonnegative weights on persistence coordinates.

    ramp: min(1,max(0,p)/weight_scale). Globally bounded and Lipschitz.
    linear_ramp: zero to one between start and end (start>=0).
    power: p**exponent (exponent>=1), unbounded; no global W1 bound
      follows from the bounded-ramp theorem.
    callable: receives a vector of persistence values; no stability
      property is inferred, even if all returned weights are nonnegative.
    """
    p = np.asarray(persistence, dtype=np.float64)
    if np.any(~np.isfinite(p)) or np.any(p < 0):
        raise ValueError("Persistence must be finite and nonnegative")
    params = dict(weight_params or {})
    if callable(weight):
        vals = np.asarray(weight(p), dtype=np.float64)
    elif weight == "ramp":
        if not np.isfinite(weight_scale) or weight_scale <= 0:
            raise ValueError("weight_scale must be finite and positive")
        if params:
            raise ValueError("ramp uses weight_scale, not weight_params")
        vals = np.minimum(p / weight_scale, 1.0)
    elif weight == "linear_ramp":
        if set(params) - {"start", "end"}:
            raise ValueError("Unknown linear_ramp weight parameter")
        start = float(params.get("start", 0.0))
        end = float(params.get("end", weight_scale))
        if not np.isfinite(start) or not np.isfinite(end) or start < 0 or end <= start:
            raise ValueError("linear_ramp requires 0<=start<end")
        vals = np.clip((p - start) / (end - start), 0.0, 1.0)
    elif weight == "power":
        if set(params) - {"exponent"}:
            raise ValueError("Unknown power weight parameter")
        exponent = float(params.get("exponent", 1.0))
        if not np.isfinite(exponent) or exponent < 1:
            raise ValueError("power exponent must be finite and >=1")
        vals = p**exponent
    else:
        raise ValueError("weight must be ramp, linear_ramp, power or callable")
    if vals.ndim == 0:
        vals = np.full(p.shape, float(vals), dtype=np.float64)
    if vals.shape != p.shape or not np.all(np.isfinite(vals)) or np.any(vals < 0):
        raise ValueError("Weights must be finite, nonnegative and have one value per point")
    return vals


def _checked_covariance(
    sigma_x: float, sigma_y: float, covariance: ArrayLike | None
) -> NDArray[np.float64]:
    _validate_params(1, 1, sigma_x, sigma_y, 1)
    sigma = (np.diag((sigma_x**2, sigma_y**2)) if covariance is None
             else np.asarray(covariance, dtype=np.float64))
    if sigma.shape != (2, 2) or not np.all(np.isfinite(sigma)):
        raise ValueError("Covariance must be a finite 2x2 matrix")
    if not np.allclose(sigma, sigma.T, atol=1e-14, rtol=1e-12):
        raise ValueError("Covariance must be symmetric")
    if sigma[0, 0] <= 0 or sigma[1, 1] <= 0 or np.linalg.det(sigma) <= 0:
        raise ValueError("Covariance must be positive definite")
    return sigma


def flexible_persistence_image(
    diagram: ArrayLike, bounds: Bounds,
    nx: int = 10, ny: int = 8,
    sigma_x: float = 0.86, sigma_y: float = 0.63984,
    *, weight: str | WeightFunction = "ramp",
    weight_scale: float = 1.0,
    weight_params: Mapping[str, float] | None = None,
    covariance: ArrayLike | None = None,
    batch_size: int = 4096,
    quadrature_nodes: int = 48,
) -> NDArray[np.float64]:
    """Flattened (nx*ny,) birth-major weighted Gaussian pixel integrals.

    Diagonal covariance: analytic normal-CDF differences.
    Correlated covariance: deterministic Gaussian-Legendre quadrature
    along each x pixel and exact conditional-Gaussian y CDF differences.
    Correlated mode is *not* exact/certified; increase quadrature_nodes and
    compare to a higher-order run when kernels are narrow.
    The output is NOT divided by its total mass.
    """
    _validate_params(nx, ny, sigma_x, sigma_y, batch_size)
    pts = _points(diagram)
    if np.any(pts[:, 1] < 0):
        raise ValueError("Persistence must be nonnegative")
    xe, ye = _axes(bounds, nx, ny)
    cov = _checked_covariance(sigma_x, sigma_y, covariance)
    weights = persistence_weights(
        pts[:, 1], weight, weight_scale=weight_scale, weight_params=weight_params
    )
    out = np.zeros((nx, ny), dtype=np.float64)
    if len(pts) == 0 or not np.any(weights):
        return out.ravel(order="C")
    sx, sy = sqrt(float(cov[0, 0])), sqrt(float(cov[1, 1]))
    if cov[0, 1] == 0 and cov[1, 0] == 0:
        for start in range(0, len(pts), batch_size):
            chunk, w = pts[start:start+batch_size], weights[start:start+batch_size]
            if not np.any(w):
                continue
            a = _cdf_intervals(xe, chunk[:, 0], sx)
            d = _cdf_intervals(ye, chunk[:, 1], sy)
            out += (a * w[None, :]) @ d.T
        return out.ravel(order="C")
    if not isinstance(quadrature_nodes, int) or not (4 <= quadrature_nodes <= 512):
        raise ValueError("quadrature_nodes must be an integer from 4 to 512")
    nodes, coeffs = np.polynomial.legendre.leggauss(quadrature_nodes)
    cov_xy = float(cov[0, 1])
    beta = cov_xy / float(cov[0, 0])
    conditional_sigma = sqrt(float(cov[1, 1]) - cov_xy*beta)
    root2pi = sqrt(2*pi)
    for (b, p), w in zip(pts, weights):
        if w == 0:
            continue
        for i in range(nx):
            low, high = xe[i], xe[i+1]
            x = 0.5*(high-low)*nodes + 0.5*(high+low)
            z = (x-b)/sx
            density = np.exp(-0.5*z*z)/(root2pi*sx)
            cond_means = p + beta*(x-b)
            # Shape: (ny, quadrature_nodes), exact conditional y-pixel masses
            conditional = _cdf_intervals(ye, cond_means, conditional_sigma)
            out[i] += (w*0.5*(high-low))*(
                conditional @ (coeffs*density))
    return out.ravel(order="C")
