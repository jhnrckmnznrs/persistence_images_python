"""Scikit-learn-compatible persistence-image transformer.

Call fit on training diagrams only: learned bounds are then frozen for all
transform calls, avoiding leakage and preserving a fixed feature map.
"""
from __future__ import annotations

from typing import Mapping
import numpy as np
from numpy.typing import ArrayLike

from .core import Bounds, _points, compute_bounds
from .stable import stable_batch_images
from .flexible import flexible_persistence_image

try:
    from sklearn.base import BaseEstimator, TransformerMixin
except ImportError:
    class BaseEstimator:
        """Lightweight fallback when the optional sklearn extra is missing."""
        def get_params(self, deep=True):
            return {name: getattr(self, name) for name in (
                "nx","ny","sigma_x","sigma_y","bounds","padding",
                "coordinates","weight","weight_scale","weight_params",
                "covariance","engine","absolute_error","batch_size",
                "quadrature_nodes")}
        def set_params(self, **params):
            for name, value in params.items():
                if name not in self.get_params():
                    raise ValueError(f"Unknown parameter {name!r}")
                setattr(self, name, value)
            return self

    class TransformerMixin:
        def fit_transform(self, X, y=None, **fit_params):
            return self.fit(X, y=y, **fit_params).transform(X)


def _as_diagrams(X, coordinates: str):
    if coordinates not in {"birth_persistence", "birth_death"}:
        raise ValueError("coordinates must be birth_persistence or birth_death")
    arr = None
    try:
        arr = np.asarray(X, dtype=np.float64)
    except (ValueError, TypeError):
        pass
    if arr is not None and arr.ndim == 2 and arr.shape[1] == 2:
        inputs = [arr]
    elif arr is not None and arr.ndim == 3 and arr.shape[2] == 2:
        inputs = [arr[i] for i in range(arr.shape[0])]
    else:
        if X is None:
            raise ValueError("Expected one diagram or an iterable of diagrams")
        inputs = list(X)
        # [] means no diagrams; the empty diagram must have shape (0,2)
        if len(inputs) == 0:
            return []
    result = []
    for d in inputs:
        pts = _points(d).copy()
        if coordinates == "birth_death":
            pts[:, 1] -= pts[:, 0]
        if np.any(pts[:, 1] < 0):
            raise ValueError("Persistence must be nonnegative")
        result.append(pts)
    return result


class PersistenceImageTransformer(BaseEstimator, TransformerMixin):
    """Fit fixed image bounds and vectorize persistence diagrams.

    Parameters are immutable conventions across a fitted transformer.
    Default bounded ramp + diagonal Gaussian uses the accelerated engines.
    Custom weights and non-diagonal covariance use the separate reference
    implementation and carry no automatic Wasserstein stability guarantee.

    X is a list of (n_i,2) diagrams, one (n,2) diagram, or a (k,n,2) array.
    transform always returns a (number_of_diagrams, nx*ny) ndarray.
    """

    def __init__(
        self, nx: int = 20, ny: int = 20,
        sigma_x: float = 0.2, sigma_y: float = 0.2,
        bounds: Bounds | None = None, padding: float = 0.05,
        coordinates: str = "birth_persistence",
        weight="ramp", weight_scale: float = 1.0,
        weight_params: Mapping[str, float] | None = None,
        covariance: ArrayLike | None = None,
        engine: str = "auto", absolute_error: float | None = None,
        batch_size: int = 4096, quadrature_nodes: int = 48,
    ):
        self.nx = nx
        self.ny = ny
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.bounds = bounds
        self.padding = padding
        self.coordinates = coordinates
        self.weight = weight
        self.weight_scale = weight_scale
        self.weight_params = weight_params
        self.covariance = covariance
        self.engine = engine
        self.absolute_error = absolute_error
        self.batch_size = batch_size
        self.quadrature_nodes = quadrature_nodes

    def fit(self, X, y=None):
        from .core import _validate_params
        _validate_params(self.nx, self.ny, self.sigma_x, self.sigma_y, self.batch_size)
        if not np.isfinite(self.padding) or self.padding < 0:
            raise ValueError("padding must be finite and nonnegative")
        diagrams = _as_diagrams(X, self.coordinates)
        if self.bounds is None:
            fitted = compute_bounds(diagrams)
            if fitted is None:
                raise ValueError("Cannot fit bounds from entirely empty diagrams; provide bounds")
            # Ensure a strictly positive pixel width even for one repeated point.
            dx = max(fitted.x_max - fitted.x_min, self.sigma_x)
            dy = max(fitted.y_max - fitted.y_min, self.sigma_y)
            xmid = 0.5*(fitted.x_max + fitted.x_min)
            ymid = 0.5*(fitted.y_max + fitted.y_min)
            xspan = max(fitted.x_max - fitted.x_min, self.sigma_x*1e-6)
            yspan = max(fitted.y_max - fitted.y_min, self.sigma_y*1e-6)
            # Preserve observed nondegenerate ranges, only expand by padding.
            fullx = xspan + 2*self.padding*dx
            fully = yspan + 2*self.padding*dy
            fitted = Bounds(
                xmid-0.5*fullx, xmid+0.5*fullx,
                ymid-0.5*fully, ymid+0.5*fully
            )
        else:
            if not isinstance(self.bounds, Bounds):
                raise ValueError("bounds must be a Bounds instance or None")
            fitted = self.bounds
        if fitted.x_min >= fitted.x_max or fitted.y_min >= fitted.y_max:
            raise ValueError("Fitted image bounds must have positive extent")
        self.bounds_ = fitted
        self.n_features_out_ = self.nx*self.ny
        return self

    def transform(self, X):
        if not hasattr(self, "bounds_"):
            raise RuntimeError("Call fit before transform")
        diagrams = _as_diagrams(X, self.coordinates)
        if not diagrams:
            return np.empty((0, self.n_features_out_), dtype=np.float64)
        optimized = (
            isinstance(self.weight, str) and self.weight == "ramp"
            and self.weight_params is None and self.covariance is None
        )
        if optimized:
            images = stable_batch_images(
                diagrams, self.bounds_, self.nx, self.ny,
                self.sigma_x, self.sigma_y,
                weight_scale=self.weight_scale, batch_size=self.batch_size,
                engine=self.engine, absolute_error=self.absolute_error
            )
        else:
            if self.engine not in ("auto", "dense"):
                raise ValueError(
                    "Custom weights or covariance require engine='auto' or 'dense'; "
                    "CDF shared/FFT certificates only support the bounded ramp "
                    "and diagonal covariance"
                )
            images = [
                flexible_persistence_image(
                    d, self.bounds_, self.nx, self.ny,
                    self.sigma_x, self.sigma_y,
                    weight=self.weight, weight_scale=self.weight_scale,
                    weight_params=self.weight_params, covariance=self.covariance,
                    batch_size=self.batch_size, quadrature_nodes=self.quadrature_nodes
                ) for d in diagrams
            ]
        return np.asarray(images, dtype=np.float64).reshape(
            len(diagrams), self.n_features_out_
        )

    def get_feature_names_out(self, input_features=None):
        if not hasattr(self, "bounds_"):
            raise RuntimeError("Call fit before get_feature_names_out")
        return np.asarray(
            [f"pi_x{i:03d}_y{j:03d}" for i in range(self.nx)
             for j in range(self.ny)], dtype=object
        )

    def transform_images(self, X):
        """Return a batch of (nx, ny) images instead of flattened vectors."""
        return self.transform(X).reshape((-1, self.nx, self.ny))
