"""Optional Matplotlib diagram and pixel-integrated image plotting."""
from __future__ import annotations
import numpy as np
from .core import Bounds, _points


def plot_diagram(diagram, *, coordinates="birth_persistence", ax=None):
    """Plot birth-persistence coordinates (or convert birth-death input)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("Install .[viz] for Matplotlib plotting") from e
    points=_points(diagram).copy()
    if coordinates == "birth_death":
        points[:,1]-=points[:,0]
    elif coordinates != "birth_persistence":
        raise ValueError("Unknown coordinate convention")
    if np.any(points[:,1]<0):
        raise ValueError("Persistence must be nonnegative")
    if ax is None:
        _,ax=plt.subplots()
    if len(points):
        ax.scatter(points[:,0],points[:,1],s=16,alpha=0.7)
    ax.axhline(0,color="gray",linewidth=0.8)
    ax.set_xlabel("Birth")
    ax.set_ylabel("Persistence")
    return ax


def plot_image(image, bounds:Bounds, nx:int, ny:int, *, ax=None, cmap="viridis",
               colorbar:bool=True):
    """Show (nx,ny) Gaussian pixel masses in birth-persistence orientation."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("Install .[viz] for Matplotlib plotting") from e
    data=np.asarray(image,dtype=np.float64)
    if data.size!=nx*ny:
        raise ValueError("Image size does not match grid")
    data=data.reshape(nx,ny)
    if ax is None:
        _,ax=plt.subplots()
    im=ax.imshow(data.T,origin="lower",aspect="auto",
                 extent=(bounds.x_min,bounds.x_max,bounds.y_min,bounds.y_max),
                 cmap=cmap,interpolation="nearest")
    ax.set_xlabel("Birth")
    ax.set_ylabel("Persistence")
    if colorbar:
        ax.figure.colorbar(im,ax=ax,label="Pixel-integrated Gaussian mass")
    return ax
