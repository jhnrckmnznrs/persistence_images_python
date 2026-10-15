"""Python implementation of jhnrckmnznrs/persistence_images."""

from .batch import SharedGaussianBasis, batch_persistence_images
from .stable import stable_persistence_image, stable_batch_images
from ._stable_basis import StableCdfBasis
from ._stable_interp_fft import stable_interp_fft_image, StableIntegratedFft, StableFftCertificate, select_stable_fft_stencil
from .core import Bounds, compute_bounds, persistence_image, persistence_image_reference
from .dataset import SpacingDiagram, apply_spacing, load_dataset, to_scaled_diagrams
from .io import (
    collect_csv_files,
    load_spacing,
    read_diagram,
    remove_infinite_class,
    save_images_to_csv,
)

__all__ = [
    "Bounds", "SharedGaussianBasis", "batch_persistence_images", "SpacingDiagram", "apply_spacing", "collect_csv_files",
    "compute_bounds", "load_dataset", "load_spacing", "persistence_image",
    "persistence_image_reference", "read_diagram", "remove_infinite_class",
    "save_images_to_csv", "to_scaled_diagrams", "stable_persistence_image", "stable_batch_images", "StableCdfBasis", "stable_interp_fft_image", "StableIntegratedFft", "StableFftCertificate", "select_stable_fft_stencil",
]
