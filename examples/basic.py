"""Run with: python examples/basic.py (after pip install -e .)."""
import numpy as np
from persistence_images import Bounds, persistence_image

points = np.array([[0.1, 0.8], [0.4, 0.3], [0.6, 0.5]])
features = persistence_image(points, Bounds(0, 1, 0, 1), nx=32, ny=24,
                             sigma_x=0.2, sigma_y=0.2)
print(f"Image shape: {features.reshape(32, 24).shape}")
print(f"Total intensity: {features.sum():.6f}")
