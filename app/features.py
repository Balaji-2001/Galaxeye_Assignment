"""
Feature extraction for offline land-use tile classification.

Design choice (see design_note.md, Part 1): rather than depending on a
downloaded deep-net checkpoint (large, requires internet the first time,
awkward to vendor onto isolated hardware), we use a small set of
hand-engineered spectral + texture features computed directly from the RGB
tile. This is the classical remote-sensing approach for lightweight,
fully-offline classifiers and is trivially explainable to an analyst
("this tile was called 'water' because it's dark and blue-dominant").

Features (10-dim vector):
  0 mean_R, 1 mean_G, 2 mean_B      - channel means (0-1)
  3 std_R,  4 std_G,  5 std_B       - channel std devs (texture/uniformity)
  6 greenness   = mean_G - mean_R          (vegetation proxy, NDVI-ish)
  7 blueness    = mean_B - (mean_R+mean_G)/2   (water proxy)
  8 brightness  = mean of all channels      (cloud/bare-soil proxy)
  9 edge_density = mean absolute Laplacian of grayscale (built-up/texture proxy)
"""
from __future__ import annotations
import numpy as np
from PIL import Image

FEATURE_NAMES = [
    "mean_R", "mean_G", "mean_B",
    "std_R", "std_G", "std_B",
    "greenness", "blueness", "brightness", "edge_density",
]


def _laplacian_edge_density(gray: np.ndarray) -> float:
    """Cheap, dependency-free edge/texture measure (no scipy/cv2 needed)."""
    # simple discrete Laplacian via finite differences
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    # pad back to comparable scale and combine
    edge = np.abs(gx[:-1, :]) + np.abs(gy[:, :-1])
    return float(edge.mean())


def extract_features(img: Image.Image) -> np.ndarray:
    """img: PIL Image (any mode/size) -> 10-dim float32 feature vector."""
    img = img.convert("RGB").resize((64, 64))
    arr = np.asarray(img).astype(np.float32) / 255.0  # HxWx3, 0-1

    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mean_r, mean_g, mean_b = r.mean(), g.mean(), b.mean()
    std_r, std_g, std_b = r.std(), g.std(), b.std()

    greenness = mean_g - mean_r
    blueness = mean_b - (mean_r + mean_g) / 2.0
    brightness = (mean_r + mean_g + mean_b) / 3.0

    gray = arr.mean(axis=2)
    edge_density = _laplacian_edge_density(gray)

    return np.array(
        [mean_r, mean_g, mean_b, std_r, std_g, std_b,
         greenness, blueness, brightness, edge_density],
        dtype=np.float32,
    )
