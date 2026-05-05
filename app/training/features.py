from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def extract_feature_vector(image_path: Path, image_size: int = 64, gray_size: int = 32) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    rgb = image.resize((image_size, image_size))
    rgb_arr = np.asarray(rgb, dtype=np.float32) / 255.0

    hist_parts = []
    for c in range(3):
        hist, _ = np.histogram(rgb_arr[:, :, c], bins=16, range=(0.0, 1.0), density=True)
        hist_parts.append(hist.astype(np.float32))
    color_hist = np.concatenate(hist_parts, axis=0)

    gray = image.convert("L").resize((gray_size, gray_size))
    gray_arr = np.asarray(gray, dtype=np.float32).reshape(-1) / 255.0

    return np.concatenate([color_hist, gray_arr], axis=0).astype(np.float32)
