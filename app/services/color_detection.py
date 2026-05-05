from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _rgb_to_hsv(r: float, g: float, b: float):
    mx = max(r, g, b)
    mn = min(r, g, b)
    diff = mx - mn

    if diff == 0:
        h = 0
    elif mx == r:
        h = (60 * ((g - b) / diff) + 360) % 360
    elif mx == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    else:
        h = (60 * ((r - g) / diff) + 240) % 360

    s = 0 if mx == 0 else diff / mx
    v = mx
    return h, s, v


def detect_dominant_color(image_path: str) -> str | None:
    image = Image.open(Path(image_path)).convert("RGB").resize((80, 80))
    arr = np.asarray(image, dtype=np.float32) / 255.0

    # Ignore very dark/very bright noise by focusing on mid-value pixels when possible.
    flat = arr.reshape(-1, 3)
    hsv = np.array([_rgb_to_hsv(*px) for px in flat], dtype=np.float32)

    sat = hsv[:, 1]
    val = hsv[:, 2]

    mask = (val > 0.15) & (sat > 0.15)
    if np.any(mask):
        sel = hsv[mask]
    else:
        sel = hsv

    h = float(np.median(sel[:, 0]))
    s = float(np.median(sel[:, 1]))
    v = float(np.median(sel[:, 2]))

    if v < 0.2:
        return "Black"
    if s < 0.12 and v > 0.85:
        return "White"
    if s < 0.12:
        return "Gray"

    if h < 15 or h >= 345:
        return "Red"
    if h < 45:
        return "Orange"
    if h < 70:
        return "Yellow"
    if h < 170:
        return "Green"
    if h < 260:
        return "Blue"
    if h < 315:
        return "Purple"
    if h < 345:
        return "Pink"

    return None
