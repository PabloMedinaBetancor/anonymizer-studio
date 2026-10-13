"""Irreversible blur of a box, in place, on a BGR photo or on one Y, U or V plane of a video frame.

Ported from the earlier application, where it was tested at forced contrast. Three rules make it irreversible:
all geometry is measured on the box's short side; the content is pixelated into blocks the size of a character
before anything else; and the mask is drawn twice, a feathered outside first and then an opaque core, so the
feathering never lets the original through over what is being hidden.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import cv2
import numpy as np

# block: short side / block = pixelation block; gauss, feather, core: fractions of the short side.
PROFILE = {"block": 1.2, "gauss": 0.30, "feather": 0.10, "core": 0.06}
# The subtle profile (v2.2, the default of the engine's fit="tight"): the same hiding inside the box, but the soft edge and the opaque
# rim that PROFILE puts around the box (core + 2 feathers = 26 % of its short side per side) shrink to what a tight box needs (SUBTLE_EDGE).
SUBTLE_PROFILE = {"block": 1.2, "gauss": 0.30, "feather": 0.05, "core": 0.02}

GAUSSIAN, PIXELATE, SOLID = "gaussian", "pixelate", "solid"
STYLES = (GAUSSIAN, PIXELATE, SOLID)
ELLIPSE, RECT = "ellipse", "rect"

# Strength scales the blocks and the gaussian; 1.0 is the profile of the earlier application and also the floor. Measured on text plates
# of 30-700 px blurred with the photo margins, the worst residual at 1.0 is 1.2 (gaussian and pixelate); at 0.8 the
# gaussian already leaves 2.8, so below the profile of the earlier application the residual test (< 1.5) no longer holds.
DEFAULT_STRENGTH = 1.0
MIN_STRENGTH = 1.0
MAX_STRENGTH = 2.0
PIXELATE_BOOST = 1.5


def blur(
    plane: np.ndarray,
    box: Sequence[float],
    shape: str = RECT,
    style: str = GAUSSIAN,
    strength: float = DEFAULT_STRENGTH,
    profile: Optional[dict] = None,
) -> None:
    """Hide `box` (x0, y0, x1, y1 in pixels of `plane`) in place. `plane` is HxW or HxWxC, uint8 or uint16. `profile` is PROFILE (the
    default, the earlier application's) or SUBTLE_PROFILE."""
    profile = PROFILE if profile is None else profile
    if style not in STYLES:
        raise ValueError(f"unknown blur style: {style}")
    if not MIN_STRENGTH - 1e-9 <= strength <= MAX_STRENGTH + 1e-9:
        raise ValueError(f"blur strength {strength} outside [{MIN_STRENGTH}, {MAX_STRENGTH}]")
    height, width = plane.shape[:2]
    x0, y0, x1, y1 = [float(v) for v in box[:4]]
    x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(float(width), x1), min(float(height), y1)
    w, h = x1 - x0, y1 - y0
    if w < 1 or h < 1:
        return
    ref = min(w, h)
    feather = max(1.0, ref * profile["feather"])
    core = max(1.0, ref * profile["core"])
    pad = int(math.ceil(core + 4 * feather + 4))
    rx0, ry0 = max(0, int(x0) - pad), max(0, int(y0) - pad)
    rx1, ry1 = min(width, int(math.ceil(x1)) + pad), min(height, int(math.ceil(y1)) + pad)
    region = plane[ry0:ry1, rx0:rx1]
    rh, rw = region.shape[:2]
    if rw < 2 or rh < 2:
        return
    # Block edges count as structure in the residual test, so pixelate cuts its blocks from a blur 1.5 times
    # stronger (worst residual 1.1 on text plates of 30-700 px; from the blur strength of the earlier application, up to 3.5).
    smoothing = strength * (PIXELATE_BOOST if style == PIXELATE else 1.0)
    # All smoothing happens at scale g, where sigma is at most 6 px: the cost no longer grows with the box (a 600 px
    # face with a full-size gaussian took seconds per frame in the earlier application).
    sigma = max(1.5, ref * profile["gauss"] * smoothing)
    g = min(1.0, 6.0 / sigma)
    if style == SOLID:
        hidden = np.empty_like(region)
        hidden[...] = region.reshape(-1, *region.shape[2:]).mean(axis=0)
    else:
        hidden = _pixelate_and_smooth(region, ref, sigma, g, smoothing, profile)
        if style == PIXELATE:
            hidden = _mosaic(hidden, ref, strength)
    mask = _mask(rw, rh, (x0 - rx0, y0 - ry0, x1 - rx0, y1 - ry0), shape, ref, g, profile)
    if region.ndim == 3:
        mask = mask[:, :, None]
    mixed = hidden.astype(np.float32) * mask + region.astype(np.float32) * (1.0 - mask)
    info = np.iinfo(plane.dtype)
    region[...] = np.clip(np.rint(mixed), info.min, info.max).astype(plane.dtype)


def _pixelate_and_smooth(
    region: np.ndarray, ref: float, sigma: float, g: float, strength: float, profile: dict = PROFILE
) -> np.ndarray:
    """The blur of the earlier application: character-sized blocks, then a gaussian that melts them without widening the patch."""
    rh, rw = region.shape[:2]
    sw, sh = max(2, int(round(rw * g))), max(2, int(round(rh * g)))
    small_region = cv2.resize(region, (sw, sh), interpolation=cv2.INTER_AREA) if g < 1 else region
    block = max(2.0, ref * strength / profile["block"]) * (sw / rw)
    pw, ph = max(1, int(round(sw / max(1.0, block)))), max(1, int(round(sh / max(1.0, block))))
    blocks = cv2.resize(small_region, (pw, ph), interpolation=cv2.INTER_AREA)
    out = cv2.resize(blocks, (sw, sh), interpolation=cv2.INTER_CUBIC)
    sg = max(0.8, sigma * (sw / rw))
    k = int(math.ceil(sg * 3)) * 2 + 1
    out = cv2.GaussianBlur(out, (k, k), sg)
    if (sw, sh) != (rw, rh):
        out = cv2.resize(out, (rw, rh), interpolation=cv2.INTER_LINEAR)
    return out


def _mosaic(region: np.ndarray, ref: float, strength: float) -> np.ndarray:
    """Visible square blocks of half the short side, cut from the already blurred region: a mosaic of the
    original itself can be depixelated."""
    rh, rw = region.shape[:2]
    block = max(2.0, ref * strength / 2)
    pw, ph = max(1, int(round(rw / block))), max(1, int(round(rh / block)))
    blocks = cv2.resize(region, (pw, ph), interpolation=cv2.INTER_AREA)
    return cv2.resize(blocks, (rw, rh), interpolation=cv2.INTER_NEAREST)


def _mask(rw: int, rh: int, local_box, shape: str, ref: float, g: float = 1.0, profile: dict = PROFILE) -> np.ndarray:
    """Float32 0..1 mask of the region: feathered outside plus opaque core. The feathering is computed at scale `g`
    and enlarged; the core is drawn afterwards at full size, so it stays exactly 1."""
    x0, y0, x1, y1 = local_box
    feather = max(1.0, ref * profile["feather"])
    core = max(1.0, ref * profile["core"])
    outside = core + 2 * feather
    sw, sh = max(2, int(round(rw * g))), max(2, int(round(rh * g)))
    gx, gy = sw / rw, sh / rh
    m = np.zeros((sh, sw), np.float32)
    _fill(
        m, ((x0 - outside) * gx, (y0 - outside) * gy, (x1 + outside) * gx, (y1 + outside) * gy), shape, ref * 0.30 * g
    )
    fs = max(0.8, feather * g)
    k = int(math.ceil(fs * 3)) * 2 + 1
    m = cv2.GaussianBlur(m, (k, k), fs)
    if (sw, sh) != (rw, rh):
        m = cv2.resize(m, (rw, rh), interpolation=cv2.INTER_LINEAR)
    # The opaque core goes on after the smoothing: nothing of the original leaks inside the box.
    _fill(m, (x0 - core, y0 - core, x1 + core, y1 + core), shape, ref * 0.25)
    return m


def _fill(m: np.ndarray, box, shape: str, radius: float) -> None:
    x0, y0, x1, y1 = box
    if shape == ELLIPSE:
        centre = (int(round((x0 + x1) / 2)), int(round((y0 + y1) / 2)))
        axes = (max(1, int(round((x1 - x0) / 2))), max(1, int(round((y1 - y0) / 2))))
        cv2.ellipse(m, centre, axes, 0, 0, 360, 1.0, -1, cv2.LINE_AA)
        # LINE_AA leaves the rim half-filled; drawing again without it makes the inside exactly 1.
        cv2.ellipse(m, centre, axes, 0, 0, 360, 1.0, -1, cv2.LINE_8)
        return
    if shape != RECT:
        raise ValueError(f"unknown blur shape: {shape}")
    xa, ya, xb, yb = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
    r = int(max(0, min(radius, (xb - xa) / 2, (yb - ya) / 2)))
    if r <= 1:
        cv2.rectangle(m, (xa, ya), (xb, yb), 1.0, -1)
        return
    cv2.rectangle(m, (xa + r, ya), (xb - r, yb), 1.0, -1)
    cv2.rectangle(m, (xa, ya + r), (xb, yb - r), 1.0, -1)
    for cx, cy in ((xa + r, ya + r), (xb - r, ya + r), (xa + r, yb - r), (xb - r, yb - r)):
        cv2.circle(m, (cx, cy), r, 1.0, -1, cv2.LINE_8)


def residual(plane: np.ndarray, box: Sequence[float]) -> float:
    """High-frequency structure left inside a hidden box: the standard deviation of (image - its gaussian) inside
    the box minus a 10 % border. A good blur gives < 1.5; readable characters or features give much more."""
    height, width = plane.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in box[:4]]
    ref = min(x1 - x0, y1 - y0)
    inset = max(1, int(ref * 0.10))
    x0, y0, x1, y1 = max(0, x0 + inset), max(0, y0 + inset), min(width, x1 - inset), min(height, y1 - inset)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return 0.0
    z = plane[y0:y1, x0:x1]
    if z.ndim == 3:
        # a CMYK photo has four channels (an earlier audit, A9)
        z = cv2.cvtColor(z, cv2.COLOR_BGR2GRAY) if z.shape[2] == 3 else z.astype(np.float32).mean(axis=2)
    z = z.astype(np.float32)
    if plane.dtype == np.uint16:  # measured in 8-bit levels, as for every other photo (an earlier audit, A1)
        z = z / 257.0
    s = max(2.0, ref * 0.12)
    if s > 4.0:  # the same measure at a reduced scale keeps the cost bounded
        g = 4.0 / s
        z = cv2.resize(z, (max(3, int(z.shape[1] * g)), max(3, int(z.shape[0] * g))), interpolation=cv2.INTER_AREA)
        s = 4.0
    k = int(math.ceil(s * 3)) * 2 + 1
    return float((z - cv2.GaussianBlur(z, (k, k), s)).std())
