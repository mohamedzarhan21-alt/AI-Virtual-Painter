"""AI Virtual Painter — a holographic air-drawing studio.

A single-file, production-quality desktop application. The user paints in the air
with an index finger while MediaPipe tracks the hand in real time and a complete
raster painting engine turns the motion into digital artwork.

Run:
    python ai_virtual_painter.py

Optional environment overrides:
    AIVP_CAMERA_INDEX   Camera device index                        (default: 0)
    AIVP_CAMERA_WIDTH   Requested capture width                    (default: 1280)
    AIVP_CAMERA_HEIGHT  Requested capture height                   (default: 720)
    AIVP_HAND_MODEL     Path to an existing hand_landmarker.task bundle
    AIVP_MODEL_CACHE    Directory used to cache the downloaded model bundle
    AIVP_OUTPUT_DIR     Directory artwork is exported to
    AIVP_LOG_LEVEL      Logging level name                         (default: INFO)

The MediaPipe hand-landmark bundle is a ~7.5 MB binary published by Google. It is
not vendored with this file; on first launch it is fetched once from the official
Google Cloud Storage endpoint into a per-user cache directory. Point
``AIVP_HAND_MODEL`` at a local copy to run entirely offline.
"""

from __future__ import annotations

import os

# MediaPipe/TFLite emit verbose native logs at import time; silence them before the
# native extension is loaded so the console stays usable.
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_LOGGING_MIN_SEVERITY", "3")

import contextlib
import logging
import math
import random
import sys
import threading
import time
import tkinter as tk
import urllib.error
import urllib.request
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum, IntEnum, auto
from itertools import pairwise
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Final

import customtkinter as ctk
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image, ImageDraw, ImageFont, ImageTk

LOGGER: Final = logging.getLogger("ai_virtual_painter")

# =============================================================================
# SECTION 1 — Configuration
# =============================================================================

#: Official Google-hosted MediaPipe hand landmarker bundle (float16, revision 1).
HAND_MODEL_URL: Final[str] = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_MODEL_FILENAME: Final[str] = "hand_landmarker.task"
#: Smallest plausible size of the bundle; guards against truncated downloads.
HAND_MODEL_MIN_BYTES: Final[int] = 4_000_000


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    """Read an integer of at least ``minimum`` from the environment."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        LOGGER.warning("Ignoring non-integer %s=%r", name, raw)
        return default
    if parsed < minimum:
        LOGGER.warning("Ignoring out-of-range %s=%r", name, raw)
        return default
    return parsed


#: Optional application icon, looked for beside this file. Purely decorative: it is
#: shown during the launch sequence and used as the window icon when present.
ICON_FILENAME: Final[str] = "AI Virtual Painter.png"


def find_app_icon() -> Path | None:
    """Locate the application icon, or return ``None`` when it is absent."""
    override = os.environ.get("AIVP_ICON")
    candidate = Path(override) if override else Path(__file__).resolve().parent / ICON_FILENAME
    return candidate if candidate.is_file() else None


def _default_cache_dir() -> Path:
    """Return the per-user directory used to cache the landmark model bundle."""
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "AIVirtualPainter" / "models"
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg_cache) if xdg_cache else Path.home() / ".cache"
    return base / "ai-virtual-painter" / "models"


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Immutable runtime configuration resolved from the environment."""

    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    canvas_width: int = 1280
    canvas_height: int = 720
    #: Width the frame is downscaled to before hand inference (accuracy/speed trade).
    inference_width: int = 480
    target_fps: int = 60
    model_path: Path = field(default_factory=lambda: _default_cache_dir() / HAND_MODEL_FILENAME)
    output_dir: Path = field(default_factory=lambda: Path.home() / "Pictures" / "AI Virtual Painter")
    #: Multiplier applied when exporting a high-resolution render.
    export_scale: int = 3

    @classmethod
    def from_environment(cls) -> AppConfig:
        """Build a configuration from ``AIVP_*`` environment variables."""
        override = os.environ.get("AIVP_HAND_MODEL")
        cache_dir = Path(os.environ.get("AIVP_MODEL_CACHE") or _default_cache_dir())
        model_path = Path(override) if override else cache_dir / HAND_MODEL_FILENAME
        output_override = os.environ.get("AIVP_OUTPUT_DIR")
        output_dir = Path(output_override) if output_override else Path.home() / "Pictures" / "AI Virtual Painter"
        return cls(
            camera_index=_env_int("AIVP_CAMERA_INDEX", 0, minimum=0),
            camera_width=_env_int("AIVP_CAMERA_WIDTH", 1280),
            camera_height=_env_int("AIVP_CAMERA_HEIGHT", 720),
            model_path=model_path,
            output_dir=output_dir,
        )


def configure_logging() -> None:
    """Install a concise console logging handler."""
    level_name = os.environ.get("AIVP_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# =============================================================================
# SECTION 2 — Theme
# =============================================================================

Bgr = tuple[int, int, int]


def hex_to_bgr(value: str) -> Bgr:
    """Convert ``#RRGGBB`` to an OpenCV BGR triple."""
    text = value.lstrip("#")
    if len(text) != 6:
        raise ValueError(f"Expected #RRGGBB, got {value!r}")
    red, green, blue = (int(text[i : i + 2], 16) for i in (0, 2, 4))
    return (blue, green, red)


@dataclass(frozen=True, slots=True)
class Theme:
    """The single source of truth for every colour used by the interface.

    The direction is deliberate and committed to end-to-end: a near-black
    cinematic void, one dominant electric-cyan instrument colour, a magenta
    signal accent and an amber warning tone — Iron-Man-style FUI read through a
    Nothing-OS restraint filter. No other hex value appears anywhere else.
    """

    void: Bgr = hex_to_bgr("#04070B")
    glass: Bgr = hex_to_bgr("#0A121A")
    glass_light: Bgr = hex_to_bgr("#12202C")
    primary: Bgr = hex_to_bgr("#22E6FF")
    primary_deep: Bgr = hex_to_bgr("#0A7F96")
    accent: Bgr = hex_to_bgr("#FF2E88")
    amber: Bgr = hex_to_bgr("#FFC246")
    mint: Bgr = hex_to_bgr("#37F5A8")
    text_hi: Bgr = hex_to_bgr("#E9F7FB")
    text_mid: Bgr = hex_to_bgr("#8FA9B6")
    text_dim: Bgr = hex_to_bgr("#4C6472")
    line: Bgr = hex_to_bgr("#1B3B49")
    danger: Bgr = hex_to_bgr("#FF4D5E")

    #: Base spacing unit; every offset in the HUD is a multiple of this.
    unit: int = 8

    def spacing(self, steps: float) -> int:
        """Return ``steps`` spacing units, rounded to whole pixels."""
        return round(self.unit * steps)


THEME: Final = Theme()


# =============================================================================
# SECTION 3 — Maths, easing and signal filtering
# =============================================================================


def lerp(start: float, end: float, amount: float) -> float:
    """Linearly interpolate between two scalars."""
    return start + (end - start) * amount


def clamp(value: float, low: float, high: float) -> float:
    """Constrain ``value`` to the inclusive ``[low, high]`` range."""
    return low if value < low else high if value > high else value


def ease_out_cubic(t: float) -> float:
    """Cubic ease-out, used for panels that decelerate into place."""
    inverted = 1.0 - clamp(t, 0.0, 1.0)
    return 1.0 - inverted * inverted * inverted


def ease_out_back(t: float) -> float:
    """Ease-out with a slight overshoot, used for blooming controls."""
    c1 = 1.70158
    c3 = c1 + 1.0
    inverted = clamp(t, 0.0, 1.0) - 1.0
    return 1.0 + c3 * inverted**3 + c1 * inverted**2


def ease_in_out_sine(t: float) -> float:
    """Symmetric sine easing, used for idle breathing animations."""
    return -(math.cos(math.pi * clamp(t, 0.0, 1.0)) - 1.0) / 2.0


def mix_bgr(first: Bgr, second: Bgr, amount: float) -> Bgr:
    """Blend two BGR colours."""
    ratio = clamp(amount, 0.0, 1.0)
    return (
        int(lerp(first[0], second[0], ratio)),
        int(lerp(first[1], second[1], ratio)),
        int(lerp(first[2], second[2], ratio)),
    )


def hue_to_bgr(hue: float, saturation: float = 1.0, value: float = 1.0) -> Bgr:
    """Convert an HSV triple (hue in turns) to BGR without allocating arrays."""
    hue = hue % 1.0
    sector = hue * 6.0
    index = int(sector) % 6
    fraction = sector - int(sector)
    p = value * (1.0 - saturation)
    q = value * (1.0 - saturation * fraction)
    t = value * (1.0 - saturation * (1.0 - fraction))
    table = (
        (value, t, p),
        (q, value, p),
        (p, value, t),
        (p, q, value),
        (t, p, value),
        (value, p, q),
    )
    red, green, blue = table[index]
    return (int(blue * 255), int(green * 255), int(red * 255))


class Animated:
    """A scalar that eases towards a target with frame-rate independent damping."""

    __slots__ = ("_current", "_speed", "_target")

    def __init__(self, value: float = 0.0, speed: float = 12.0) -> None:
        self._current = float(value)
        self._target = float(value)
        self._speed = float(speed)

    @property
    def value(self) -> float:
        """The eased current value."""
        return self._current

    @property
    def target(self) -> float:
        """The value being eased towards."""
        return self._target

    def set(self, target: float) -> None:
        """Retarget the animation without snapping."""
        self._target = float(target)

    def snap(self, value: float) -> None:
        """Jump immediately to ``value``."""
        self._current = self._target = float(value)

    def update(self, delta_time: float) -> float:
        """Advance the animation by ``delta_time`` seconds and return the value."""
        if delta_time <= 0.0:
            return self._current
        factor = 1.0 - math.exp(-self._speed * delta_time)
        self._current += (self._target - self._current) * factor
        if abs(self._target - self._current) < 1e-4:
            self._current = self._target
        return self._current


class AnimatedColor:
    """A BGR colour that cross-fades smoothly whenever it is retargeted."""

    __slots__ = ("_channels",)

    def __init__(self, color: Bgr, speed: float = 9.0) -> None:
        self._channels = tuple(Animated(float(channel), speed) for channel in color)

    def set(self, color: Bgr) -> None:
        """Cross-fade towards ``color``."""
        for channel, target in zip(self._channels, color, strict=True):
            channel.set(float(target))

    def update(self, delta_time: float) -> Bgr:
        """Advance the fade and return the current colour."""
        return tuple(round(channel.update(delta_time)) for channel in self._channels)  # type: ignore[return-value]

    @property
    def value(self) -> Bgr:
        """The current colour without advancing time."""
        return tuple(round(channel.value) for channel in self._channels)  # type: ignore[return-value]


class OneEuroFilter:
    """The 1-Euro filter — low jitter when still, low lag when moving fast.

    Casiez, Roussel & Vogel (CHI 2012). It adapts its cutoff frequency to the
    signal's speed, which is exactly what fingertip painting needs: rock-steady
    dots when the hand hovers, and no rubber-banding on fast flicks.
    """

    __slots__ = ("_beta", "_d_cutoff", "_dx_prev", "_min_cutoff", "_t_prev", "_x_prev")

    def __init__(self, min_cutoff: float = 1.1, beta: float = 0.012, d_cutoff: float = 1.0) -> None:
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self._x_prev: float | None = None
        self._dx_prev = 0.0
        self._t_prev = 0.0

    @staticmethod
    def _alpha(cutoff: float, delta_time: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / delta_time)

    def reset(self) -> None:
        """Forget history so the next sample is taken verbatim."""
        self._x_prev = None
        self._dx_prev = 0.0

    def filter(self, value: float, timestamp: float) -> float:
        """Filter one sample taken at ``timestamp`` seconds."""
        if self._x_prev is None:
            self._x_prev = value
            self._t_prev = timestamp
            return value
        delta_time = max(timestamp - self._t_prev, 1e-3)
        self._t_prev = timestamp
        derivative = (value - self._x_prev) / delta_time
        alpha_d = self._alpha(self._d_cutoff, delta_time)
        self._dx_prev += alpha_d * (derivative - self._dx_prev)
        cutoff = self._min_cutoff + self._beta * abs(self._dx_prev)
        alpha = self._alpha(cutoff, delta_time)
        self._x_prev += alpha * (value - self._x_prev)
        return self._x_prev


class PointFilter:
    """Applies an independent 1-Euro filter to each axis of a 2-D point."""

    __slots__ = ("_x", "_y")

    def __init__(self, min_cutoff: float = 1.1, beta: float = 0.012) -> None:
        self._x = OneEuroFilter(min_cutoff, beta)
        self._y = OneEuroFilter(min_cutoff, beta)

    def reset(self) -> None:
        """Forget history for both axes."""
        self._x.reset()
        self._y.reset()

    def filter(self, point: tuple[float, float], timestamp: float) -> tuple[float, float]:
        """Filter a point sampled at ``timestamp`` seconds."""
        return (self._x.filter(point[0], timestamp), self._y.filter(point[1], timestamp))


def catmull_rom(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    """Evaluate a centripetal-style Catmull-Rom spline segment between p1 and p2.

    Catmull-Rom is used rather than a raw Bezier because it interpolates its
    control points: the drawn line passes exactly through the tracked fingertip
    positions while still producing C1-continuous curvature.
    """
    t2 = t * t
    t3 = t2 * t
    x = 0.5 * (
        (2.0 * p1[0])
        + (-p0[0] + p2[0]) * t
        + (2.0 * p0[0] - 5.0 * p1[0] + 4.0 * p2[0] - p3[0]) * t2
        + (-p0[0] + 3.0 * p1[0] - 3.0 * p2[0] + p3[0]) * t3
    )
    y = 0.5 * (
        (2.0 * p1[1])
        + (-p0[1] + p2[1]) * t
        + (2.0 * p0[1] - 5.0 * p1[1] + 4.0 * p2[1] - p3[1]) * t2
        + (-p0[1] + 3.0 * p1[1] - 3.0 * p2[1] + p3[1]) * t3
    )
    return (x, y)


def resample_polyline(points: Sequence[tuple[float, float]], spacing: float) -> list[tuple[float, float]]:
    """Resample a polyline to approximately uniform ``spacing`` between points."""
    if len(points) < 2:
        return list(points)
    output: list[tuple[float, float]] = [points[0]]
    carry = 0.0
    for start, end in pairwise(points):
        segment = math.dist(start, end)
        if segment <= 1e-6:
            continue
        position = spacing - carry
        while position <= segment:
            ratio = position / segment
            output.append((lerp(start[0], end[0], ratio), lerp(start[1], end[1], ratio)))
            position += spacing
        carry = (carry + segment) % spacing
    if math.dist(output[-1], points[-1]) > 1e-6:
        output.append(points[-1])
    return output


# =============================================================================
# SECTION 4 — Raster primitives
# =============================================================================
#
# Every layer in the application is a contiguous BGRA ``uint8`` array so that it
# can be handed to OpenCV without conversion. Straight (non-premultiplied) alpha
# is used because the transparent-PNG export needs the unmodified colour channels.

Rect = tuple[int, int, int, int]  # (x0, y0, x1, y1), half-open on the upper bound

#: Glass panels narrower than this skip the frost blur — see draw_glass_panel.
GLASS_BLUR_MIN_EDGE: Final[int] = 44


class BlendMode(Enum):
    """How a brush stamp merges into the stroke buffer."""

    #: Keep whichever contribution is most opaque — gives a stroke a flat,
    #: non-compounding opacity even where it crosses itself.
    MAX = auto()
    #: Regular source-over, so many faint stamps accumulate into density.
    OVER = auto()


def new_layer(width: int, height: int) -> np.ndarray:
    """Allocate a transparent BGRA layer."""
    return np.zeros((height, width, 4), dtype=np.uint8)


def union_rect(first: Rect | None, second: Rect | None) -> Rect | None:
    """Return the smallest rectangle containing both inputs."""
    if first is None:
        return second
    if second is None:
        return first
    return (
        min(first[0], second[0]),
        min(first[1], second[1]),
        max(first[2], second[2]),
        max(first[3], second[3]),
    )


def clip_rect(rect: Rect, width: int, height: int) -> Rect | None:
    """Clip a rectangle to a layer, returning ``None`` when nothing remains."""
    x0 = max(0, rect[0])
    y0 = max(0, rect[1])
    x1 = min(width, rect[2])
    y1 = min(height, rect[3])
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def composite_over(destination: np.ndarray, source: np.ndarray, rect: Rect | None = None) -> None:
    """Alpha-composite ``source`` over ``destination`` in place (BGRA, straight alpha)."""
    height, width = destination.shape[:2]
    area = clip_rect(rect or (0, 0, width, height), width, height)
    if area is None:
        return
    x0, y0, x1, y1 = area
    dst = destination[y0:y1, x0:x1]
    src = source[y0:y1, x0:x1]
    src_alpha = src[:, :, 3].astype(np.float32) / 255.0
    dst_alpha = dst[:, :, 3].astype(np.float32) / 255.0
    out_alpha = src_alpha + dst_alpha * (1.0 - src_alpha)
    safe_alpha = np.maximum(out_alpha, 1e-6)
    weight_src = (src_alpha / safe_alpha)[..., None]
    weight_dst = ((dst_alpha * (1.0 - src_alpha)) / safe_alpha)[..., None]
    colour = src[:, :, :3].astype(np.float32) * weight_src + dst[:, :, :3].astype(np.float32) * weight_dst
    dst[:, :, :3] = np.clip(colour, 0.0, 255.0).astype(np.uint8)
    dst[:, :, 3] = np.clip(out_alpha * 255.0, 0.0, 255.0).astype(np.uint8)


def erase_with(destination: np.ndarray, mask_layer: np.ndarray, rect: Rect | None = None) -> None:
    """Subtract ``mask_layer``'s alpha from ``destination``'s alpha in place."""
    height, width = destination.shape[:2]
    area = clip_rect(rect or (0, 0, width, height), width, height)
    if area is None:
        return
    x0, y0, x1, y1 = area
    dst_alpha = destination[y0:y1, x0:x1, 3].astype(np.int16)
    cut = mask_layer[y0:y1, x0:x1, 3].astype(np.int16)
    destination[y0:y1, x0:x1, 3] = np.clip(dst_alpha - cut, 0, 255).astype(np.uint8)


def _scaled_mask(mask: np.ndarray, opacity: float) -> np.ndarray | None:
    """Scale a uint8 coverage mask by ``opacity``; ``None`` means fully clear."""
    if opacity <= 0.004:
        return None
    if opacity >= 0.996:
        return mask
    return cv2.convertScaleAbs(mask, alpha=opacity)


def blend_color_mask(region: np.ndarray, mask: np.ndarray, color: Bgr, opacity: float = 1.0) -> None:
    """Lerp a BGR region towards a flat colour using a uint8 coverage mask.

    Together with :func:`blend_image_mask` this is the single compositing
    primitive behind every HUD element. It is expressed entirely in 8-bit OpenCV
    operations because the equivalent float32 NumPy expression measured about
    seven times slower, and the interface performs dozens of these per frame.
    """
    scaled = _scaled_mask(mask, opacity)
    if scaled is None:
        return
    coverage = cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)
    tinted = cv2.multiply(coverage, np.asarray(color, dtype=np.float64) / 255.0, dtype=cv2.CV_8U)
    cv2.bitwise_not(coverage, dst=coverage)
    region[:] = cv2.add(tinted, cv2.multiply(region, coverage, scale=1.0 / 255.0))


def blend_image_mask(region: np.ndarray, image: np.ndarray, mask: np.ndarray, opacity: float = 1.0) -> None:
    """Lerp a BGR region towards another BGR image using a uint8 coverage mask."""
    scaled = _scaled_mask(mask, opacity)
    if scaled is None:
        return
    coverage = cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)
    tinted = cv2.multiply(image, coverage, scale=1.0 / 255.0)
    cv2.bitwise_not(coverage, dst=coverage)
    region[:] = cv2.add(tinted, cv2.multiply(region, coverage, scale=1.0 / 255.0))


def add_color_mask(region: np.ndarray, mask: np.ndarray, color: Bgr, gain: float = 1.0) -> None:
    """Additively bloom a flat colour into a BGR region (saturating)."""
    scaled = _scaled_mask(mask, gain)
    if scaled is None:
        return
    coverage = cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)
    glow = cv2.multiply(coverage, np.asarray(color, dtype=np.float64) / 255.0, dtype=cv2.CV_8U)
    region[:] = cv2.add(region, glow)


def blit_layer(frame: np.ndarray, layer: np.ndarray, x: int, y: int, opacity: float = 1.0) -> None:
    """Alpha-composite a small BGRA sprite onto the BGR frame at ``(x, y)``."""
    if opacity <= 0.004:
        return
    sprite_h, sprite_w = layer.shape[:2]
    height, width = frame.shape[:2]
    area = clip_rect((x, y, x + sprite_w, y + sprite_h), width, height)
    if area is None:
        return
    x0, y0, x1, y1 = area
    patch = layer[y0 - y : y1 - y, x0 - x : x1 - x]
    blend_image_mask(frame[y0:y1, x0:x1], patch[:, :, :3], patch[:, :, 3], opacity)


def blit_layer_centred(frame: np.ndarray, layer: np.ndarray, centre: tuple[int, int], opacity: float = 1.0) -> None:
    """Alpha-composite a BGRA sprite so that its centre lands on ``centre``."""
    blit_layer(frame, layer, centre[0] - layer.shape[1] // 2, centre[1] - layer.shape[0] // 2, opacity)


def flatten_onto(background: np.ndarray, layer: np.ndarray, rect: Rect | None = None) -> None:
    """Composite a BGRA layer onto an opaque BGR image in place."""
    height, width = background.shape[:2]
    area = clip_rect(rect or (0, 0, width, height), width, height)
    if area is None:
        return
    x0, y0, x1, y1 = area
    source = layer[y0:y1, x0:x1]
    blend_image_mask(background[y0:y1, x0:x1], source[:, :, :3], source[:, :, 3])


class SpriteCache:
    """An LRU cache of small float32 alpha sprites used as brush stamps.

    Rebuilding a soft radial falloff for every stamp would dominate the frame
    budget; quantising the radius to a half pixel makes the cache hit rate close
    to 100 % while remaining visually lossless.
    """

    def __init__(self, capacity: int = 768) -> None:
        self._capacity = capacity
        self._entries: OrderedDict[tuple, np.ndarray] = OrderedDict()

    def get_or_build(self, key: tuple, build: Callable[[], np.ndarray]) -> np.ndarray:
        """Return the cached sprite for ``key``, building it on first request."""
        sprite = self._entries.get(key)
        if sprite is None:
            sprite = build()
            self._entries[key] = sprite
            if len(self._entries) > self._capacity:
                self._entries.popitem(last=False)
        else:
            self._entries.move_to_end(key)
        return sprite

    def disc(self, radius: float, softness: float) -> np.ndarray:
        """Return a soft-edged circular alpha sprite."""
        radius_key = max(0.5, round(radius * 2.0) / 2.0)
        softness_key = round(clamp(softness, 0.0, 1.0) * 20.0) / 20.0
        return self.get_or_build(("disc", radius_key, softness_key), lambda: _build_disc(radius_key, softness_key))

    def ellipse(self, radius_x: float, radius_y: float, angle_deg: float, softness: float) -> np.ndarray:
        """Return a soft-edged rotated elliptical alpha sprite (chisel nibs)."""
        rx_key = max(0.5, round(radius_x * 2.0) / 2.0)
        ry_key = max(0.5, round(radius_y * 2.0) / 2.0)
        angle_key = round(angle_deg / 6.0) * 6.0 % 180.0
        softness_key = round(clamp(softness, 0.0, 1.0) * 10.0) / 10.0
        return self.get_or_build(
            ("ellipse", rx_key, ry_key, angle_key, softness_key),
            lambda: _build_ellipse(rx_key, ry_key, angle_key, softness_key),
        )

    def star(self, radius: float) -> np.ndarray:
        """Return a four-point glint sprite used by sparkle brushes."""
        radius_key = max(1.0, round(radius))
        return self.get_or_build(("star", radius_key), lambda: _build_star(radius_key))

    def disc_mask(self, radius: float, softness: float) -> np.ndarray:
        """Return a soft disc as a uint8 coverage mask, for interface drawing."""
        radius_key = max(0.5, round(radius * 2.0) / 2.0)
        softness_key = round(clamp(softness, 0.0, 1.0) * 20.0) / 20.0
        return self.get_or_build(
            ("disc8", radius_key, softness_key),
            lambda: to_mask(self.disc(radius_key, softness_key)),
        )


def to_mask(coverage: np.ndarray) -> np.ndarray:
    """Convert a float 0..1 coverage array into a uint8 mask."""
    return np.clip(coverage * 255.0, 0.0, 255.0).astype(np.uint8)


def _build_disc(radius: float, softness: float) -> np.ndarray:
    size = math.ceil(radius * 2.0) + 3
    centre = (size - 1) / 2.0
    grid = np.arange(size, dtype=np.float32) - centre
    distance = np.sqrt(grid[None, :] ** 2 + grid[:, None] ** 2)
    falloff = max(radius * softness, 0.8)
    mask = np.clip((radius - distance) / falloff, 0.0, 1.0)
    return (mask * mask * (3.0 - 2.0 * mask)).astype(np.float32)


def _build_ellipse(radius_x: float, radius_y: float, angle_deg: float, softness: float) -> np.ndarray:
    extent = math.ceil(max(radius_x, radius_y) * 2.0) + 5
    canvas = np.zeros((extent, extent), dtype=np.uint8)
    centre = (extent // 2, extent // 2)
    cv2.ellipse(
        canvas,
        centre,
        (max(1, round(radius_x)), max(1, round(radius_y))),
        angle_deg,
        0,
        360,
        255,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )
    blur = int(max(radius_x, radius_y) * softness)
    if blur >= 1:
        kernel = blur * 2 + 1
        canvas = cv2.GaussianBlur(canvas, (kernel, kernel), 0).astype(np.uint8)
    return np.asarray(canvas, dtype=np.float32) / 255.0


def _build_star(radius: float) -> np.ndarray:
    extent = int(radius * 2) + 5
    canvas = np.zeros((extent, extent), dtype=np.float32)
    centre = extent // 2
    length = int(radius)
    for offset in range(-length, length + 1):
        weight = 1.0 - abs(offset) / (length + 1.0)
        canvas[centre, centre + offset] = max(canvas[centre, centre + offset], weight)
        canvas[centre + offset, centre] = max(canvas[centre + offset, centre], weight)
    canvas = np.asarray(cv2.GaussianBlur(canvas, (3, 3), 0), dtype=np.float32)
    core = _build_disc(max(1.0, radius * 0.28), 1.0)
    core_extent = core.shape[0]
    start = centre - core_extent // 2
    region = canvas[start : start + core_extent, start : start + core_extent]
    np.maximum(region, core, out=region)
    return np.clip(canvas / max(canvas.max(), 1e-6), 0.0, 1.0).astype(np.float32)


SPRITES: Final = SpriteCache()


def stamp_sprite(
    layer: np.ndarray,
    sprite: np.ndarray,
    centre_x: float,
    centre_y: float,
    color: Bgr,
    opacity: float,
    mode: BlendMode,
) -> Rect | None:
    """Blend one alpha sprite into a BGRA layer and return the touched rectangle."""
    if opacity <= 0.002:
        return None
    sprite_h, sprite_w = sprite.shape
    left = round(centre_x - sprite_w / 2.0)
    top = round(centre_y - sprite_h / 2.0)
    height, width = layer.shape[:2]
    area = clip_rect((left, top, left + sprite_w, top + sprite_h), width, height)
    if area is None:
        return None
    x0, y0, x1, y1 = area
    patch = sprite[y0 - top : y1 - top, x0 - left : x1 - left]
    target = layer[y0:y1, x0:x1]
    if mode is BlendMode.MAX:
        alpha = (patch * (opacity * 255.0)).astype(np.uint8)
        selected = alpha > target[:, :, 3]
        if not selected.any():
            return None
        for channel, level in enumerate(color):
            target[:, :, channel][selected] = level
        target[:, :, 3][selected] = alpha[selected]
    else:
        src_alpha = patch * opacity
        dst_alpha = target[:, :, 3].astype(np.float32) / 255.0
        out_alpha = src_alpha + dst_alpha * (1.0 - src_alpha)
        safe = np.maximum(out_alpha, 1e-6)
        weight_src = src_alpha / safe
        weight_dst = (dst_alpha * (1.0 - src_alpha)) / safe
        for channel, level in enumerate(color):
            existing = target[:, :, channel].astype(np.float32)
            target[:, :, channel] = np.clip(level * weight_src + existing * weight_dst, 0.0, 255.0).astype(np.uint8)
        target[:, :, 3] = np.clip(out_alpha * 255.0, 0.0, 255.0).astype(np.uint8)
    return area


def stamp_streak(
    layer: np.ndarray,
    start: tuple[float, float],
    end: tuple[float, float],
    color: Bgr,
    thickness: float,
    opacity: float,
    mode: BlendMode,
) -> Rect | None:
    """Draw a short antialiased line into a BGRA layer (spark and glint brushes)."""
    pad = int(thickness) + 3
    left = int(min(start[0], end[0])) - pad
    top = int(min(start[1], end[1])) - pad
    right = int(max(start[0], end[0])) + pad
    bottom = int(max(start[1], end[1])) + pad
    height, width = layer.shape[:2]
    area = clip_rect((left, top, right, bottom), width, height)
    if area is None:
        return None
    x0, y0, x1, y1 = area
    scratch = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.line(
        scratch,
        (round(start[0] - x0), round(start[1] - y0)),
        (round(end[0] - x0), round(end[1] - y0)),
        255,
        max(1, round(thickness)),
        lineType=cv2.LINE_AA,
    )
    return stamp_sprite(
        layer,
        scratch.astype(np.float32) / 255.0,
        (x0 + x1 - 1) / 2.0,
        (y0 + y1 - 1) / 2.0,
        color,
        opacity,
        mode,
    )


def rounded_rect_mask(width: int, height: int, radius: int) -> np.ndarray:
    """Build (and cache) an antialiased rounded-rectangle alpha mask."""
    return SPRITES.get_or_build(("round", width, height, radius), lambda: _build_rounded_rect(width, height, radius))


def _build_rounded_rect(width: int, height: int, radius: int) -> np.ndarray:
    scale = 3  # supersample, then downscale for clean antialiased corners
    canvas = np.zeros((height * scale, width * scale), dtype=np.uint8)
    corner = max(0, radius * scale)
    cv2.rectangle(canvas, (corner, 0), (width * scale - corner, height * scale), 255, -1)
    cv2.rectangle(canvas, (0, corner), (width * scale, height * scale - corner), 255, -1)
    for cx, cy in (
        (corner, corner),
        (width * scale - corner, corner),
        (corner, height * scale - corner),
        (width * scale - corner, height * scale - corner),
    ):
        cv2.circle(canvas, (cx, cy), corner, 255, -1)
    return cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)


def rounded_ring_mask(width: int, height: int, radius: int, thickness: int) -> np.ndarray:
    """Build (and cache) a rounded-rectangle outline as a uint8 coverage mask."""
    return SPRITES.get_or_build(
        ("ring", width, height, radius, thickness),
        lambda: _build_rounded_ring(width, height, radius, thickness),
    )


def _build_rounded_ring(width: int, height: int, radius: int, thickness: int) -> np.ndarray:
    outer = rounded_rect_mask(width, height, radius)
    inset = max(1, thickness)
    if width <= inset * 2 + 2 or height <= inset * 2 + 2:
        return outer
    inner = rounded_rect_mask(width - inset * 2, height - inset * 2, max(0, radius - inset))
    ring = outer.copy()
    core = ring[inset : inset + inner.shape[0], inset : inset + inner.shape[1]]
    core[:] = cv2.subtract(core, inner)
    return ring


def draw_glass_panel(
    frame: np.ndarray,
    rect: Rect,
    radius: int,
    tint: Bgr = THEME.glass,
    opacity: float = 0.62,
    border: Bgr | None = None,
    border_opacity: float = 0.5,
) -> None:
    """Render a frosted-glass panel directly onto the BGR frame.

    The blur is computed on a quarter-resolution copy of the region and scaled
    back up: visually indistinguishable from a full-resolution Gaussian at this
    radius and roughly sixteen times cheaper. Panels smaller than
    :data:`GLASS_BLUR_MIN_EDGE` skip the blur entirely, because at that size the
    frost is invisible but the cost is not — the radial palette alone draws
    fifteen of them.
    """
    height, width = frame.shape[:2]
    area = clip_rect(rect, width, height)
    if area is None:
        return
    x0, y0, x1, y1 = area
    region = frame[y0:y1, x0:x1]
    panel_w, panel_h = x1 - x0, y1 - y0
    corner = min(radius, min(panel_w, panel_h) // 2)
    if min(panel_w, panel_h) >= GLASS_BLUR_MIN_EDGE:
        plate = np.empty_like(region)
        plate[:] = tint
        small = cv2.resize(region, (max(1, panel_w // 4), max(1, panel_h // 4)), interpolation=cv2.INTER_AREA)
        frosted = cv2.resize(cv2.blur(small, (7, 7)), (panel_w, panel_h), interpolation=cv2.INTER_LINEAR)
        cv2.addWeighted(frosted, 1.0 - opacity, plate, opacity, 0.0, dst=plate)
        blend_image_mask(region, plate, rounded_rect_mask(panel_w, panel_h, corner))
    else:
        blit_layer(frame, rounded_tile(panel_w, panel_h, corner, tint, opacity), x0, y0)
    if border is not None:
        draw_rounded_outline(frame, area, radius, border, border_opacity)


def rounded_tile(width: int, height: int, radius: int, color: Bgr, opacity: float) -> np.ndarray:
    """Build (and cache) a flat translucent rounded tile as a BGRA sprite."""
    level = round(clamp(opacity, 0.0, 1.0) * 32.0) / 32.0
    return SPRITES.get_or_build(
        ("tile", width, height, radius, color, level),
        lambda: _build_rounded_tile(width, height, radius, color, level),
    )


def _build_rounded_tile(width: int, height: int, radius: int, color: Bgr, opacity: float) -> np.ndarray:
    tile = new_layer(width, height)
    tile[:, :, :3] = color
    tile[:, :, 3] = cv2.convertScaleAbs(rounded_rect_mask(width, height, radius), alpha=opacity)
    return tile


def draw_rounded_outline(
    frame: np.ndarray,
    rect: Rect,
    radius: int,
    color: Bgr,
    opacity: float = 1.0,
    thickness: int = 1,
) -> None:
    """Stroke a rounded rectangle outline with sub-pixel-smooth antialiasing."""
    height, width = frame.shape[:2]
    area = clip_rect(rect, width, height)
    if area is None or opacity <= 0.004:
        return
    x0, y0, x1, y1 = area
    panel_w, panel_h = x1 - x0, y1 - y0
    corner = min(radius, min(panel_w, panel_h) // 2)
    ring = rounded_ring_mask(panel_w, panel_h, corner, thickness)
    blend_color_mask(frame[y0:y1, x0:x1], ring, color, opacity)


def draw_soft_glow(
    frame: np.ndarray,
    centre: tuple[int, int],
    radius: int,
    color: Bgr,
    intensity: float,
) -> None:
    """Add a radial neon bloom onto the frame using additive blending."""
    if intensity <= 0.004 or radius < 2:
        return
    sprite = SPRITES.disc_mask(float(radius), 1.0)
    extent = sprite.shape[0]
    left = centre[0] - extent // 2
    top = centre[1] - extent // 2
    height, width = frame.shape[:2]
    area = clip_rect((left, top, left + extent, top + extent), width, height)
    if area is None:
        return
    x0, y0, x1, y1 = area
    patch = sprite[y0 - top : y1 - top, x0 - left : x1 - left]
    add_color_mask(frame[y0:y1, x0:x1], patch, color, intensity)


def draw_arc(
    frame: np.ndarray,
    centre: tuple[int, int],
    radius: int,
    start_deg: float,
    end_deg: float,
    color: Bgr,
    thickness: int = 2,
    opacity: float = 1.0,
) -> None:
    """Draw an antialiased circular arc, blended at ``opacity``."""
    if opacity <= 0.004 or end_deg - start_deg <= 0.05:
        return
    pad = thickness + 3
    height, width = frame.shape[:2]
    area = clip_rect(
        (centre[0] - radius - pad, centre[1] - radius - pad, centre[0] + radius + pad, centre[1] + radius + pad),
        width,
        height,
    )
    if area is None:
        return
    x0, y0, x1, y1 = area
    scratch = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.ellipse(
        scratch,
        (centre[0] - x0, centre[1] - y0),
        (radius, radius),
        0.0,
        start_deg,
        end_deg,
        255,
        thickness,
        lineType=cv2.LINE_AA,
    )
    blend_color_mask(frame[y0:y1, x0:x1], scratch, color, opacity)


class MaskBatch:
    """Collects many marks in one region, then composites them in a single pass.

    Blending the hand wireframe as twenty-one separate strokes costs far more in
    per-call overhead than its pixels justify. Painting every mark into one
    scratch coverage mask and compositing once collapses that to a single blend,
    which is where most of the interface's frame budget was going.
    """

    __slots__ = ("_area", "_frame", "_scratch")

    def __init__(self, frame: np.ndarray, rect: Rect) -> None:
        height, width = frame.shape[:2]
        self._frame = frame
        self._area = clip_rect(rect, width, height)
        self._scratch: np.ndarray | None = (
            np.zeros((self._area[3] - self._area[1], self._area[2] - self._area[0]), dtype=np.uint8)
            if self._area is not None
            else None
        )

    @property
    def active(self) -> bool:
        """Whether any part of the region is on screen."""
        return self._scratch is not None

    def _local(self, point: tuple[float, float]) -> tuple[int, int]:
        assert self._area is not None
        return (round(point[0]) - self._area[0], round(point[1]) - self._area[1])

    def line(self, start: tuple[float, float], end: tuple[float, float], intensity: float, thickness: int = 1) -> None:
        """Add an antialiased segment at the given coverage."""
        if self._scratch is None or intensity <= 0.004:
            return
        cv2.line(
            self._scratch,
            self._local(start),
            self._local(end),
            round(clamp(intensity, 0.0, 1.0) * 255),
            thickness,
            cv2.LINE_AA,
        )

    def circle(self, centre: tuple[float, float], radius: float, intensity: float, thickness: int = -1) -> None:
        """Add an antialiased circle, filled by default."""
        if self._scratch is None or intensity <= 0.004 or radius < 0.4:
            return
        cv2.circle(
            self._scratch,
            self._local(centre),
            max(1, round(radius)),
            round(clamp(intensity, 0.0, 1.0) * 255),
            thickness,
            cv2.LINE_AA,
        )

    def flush(self, color: Bgr, additive: bool = False) -> None:
        """Composite everything collected so far and reset the batch."""
        if self._scratch is None or self._area is None:
            return
        region = self._frame[self._area[1] : self._area[3], self._area[0] : self._area[2]]
        if additive:
            add_color_mask(region, self._scratch, color)
        else:
            blend_color_mask(region, self._scratch, color)
        self._scratch[:] = 0


def bounding_rect(points: Iterable[tuple[float, float]], padding: int) -> Rect | None:
    """Return the padded integer bounding box of a set of points."""
    listed = list(points)
    if not listed:
        return None
    xs = [point[0] for point in listed]
    ys = [point[1] for point in listed]
    return (
        int(min(xs)) - padding,
        int(min(ys)) - padding,
        int(max(xs)) + padding,
        int(max(ys)) + padding,
    )


def draw_ring(
    frame: np.ndarray,
    centre: tuple[int, int],
    radius: int,
    color: Bgr,
    thickness: int = 1,
    opacity: float = 1.0,
) -> None:
    """Draw a full antialiased circle outline."""
    draw_arc(frame, centre, radius, 0.0, 360.0, color, thickness, opacity)


def draw_filled_circle(
    frame: np.ndarray,
    centre: tuple[int, int],
    radius: float,
    color: Bgr,
    opacity: float = 1.0,
    softness: float = 0.06,
) -> None:
    """Draw a filled antialiased disc onto the BGR frame."""
    if radius < 0.4 or opacity <= 0.004:
        return
    sprite = SPRITES.disc_mask(radius, softness)
    extent = sprite.shape[0]
    left = round(centre[0] - extent / 2)
    top = round(centre[1] - extent / 2)
    height, width = frame.shape[:2]
    area = clip_rect((left, top, left + extent, top + extent), width, height)
    if area is None:
        return
    x0, y0, x1, y1 = area
    patch = sprite[y0 - top : y1 - top, x0 - left : x1 - left]
    blend_color_mask(frame[y0:y1, x0:x1], patch, color, opacity)


def draw_line_aa(
    frame: np.ndarray,
    start: tuple[float, float],
    end: tuple[float, float],
    color: Bgr,
    thickness: int = 1,
    opacity: float = 1.0,
) -> None:
    """Draw an antialiased straight line at a given opacity.

    Translucent lines are blended inside the segment's own bounding box. Using a
    whole-frame overlay copy instead costs about a hundred times more, and the HUD
    draws dozens of these — the hand wireframe alone is twenty-one — every frame.
    """
    if opacity <= 0.004:
        return
    first = (round(start[0]), round(start[1]))
    second = (round(end[0]), round(end[1]))
    if opacity >= 0.996:
        cv2.line(frame, first, second, color, thickness, lineType=cv2.LINE_AA)
        return
    pad = thickness + 2
    height, width = frame.shape[:2]
    area = clip_rect(
        (
            min(first[0], second[0]) - pad,
            min(first[1], second[1]) - pad,
            max(first[0], second[0]) + pad,
            max(first[1], second[1]) + pad,
        ),
        width,
        height,
    )
    if area is None:
        return
    x0, y0, x1, y1 = area
    scratch = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.line(scratch, (first[0] - x0, first[1] - y0), (second[0] - x0, second[1] - y0), 255, thickness, cv2.LINE_AA)
    blend_color_mask(frame[y0:y1, x0:x1], scratch, color, opacity)


# =============================================================================
# SECTION 5 — Typography
# =============================================================================


class FontRole(Enum):
    """Semantic font roles, so call sites never name a font file."""

    DISPLAY = "display"
    MONO = "mono"


#: Candidate font files, most-preferred first. Bahnschrift is a wide, technical
#: geometric sans that carries the instrument-panel tone; Consolas supplies the
#: telemetry voice. Neither is a generic UI default.
_FONT_CANDIDATES: Final[dict[FontRole, tuple[str, ...]]] = {
    FontRole.DISPLAY: ("bahnschrift.ttf", "Candaral.ttf", "corbell.ttf", "trebuc.ttf", "DejaVuSans.ttf"),
    FontRole.MONO: ("consola.ttf", "DejaVuSansMono.ttf", "cour.ttf"),
}


def _font_search_paths() -> Iterator[Path]:
    """Yield the directories that may hold system fonts on this platform."""
    windir = os.environ.get("WINDIR")
    if windir:
        yield Path(windir) / "Fonts"
        local = os.environ.get("LOCALAPPDATA")
        if local:
            yield Path(local) / "Microsoft" / "Windows" / "Fonts"
    yield Path("/usr/share/fonts/truetype/dejavu")
    yield Path("/Library/Fonts")


class TextRenderer:
    """Renders tracked, antialiased HUD text via cached alpha sprites.

    Drawing text with PIL costs far more than blitting a cached mask, and HUD
    labels are highly repetitive, so each unique ``(text, role, size, tracking)``
    is rasterised once into a float mask and then reused for free.
    """

    def __init__(self, capacity: int = 512) -> None:
        self._faces: dict[tuple[FontRole, int], ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}
        self._masks: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._capacity = capacity
        self._resolved: dict[FontRole, Path | None] = {}

    def _font_file(self, role: FontRole) -> Path | None:
        if role in self._resolved:
            return self._resolved[role]
        found: Path | None = None
        for directory in _font_search_paths():
            if not directory.is_dir():
                continue
            for filename in _FONT_CANDIDATES[role]:
                candidate = directory / filename
                if candidate.is_file():
                    found = candidate
                    break
            if found is not None:
                break
        if found is None:
            LOGGER.warning("No %s font found; falling back to the PIL bitmap face", role.value)
        self._resolved[role] = found
        return found

    def _face(self, role: FontRole, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        key = (role, size)
        face = self._faces.get(key)
        if face is None:
            path = self._font_file(role)
            face = ImageFont.load_default() if path is None else ImageFont.truetype(str(path), size)
            self._faces[key] = face
        return face

    def _mask(self, text: str, role: FontRole, size: int, tracking: int) -> np.ndarray:
        key = (text, role, size, tracking)
        mask = self._masks.get(key)
        if mask is not None:
            self._masks.move_to_end(key)
            return mask
        face = self._face(role, size)
        advances = [int(face.getlength(character)) + tracking for character in text]
        total = max(1, sum(advances))
        ascent, descent = face.getmetrics() if hasattr(face, "getmetrics") else (size, size // 4)
        canvas = Image.new("L", (total + size, ascent + descent + 4), 0)
        painter = ImageDraw.Draw(canvas)
        cursor = 0
        for character, advance in zip(text, advances, strict=True):
            painter.text((cursor, 0), character, font=face, fill=255)
            cursor += advance
        mask = np.asarray(canvas, dtype=np.uint8)
        self._masks[key] = mask
        if len(self._masks) > self._capacity:
            self._masks.popitem(last=False)
        return mask

    def measure(self, text: str, role: FontRole, size: int, tracking: int = 0) -> tuple[int, int]:
        """Return the pixel ``(width, height)`` of a rendered label."""
        mask = self._mask(text, role, size, tracking)
        return (mask.shape[1], mask.shape[0])

    def draw(
        self,
        frame: np.ndarray,
        text: str,
        position: tuple[int, int],
        *,
        role: FontRole = FontRole.DISPLAY,
        size: int = 16,
        color: Bgr = THEME.text_hi,
        opacity: float = 1.0,
        tracking: int = 0,
        anchor: str = "lt",
        glow: float = 0.0,
    ) -> Rect | None:
        """Blit a text label onto a BGR frame.

        ``anchor`` follows the PIL convention: the first character selects the
        horizontal origin (``l``/``m``/``r``) and the second the vertical
        (``t``/``m``/``b``).
        """
        if not text or opacity <= 0.004:
            return None
        mask = self._mask(text, role, size, tracking)
        text_h, text_w = mask.shape
        x = position[0] - {"l": 0, "m": text_w // 2, "r": text_w}[anchor[0]]
        y = position[1] - {"t": 0, "m": text_h // 2, "b": text_h}[anchor[1]]
        height, width = frame.shape[:2]
        area = clip_rect((x, y, x + text_w, y + text_h), width, height)
        if area is None:
            return None
        x0, y0, x1, y1 = area
        patch = mask[y0 - y : y1 - y, x0 - x : x1 - x]
        region = frame[y0:y1, x0:x1]
        if glow > 0.004:
            add_color_mask(region, cv2.GaussianBlur(patch, (0, 0), sigmaX=2.4), color, glow * opacity)
        blend_color_mask(region, patch, color, opacity)
        return area


TEXT: Final = TextRenderer()


# =============================================================================
# SECTION 6 — Painting domain model
# =============================================================================


class Tool(Enum):
    """The active painting tool, derived from the hand pose."""

    IDLE = "IDLE"
    DRAW = "DRAW"
    SELECT = "SELECT"
    ERASE = "ERASE"


class BrushId(IntEnum):
    """Every brush the painting engine can render."""

    BRUSH = 0
    PENCIL = 1
    NEON = 2
    MARKER = 3
    WATERCOLOR = 4
    AIRBRUSH = 5
    CALLIGRAPHY = 6
    HIGHLIGHTER = 7
    PARTICLE = 8
    GLOW = 9
    FIRE = 10
    SPARK = 11
    GALAXY = 12
    SMOKE = 13
    MAGIC = 14


class ColorMode(Enum):
    """How a palette slot produces a colour along a stroke."""

    SOLID = auto()
    GRADIENT = auto()
    RAINBOW = auto()


@dataclass(frozen=True, slots=True)
class ColorSlot:
    """One entry of the colour palette."""

    name: str
    mode: ColorMode
    color: Bgr = (255, 255, 255)
    #: Gradient stops, only meaningful when ``mode`` is :attr:`ColorMode.GRADIENT`.
    stops: tuple[Bgr, ...] = ()

    def sample(self, distance: float) -> Bgr:
        """Return the colour at ``distance`` pixels along a stroke."""
        if self.mode is ColorMode.SOLID:
            return self.color
        if self.mode is ColorMode.RAINBOW:
            return hue_to_bgr(distance / 420.0, 0.92, 1.0)
        span = 240.0
        position = (distance / span) % len(self.stops)
        index = int(position)
        return mix_bgr(self.stops[index], self.stops[(index + 1) % len(self.stops)], position - index)

    def preview_color(self) -> Bgr:
        """A single representative colour for palette chips and previews."""
        return self.color if self.mode is ColorMode.SOLID else self.sample(0.0)


PALETTE: Final[tuple[ColorSlot, ...]] = (
    ColorSlot("RED", ColorMode.SOLID, hex_to_bgr("#FF3B4E")),
    ColorSlot("ORANGE", ColorMode.SOLID, hex_to_bgr("#FF8A2B")),
    ColorSlot("YELLOW", ColorMode.SOLID, hex_to_bgr("#FFE04A")),
    ColorSlot("GREEN", ColorMode.SOLID, hex_to_bgr("#23E07A")),
    ColorSlot("CYAN", ColorMode.SOLID, hex_to_bgr("#22E6FF")),
    ColorSlot("BLUE", ColorMode.SOLID, hex_to_bgr("#2E7BFF")),
    ColorSlot("PURPLE", ColorMode.SOLID, hex_to_bgr("#9B5CFF")),
    ColorSlot("MAGENTA", ColorMode.SOLID, hex_to_bgr("#FF2ECC")),
    ColorSlot("PINK", ColorMode.SOLID, hex_to_bgr("#FF7FB8")),
    ColorSlot("BROWN", ColorMode.SOLID, hex_to_bgr("#A4633A")),
    ColorSlot("WHITE", ColorMode.SOLID, hex_to_bgr("#FFFFFF")),
    ColorSlot("BLACK", ColorMode.SOLID, hex_to_bgr("#0B0D12")),
    ColorSlot(
        "GRADIENT",
        ColorMode.GRADIENT,
        hex_to_bgr("#22E6FF"),
        stops=(hex_to_bgr("#22E6FF"), hex_to_bgr("#FF2E88"), hex_to_bgr("#FFC246")),
    ),
    ColorSlot("RAINBOW", ColorMode.RAINBOW, hex_to_bgr("#FF3B4E")),
)

#: Index of the instrument-cyan slot, used for inactive interface swatches.
NEUTRAL_COLOR_INDEX: Final[int] = next(index for index, slot in enumerate(PALETTE) if slot.name == "CYAN")

#: The six selectable stroke thicknesses, in pixels of full stroke width.
THICKNESSES: Final[tuple[int, ...]] = (5, 10, 15, 20, 30, 40)


@dataclass(frozen=True, slots=True)
class BrushSpec:
    """Static description of a brush: how it stamps and how densely."""

    brush_id: BrushId
    label: str
    #: Stamp spacing as a fraction of the brush radius.
    spacing_ratio: float
    #: Scales the user-selected thickness; expressive brushes need more room.
    size_scale: float = 1.0
    #: Short technical caption shown under the brush preview.
    caption: str = ""


@dataclass(frozen=True, slots=True)
class PathSample:
    """One stamp position along a smoothed stroke path."""

    x: float
    y: float
    index: int
    distance: float
    angle: float
    velocity: float


@dataclass(slots=True)
class Stroke:
    """A vector record of one continuous mark.

    Strokes are the document's source of truth. The raster canvas is a cache that
    can always be rebuilt from this list, which is what makes unlimited undo,
    shape snapping and true high-resolution export possible without ever storing
    a single full-frame bitmap snapshot.
    """

    brush: BrushId
    color_index: int
    thickness: int
    opacity: float
    seed: int
    points: list[tuple[float, float, float]] = field(default_factory=list)
    is_eraser: bool = False
    #: Set when shape recognition replaced the freehand path with ideal geometry.
    shape_points: list[tuple[float, float]] | None = None
    shape_name: str | None = None

    def path_points(self) -> list[tuple[float, float, float]]:
        """Return the points that should actually be rendered."""
        if self.shape_points is None:
            return self.points
        return [(x, y, 0.0) for x, y in self.shape_points]


# =============================================================================
# SECTION 7 — Path sampling
# =============================================================================


class PathSampler:
    """Streams raw fingertip positions into uniformly spaced spline samples.

    The sampler is append-only and fully deterministic: feeding a stroke's raw
    points through a fresh sampler reproduces exactly the same sample sequence,
    including sample indices. Brushes seed their randomness from those indices,
    so a rebuilt canvas is pixel-identical to the live one.
    """

    __slots__ = ("_carry", "_control", "_distance", "_index", "_next_segment", "_spacing", "_started")

    def __init__(self, spacing: float) -> None:
        self._spacing = max(0.6, spacing)
        self._control: list[tuple[float, float, float]] = []
        self._next_segment = 0
        self._carry = 0.0
        self._distance = 0.0
        self._index = 0
        self._started = False

    def push(self, x: float, y: float, velocity: float) -> list[PathSample]:
        """Feed one raw point and return every newly finalised sample."""
        self._control.append((x, y, velocity))
        produced: list[PathSample] = []
        if not self._started:
            self._started = True
            produced.append(PathSample(x, y, self._index, 0.0, 0.0, velocity))
            self._index += 1
        while self._next_segment + 3 < len(self._control):
            produced.extend(self._emit_segment(self._next_segment))
            self._next_segment += 1
        return produced

    def finish(self) -> list[PathSample]:
        """Flush the trailing segment by duplicating the final control point."""
        if len(self._control) < 2:
            return []
        self._control.append(self._control[-1])
        produced: list[PathSample] = []
        while self._next_segment + 3 < len(self._control):
            produced.extend(self._emit_segment(self._next_segment))
            self._next_segment += 1
        return produced

    def _emit_segment(self, start_index: int) -> list[PathSample]:
        control = self._control
        first = control[start_index] if start_index >= 0 else control[0]
        p0, p1, p2, p3 = first, control[start_index + 1], control[start_index + 2], control[start_index + 3]
        chord = math.dist(p1[:2], p2[:2])
        steps = int(clamp(chord / self._spacing * 2.0, 3.0, 96.0))
        produced: list[PathSample] = []
        previous = (p1[0], p1[1])
        for step in range(1, steps + 1):
            t = step / steps
            point = catmull_rom(p0[:2], p1[:2], p2[:2], p3[:2], t)
            segment = math.dist(previous, point)
            if segment <= 1e-9:
                continue
            angle = math.atan2(point[1] - previous[1], point[0] - previous[0])
            velocity = lerp(p1[2], p2[2], t)
            travelled = 0.0
            remaining = segment
            while self._carry + remaining >= self._spacing:
                needed = self._spacing - self._carry
                travelled += needed
                remaining -= needed
                ratio = travelled / segment
                self._distance += self._spacing
                produced.append(
                    PathSample(
                        lerp(previous[0], point[0], ratio),
                        lerp(previous[1], point[1], ratio),
                        self._index,
                        self._distance,
                        angle,
                        velocity,
                    )
                )
                self._index += 1
                self._carry = 0.0
            self._carry += remaining
            previous = point
        return produced


# =============================================================================
# SECTION 8 — Brush engine
# =============================================================================

#: Signature shared by every brush renderer.
Renderer = Callable[[np.ndarray, PathSample, float, Bgr, float, random.Random], Rect | None]

_FIRE_RAMP: Final[tuple[Bgr, ...]] = (
    hex_to_bgr("#FFF6C2"),
    hex_to_bgr("#FFC246"),
    hex_to_bgr("#FF7A18"),
    hex_to_bgr("#E02B12"),
    hex_to_bgr("#5C0A06"),
)
_SMOKE_TINT: Final[Bgr] = hex_to_bgr("#B9C6CE")
_STAR_TINTS: Final[tuple[Bgr, ...]] = (
    hex_to_bgr("#FFFFFF"),
    hex_to_bgr("#BFE9FF"),
    hex_to_bgr("#FFD9F2"),
    hex_to_bgr("#D9C6FF"),
)
#: Chisel orientation of the calligraphy nib, in degrees.
CALLIGRAPHY_NIB_DEG: Final[float] = 42.0


def _ramp(colors: Sequence[Bgr], position: float) -> Bgr:
    """Sample a discrete colour ramp with linear interpolation."""
    scaled = clamp(position, 0.0, 1.0) * (len(colors) - 1)
    index = min(int(scaled), len(colors) - 2)
    return mix_bgr(colors[index], colors[index + 1], scaled - index)


def _taper(radius: float, velocity: float, amount: float) -> float:
    """Thin a stroke as the hand accelerates, mimicking real brush pressure."""
    return max(0.7, radius / (1.0 + velocity * amount))


def _render_brush(
    buffer: np.ndarray, sample: PathSample, radius: float, color: Bgr, opacity: float, rng: random.Random
) -> Rect | None:
    return stamp_sprite(
        buffer,
        SPRITES.disc(_taper(radius, sample.velocity, 0.00042), 0.30),
        sample.x,
        sample.y,
        color,
        opacity,
        BlendMode.MAX,
    )


def _render_pencil(
    buffer: np.ndarray, sample: PathSample, radius: float, color: Bgr, opacity: float, rng: random.Random
) -> Rect | None:
    if rng.random() < 0.16:  # graphite grain: skipped stamps read as paper tooth
        return None
    jitter = radius * 0.22
    return stamp_sprite(
        buffer,
        SPRITES.disc(max(0.8, _taper(radius * 0.5, sample.velocity, 0.0009)), 0.88),
        sample.x + rng.uniform(-jitter, jitter),
        sample.y + rng.uniform(-jitter, jitter),
        color,
        opacity * rng.uniform(0.42, 0.85),
        BlendMode.MAX,
    )


def _render_neon(
    buffer: np.ndarray, sample: PathSample, radius: float, color: Bgr, opacity: float, rng: random.Random
) -> Rect | None:
    dirty = stamp_sprite(
        buffer, SPRITES.disc(radius * 2.9, 1.0), sample.x, sample.y, color, opacity * 0.085, BlendMode.OVER
    )
    dirty = union_rect(
        dirty,
        stamp_sprite(
            buffer, SPRITES.disc(radius * 1.5, 0.85), sample.x, sample.y, color, opacity * 0.22, BlendMode.OVER
        ),
    )
    core = mix_bgr(color, (255, 255, 255), 0.7)
    return union_rect(
        dirty,
        stamp_sprite(
            buffer, SPRITES.disc(max(1.0, radius * 0.5), 0.35), sample.x, sample.y, core, opacity, BlendMode.MAX
        ),
    )


def _render_marker(
    buffer: np.ndarray, sample: PathSample, radius: float, color: Bgr, opacity: float, rng: random.Random
) -> Rect | None:
    return stamp_sprite(buffer, SPRITES.disc(radius, 0.05), sample.x, sample.y, color, opacity, BlendMode.MAX)


def _render_watercolor(
    buffer: np.ndarray, sample: PathSample, radius: float, color: Bgr, opacity: float, rng: random.Random
) -> Rect | None:
    dirty = stamp_sprite(
        buffer, SPRITES.disc(radius * 0.9, 0.95), sample.x, sample.y, color, opacity * 0.10, BlendMode.OVER
    )
    spread = radius * 0.62
    for _ in range(3):
        bleed = mix_bgr(color, (255, 255, 255), rng.uniform(0.0, 0.22))
        dirty = union_rect(
            dirty,
            stamp_sprite(
                buffer,
                SPRITES.disc(radius * rng.uniform(0.7, 1.8), 1.0),
                sample.x + rng.gauss(0.0, spread),
                sample.y + rng.gauss(0.0, spread),
                bleed,
                opacity * rng.uniform(0.03, 0.075),
                BlendMode.OVER,
            ),
        )
    return dirty


def _render_airbrush(
    buffer: np.ndarray, sample: PathSample, radius: float, color: Bgr, opacity: float, rng: random.Random
) -> Rect | None:
    dirty: Rect | None = None
    spread = radius * 0.72
    for _ in range(16):
        dirty = union_rect(
            dirty,
            stamp_sprite(
                buffer,
                SPRITES.disc(rng.uniform(0.9, 1.7), 1.0),
                sample.x + rng.gauss(0.0, spread),
                sample.y + rng.gauss(0.0, spread),
                color,
                opacity * 0.055,
                BlendMode.OVER,
            ),
        )
    return dirty


def _render_calligraphy(
    buffer: np.ndarray, sample: PathSample, radius: float, color: Bgr, opacity: float, rng: random.Random
) -> Rect | None:
    # A fixed-angle chisel nib: width falls out of the angle between the nib and
    # the direction of travel, exactly like a real broad-edge pen.
    return stamp_sprite(
        buffer,
        SPRITES.ellipse(radius * 1.45, max(0.8, radius * 0.26), CALLIGRAPHY_NIB_DEG, 0.12),
        sample.x,
        sample.y,
        color,
        opacity,
        BlendMode.MAX,
    )


def _render_highlighter(
    buffer: np.ndarray, sample: PathSample, radius: float, color: Bgr, opacity: float, rng: random.Random
) -> Rect | None:
    return stamp_sprite(
        buffer,
        SPRITES.ellipse(radius * 0.55, radius * 1.55, 0.0, 0.06),
        sample.x,
        sample.y,
        mix_bgr(color, (255, 255, 255), 0.12),
        opacity * 0.42,
        BlendMode.MAX,
    )


def _render_particle(
    buffer: np.ndarray, sample: PathSample, radius: float, color: Bgr, opacity: float, rng: random.Random
) -> Rect | None:
    dirty = stamp_sprite(
        buffer, SPRITES.disc(max(1.0, radius * 0.35), 0.5), sample.x, sample.y, color, opacity * 0.85, BlendMode.OVER
    )
    normal = sample.angle + math.pi / 2.0
    for _ in range(4):
        offset = rng.gauss(0.0, radius * 1.25)
        drift = rng.uniform(-radius * 0.5, radius * 0.5)
        dirty = union_rect(
            dirty,
            stamp_sprite(
                buffer,
                SPRITES.disc(rng.uniform(0.9, max(1.2, radius * 0.3)), 0.65),
                sample.x + math.cos(normal) * offset + math.cos(sample.angle) * drift,
                sample.y + math.sin(normal) * offset + math.sin(sample.angle) * drift,
                mix_bgr(color, (255, 255, 255), rng.uniform(0.0, 0.5)),
                opacity * rng.uniform(0.18, 0.7),
                BlendMode.OVER,
            ),
        )
    return dirty


def _render_glow(
    buffer: np.ndarray, sample: PathSample, radius: float, color: Bgr, opacity: float, rng: random.Random
) -> Rect | None:
    dirty = stamp_sprite(
        buffer, SPRITES.disc(radius * 3.4, 1.0), sample.x, sample.y, color, opacity * 0.07, BlendMode.OVER
    )
    dirty = union_rect(
        dirty,
        stamp_sprite(
            buffer, SPRITES.disc(radius * 1.7, 1.0), sample.x, sample.y, color, opacity * 0.16, BlendMode.OVER
        ),
    )
    return union_rect(
        dirty,
        stamp_sprite(
            buffer,
            SPRITES.disc(max(1.0, radius * 0.7), 0.55),
            sample.x,
            sample.y,
            mix_bgr(color, (255, 255, 255), 0.4),
            opacity * 0.95,
            BlendMode.MAX,
        ),
    )


def _render_fire(
    buffer: np.ndarray, sample: PathSample, radius: float, color: Bgr, opacity: float, rng: random.Random
) -> Rect | None:
    dirty: Rect | None = None
    for _ in range(6):
        rise = rng.random()
        ember = mix_bgr(_ramp(_FIRE_RAMP, rise), color, 0.22)
        dirty = union_rect(
            dirty,
            stamp_sprite(
                buffer,
                SPRITES.disc(max(1.0, radius * (1.15 - rise * 0.8)), 1.0),
                sample.x + rng.gauss(0.0, radius * 0.55) * (0.4 + rise),
                sample.y - rise * radius * 3.1,
                ember,
                opacity * (0.30 - rise * 0.22),
                BlendMode.OVER,
            ),
        )
    return union_rect(
        dirty,
        stamp_sprite(
            buffer,
            SPRITES.disc(max(1.0, radius * 0.55), 0.7),
            sample.x,
            sample.y,
            _FIRE_RAMP[0],
            opacity * 0.75,
            BlendMode.OVER,
        ),
    )


def _render_spark(
    buffer: np.ndarray, sample: PathSample, radius: float, color: Bgr, opacity: float, rng: random.Random
) -> Rect | None:
    dirty = stamp_sprite(
        buffer,
        SPRITES.disc(max(1.0, radius * 0.4), 0.6),
        sample.x,
        sample.y,
        mix_bgr(color, (255, 255, 255), 0.55),
        opacity,
        BlendMode.OVER,
    )
    for _ in range(3):
        angle = rng.uniform(0.0, math.tau)
        length = radius * rng.uniform(0.9, 2.6)
        dirty = union_rect(
            dirty,
            stamp_streak(
                buffer,
                (sample.x, sample.y),
                (sample.x + math.cos(angle) * length, sample.y + math.sin(angle) * length),
                mix_bgr(color, (255, 255, 255), rng.uniform(0.2, 0.8)),
                1.0,
                opacity * rng.uniform(0.25, 0.7),
                BlendMode.OVER,
            ),
        )
    return dirty


def _render_galaxy(
    buffer: np.ndarray, sample: PathSample, radius: float, color: Bgr, opacity: float, rng: random.Random
) -> Rect | None:
    dirty: Rect | None = None
    if rng.random() < 0.35:  # sparse nebula clouds keep the density believable
        nebula = mix_bgr(color, hue_to_bgr(rng.uniform(0.62, 0.88), 0.85, 1.0), rng.uniform(0.25, 0.75))
        dirty = stamp_sprite(
            buffer,
            SPRITES.disc(radius * rng.uniform(1.6, 3.0), 1.0),
            sample.x + rng.gauss(0.0, radius * 0.8),
            sample.y + rng.gauss(0.0, radius * 0.8),
            nebula,
            opacity * rng.uniform(0.03, 0.07),
            BlendMode.OVER,
        )
    for _ in range(3):
        dirty = union_rect(
            dirty,
            stamp_sprite(
                buffer,
                SPRITES.disc(rng.uniform(0.7, 1.5), 0.75),
                sample.x + rng.gauss(0.0, radius * 1.5),
                sample.y + rng.gauss(0.0, radius * 1.5),
                rng.choice(_STAR_TINTS),
                opacity * rng.uniform(0.3, 1.0),
                BlendMode.OVER,
            ),
        )
    if rng.random() < 0.10:
        dirty = union_rect(
            dirty,
            stamp_sprite(
                buffer,
                SPRITES.star(radius * 1.6),
                sample.x + rng.gauss(0.0, radius),
                sample.y + rng.gauss(0.0, radius),
                _STAR_TINTS[0],
                opacity * 0.8,
                BlendMode.OVER,
            ),
        )
    return dirty


def _render_smoke(
    buffer: np.ndarray, sample: PathSample, radius: float, color: Bgr, opacity: float, rng: random.Random
) -> Rect | None:
    dirty: Rect | None = None
    tint = mix_bgr(color, _SMOKE_TINT, 0.62)
    turbulence = math.sin(sample.index * 0.21) * radius * 1.1
    for layer in range(2):
        dirty = union_rect(
            dirty,
            stamp_sprite(
                buffer,
                SPRITES.disc(radius * (1.9 + layer * 0.9), 1.0),
                sample.x + turbulence * (0.6 + layer * 0.5) + rng.gauss(0.0, radius * 0.4),
                sample.y - math.cos(sample.index * 0.17) * radius * (0.8 + layer),
                tint,
                opacity * (0.045 - layer * 0.016),
                BlendMode.OVER,
            ),
        )
    return dirty


def _render_magic(
    buffer: np.ndarray, sample: PathSample, radius: float, color: Bgr, opacity: float, rng: random.Random
) -> Rect | None:
    shifted = mix_bgr(color, hue_to_bgr(sample.distance / 260.0, 0.85, 1.0), 0.55)
    dirty = stamp_sprite(
        buffer, SPRITES.disc(radius * 2.4, 1.0), sample.x, sample.y, shifted, opacity * 0.10, BlendMode.OVER
    )
    dirty = union_rect(
        dirty,
        stamp_sprite(
            buffer,
            SPRITES.disc(max(1.0, radius * 0.55), 0.4),
            sample.x,
            sample.y,
            mix_bgr(shifted, (255, 255, 255), 0.55),
            opacity,
            BlendMode.MAX,
        ),
    )
    if rng.random() < 0.22:
        dirty = union_rect(
            dirty,
            stamp_sprite(
                buffer,
                SPRITES.star(radius * rng.uniform(1.2, 2.6)),
                sample.x + rng.gauss(0.0, radius * 1.6),
                sample.y + rng.gauss(0.0, radius * 1.6),
                mix_bgr(shifted, (255, 255, 255), 0.75),
                opacity * rng.uniform(0.4, 0.95),
                BlendMode.OVER,
            ),
        )
    return dirty


BRUSH_SPECS: Final[dict[BrushId, BrushSpec]] = {
    BrushId.BRUSH: BrushSpec(BrushId.BRUSH, "BRUSH", 0.26, 1.0, "soft round · pressure taper"),
    BrushId.PENCIL: BrushSpec(BrushId.PENCIL, "PENCIL", 0.20, 0.85, "graphite grain · light tooth"),
    BrushId.NEON: BrushSpec(BrushId.NEON, "NEON", 0.22, 1.0, "triple pass · hot core"),
    BrushId.MARKER: BrushSpec(BrushId.MARKER, "MARKER", 0.18, 1.0, "hard edge · flat ink"),
    BrushId.WATERCOLOR: BrushSpec(BrushId.WATERCOLOR, "WATERCOLOUR", 0.50, 1.15, "pigment bleed · wet edge"),
    BrushId.AIRBRUSH: BrushSpec(BrushId.AIRBRUSH, "AIRBRUSH", 0.45, 1.2, "gaussian spray · 16 jets"),
    BrushId.CALLIGRAPHY: BrushSpec(BrushId.CALLIGRAPHY, "CALLIGRAPHY", 0.14, 1.05, "broad edge nib · 42°"),
    BrushId.HIGHLIGHTER: BrushSpec(BrushId.HIGHLIGHTER, "HIGHLIGHTER", 0.18, 1.3, "chisel · flat translucency"),
    BrushId.PARTICLE: BrushSpec(BrushId.PARTICLE, "PARTICLE", 0.38, 1.0, "normal scatter · 4 emitters"),
    BrushId.GLOW: BrushSpec(BrushId.GLOW, "GLOW", 0.30, 1.0, "wide bloom · additive halo"),
    BrushId.FIRE: BrushSpec(BrushId.FIRE, "FIRE", 0.32, 1.0, "ember ramp · thermal rise"),
    BrushId.SPARK: BrushSpec(BrushId.SPARK, "SPARK", 0.42, 1.0, "radial streaks · white hot"),
    BrushId.GALAXY: BrushSpec(BrushId.GALAXY, "GALAXY", 0.46, 1.1, "nebula + starfield"),
    BrushId.SMOKE: BrushSpec(BrushId.SMOKE, "SMOKE", 0.55, 1.25, "turbulent volume · drift"),
    BrushId.MAGIC: BrushSpec(BrushId.MAGIC, "MAGIC TRAIL", 0.28, 1.0, "hue shift · glint sparkle"),
}

BRUSH_RENDERERS: Final[dict[BrushId, Renderer]] = {
    BrushId.BRUSH: _render_brush,
    BrushId.PENCIL: _render_pencil,
    BrushId.NEON: _render_neon,
    BrushId.MARKER: _render_marker,
    BrushId.WATERCOLOR: _render_watercolor,
    BrushId.AIRBRUSH: _render_airbrush,
    BrushId.CALLIGRAPHY: _render_calligraphy,
    BrushId.HIGHLIGHTER: _render_highlighter,
    BrushId.PARTICLE: _render_particle,
    BrushId.GLOW: _render_glow,
    BrushId.FIRE: _render_fire,
    BrushId.SPARK: _render_spark,
    BrushId.GALAXY: _render_galaxy,
    BrushId.SMOKE: _render_smoke,
    BrushId.MAGIC: _render_magic,
}


class StrokeRasteriser:
    """Turns a :class:`Stroke` into pixels, incrementally or in one pass.

    Instances are cheap and stateful: the live stroke owns one, while document
    rebuilds create a throw-away instance per stroke. Because every random draw
    is seeded from ``(stroke seed, sample index)``, both paths produce identical
    output.
    """

    __slots__ = ("_radius", "_renderer", "_sampler", "_scale", "_slot", "_stroke")

    def __init__(self, stroke: Stroke, scale: float = 1.0) -> None:
        spec = BRUSH_SPECS[stroke.brush]
        self._stroke = stroke
        self._scale = scale
        self._radius = max(0.9, stroke.thickness * spec.size_scale * scale / 2.0)
        self._sampler = PathSampler(max(0.7, self._radius * spec.spacing_ratio))
        self._renderer = BRUSH_RENDERERS[stroke.brush]
        self._slot = PALETTE[stroke.color_index % len(PALETTE)]

    @property
    def radius(self) -> float:
        """Effective stamp radius in pixels."""
        return self._radius

    def _draw(self, buffer: np.ndarray, samples: Iterable[PathSample]) -> Rect | None:
        dirty: Rect | None = None
        stroke = self._stroke
        if stroke.is_eraser:
            sprite = SPRITES.disc(self._radius, 0.35)
            for sample in samples:
                dirty = union_rect(
                    dirty, stamp_sprite(buffer, sprite, sample.x, sample.y, (255, 255, 255), 1.0, BlendMode.MAX)
                )
            return dirty
        opacity = stroke.opacity
        for sample in samples:
            rng = random.Random((stroke.seed * 1_000_003) ^ (sample.index * 2_654_435_761))
            color = self._slot.sample(sample.distance / max(self._scale, 1e-3))
            dirty = union_rect(dirty, self._renderer(buffer, sample, self._radius, color, opacity, rng))
        return dirty

    def feed(self, buffer: np.ndarray, x: float, y: float, velocity: float) -> Rect | None:
        """Render the samples produced by one new raw point."""
        return self._draw(buffer, self._sampler.push(x, y, velocity))

    def flush(self, buffer: np.ndarray) -> Rect | None:
        """Render the trailing samples once the stroke has ended."""
        return self._draw(buffer, self._sampler.finish())

    @classmethod
    def render_complete(cls, stroke: Stroke, buffer: np.ndarray, scale: float = 1.0) -> Rect | None:
        """Rasterise a finished stroke into ``buffer`` from scratch."""
        rasteriser = cls(stroke, scale)
        dirty: Rect | None = None
        for x, y, velocity in stroke.path_points():
            dirty = union_rect(dirty, rasteriser.feed(buffer, x * scale, y * scale, velocity))
        return union_rect(dirty, rasteriser.flush(buffer))


def render_brush_swatch(
    brush: BrushId,
    color_index: int,
    thickness: int,
    opacity: float,
    size: tuple[int, int],
    phase: float = 0.0,
) -> np.ndarray:
    """Rasterise a signature S-curve with a brush — used for live previews.

    The preview is produced by the real painting engine rather than a hand-drawn
    icon, so what the user sees in the HUD is exactly what the brush will paint.
    """
    width, height = size
    layer = new_layer(width, height)
    margin = width * 0.14
    points: list[tuple[float, float, float]] = []
    for step in range(19):
        t = step / 18.0
        x = margin + (width - 2.0 * margin) * t
        y = height / 2.0 + math.sin(t * math.pi * 1.7 + phase) * (height * 0.27)
        points.append((x, y, 260.0))
    stroke = Stroke(
        brush=brush,
        color_index=color_index,
        thickness=thickness,
        opacity=opacity,
        seed=int(brush) * 977 + 13,
        points=points,
    )
    StrokeRasteriser.render_complete(stroke, layer)
    return layer


# =============================================================================
# SECTION 9 — Shape recognition
# =============================================================================


class ShapeKind(Enum):
    """Geometric primitives the recogniser can snap a freehand stroke to."""

    CIRCLE = "CIRCLE"
    RECTANGLE = "RECTANGLE"
    TRIANGLE = "TRIANGLE"
    ARROW = "ARROW"
    LINE = "LINE"


@dataclass(frozen=True, slots=True)
class ShapeFit:
    """An idealised geometry that replaces a recognised freehand stroke."""

    kind: ShapeKind
    points: tuple[tuple[float, float], ...]


class ShapeRecogniser:
    """Classifies a freehand path as one of five primitives, or rejects it.

    The tests are ordered cheapest-and-most-specific first. Every threshold is
    expressed as a fraction of the stroke's own length so recognition behaves the
    same for a thumbnail-sized triangle and a full-screen one.
    """

    #: A stroke shorter than this (pixels of path length) is never snapped.
    MIN_PATH_LENGTH: Final[float] = 90.0
    MIN_POINTS: Final[int] = 8
    #: Endpoint gap below this fraction of path length means "closed".
    CLOSURE_RATIO: Final[float] = 0.26
    #: Radial deviation below this fraction of mean radius means "circle".
    CIRCLE_TOLERANCE: Final[float] = 0.19
    #: Perpendicular deviation below this fraction of length means "line".
    LINE_TOLERANCE: Final[float] = 0.055
    #: Sampling step of the generated ideal geometry.
    OUTPUT_SPACING: Final[float] = 2.5

    @classmethod
    def identify(cls, raw_points: Sequence[tuple[float, float, float]]) -> ShapeFit | None:
        """Return the primitive that best matches ``raw_points``, or ``None``."""
        if len(raw_points) < cls.MIN_POINTS:
            return None
        points = np.asarray([(x, y) for x, y, _ in raw_points], dtype=np.float32)
        deltas = np.diff(points, axis=0)
        length = float(np.sum(np.linalg.norm(deltas, axis=1)))
        if length < cls.MIN_PATH_LENGTH:
            return None
        closure = float(np.linalg.norm(points[-1] - points[0]))
        if closure < cls.CLOSURE_RATIO * length:
            return cls._identify_closed(points, length)
        return cls._identify_open(points, length)

    @classmethod
    def _identify_closed(cls, points: np.ndarray, length: float) -> ShapeFit | None:
        centroid = points.mean(axis=0)
        radii = np.linalg.norm(points - centroid, axis=1)
        mean_radius = float(radii.mean())
        if mean_radius < 12.0:
            return None
        if float(radii.std()) / mean_radius < cls.CIRCLE_TOLERANCE and cls._angular_sweep(points, centroid) > 4.9:
            return cls._make_circle(centroid, mean_radius)
        contour = points.reshape(-1, 1, 2)
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.042 * perimeter, True).reshape(-1, 2)
        if len(approx) == 3:
            return cls._make_polygon(ShapeKind.TRIANGLE, [(float(v[0]), float(v[1])) for v in approx])
        if len(approx) == 4:
            box = cv2.boxPoints(cv2.minAreaRect(contour))
            return cls._make_polygon(ShapeKind.RECTANGLE, [(float(v[0]), float(v[1])) for v in box])
        return None

    @classmethod
    def _identify_open(cls, points: np.ndarray, length: float) -> ShapeFit | None:
        start = points[0].astype(np.float64)
        end = points[-1].astype(np.float64)
        axis = end - start
        span = float(np.linalg.norm(axis))
        if span > 1e-3:
            normal = np.array([-axis[1], axis[0]], dtype=np.float64) / span
            deviation = float(np.max(np.abs((points.astype(np.float64) - start) @ normal)))
            if deviation < cls.LINE_TOLERANCE * length and span > 60.0:
                return ShapeFit(ShapeKind.LINE, cls._densify([tuple(start), tuple(end)]))
        contour = points.reshape(-1, 1, 2)
        approx = cv2.approxPolyDP(contour, 0.05 * cv2.arcLength(contour, False), False).reshape(-1, 2)
        if len(approx) == 3:
            return cls._make_arrow(approx.astype(np.float64))
        return None

    @staticmethod
    def _angular_sweep(points: np.ndarray, centroid: np.ndarray) -> float:
        """Total unwrapped angle swept around ``centroid`` — rejects arcs."""
        relative = points - centroid
        angles = np.unwrap(np.arctan2(relative[:, 1], relative[:, 0]))
        return float(abs(angles[-1] - angles[0]))

    @classmethod
    def _make_circle(cls, centre: np.ndarray, radius: float) -> ShapeFit:
        steps = max(48, int(radius * 1.2))
        ring = [
            (
                float(centre[0] + math.cos(step / steps * math.tau) * radius),
                float(centre[1] + math.sin(step / steps * math.tau) * radius),
            )
            for step in range(steps + 1)
        ]
        return ShapeFit(ShapeKind.CIRCLE, cls._densify(ring))

    @classmethod
    def _make_polygon(cls, kind: ShapeKind, vertices: list[tuple[float, float]]) -> ShapeFit:
        closed = [*vertices, vertices[0]]
        return ShapeFit(kind, cls._densify(closed))

    @classmethod
    def _make_arrow(cls, vertices: np.ndarray) -> ShapeFit | None:
        tail, tip, barb = vertices
        shaft_vector = tip - tail
        shaft = float(np.linalg.norm(shaft_vector))
        barb_length = float(np.linalg.norm(barb - tip))
        if shaft < 70.0 or not 0.10 < barb_length / shaft < 0.60:
            return None
        to_tail = (tail - tip) / max(shaft, 1e-6)
        to_barb = (barb - tip) / max(barb_length, 1e-6)
        head_angle = math.degrees(math.acos(clamp(float(np.dot(to_tail, to_barb)), -1.0, 1.0)))
        if not 12.0 < head_angle < 95.0:
            return None
        direction = math.atan2(shaft_vector[1], shaft_vector[0])
        wing = math.radians(30.0)
        head = max(barb_length, shaft * 0.18)
        left = (tip[0] - math.cos(direction - wing) * head, tip[1] - math.sin(direction - wing) * head)
        right = (tip[0] - math.cos(direction + wing) * head, tip[1] - math.sin(direction + wing) * head)
        tip_point = (float(tip[0]), float(tip[1]))
        path = [(float(tail[0]), float(tail[1])), tip_point, left, tip_point, right]
        return ShapeFit(ShapeKind.ARROW, cls._densify(path))

    @classmethod
    def _densify(cls, polyline: Sequence[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
        return tuple(resample_polyline(polyline, cls.OUTPUT_SPACING))


# =============================================================================
# SECTION 10 — Paint document
# =============================================================================


@dataclass(frozen=True, slots=True)
class BrushSettings:
    """The user's current painting parameters."""

    brush: BrushId = BrushId.NEON
    color_index: int = 4
    thickness_index: int = 2
    opacity: float = 1.0
    eraser_scale: float = 3.0

    @property
    def thickness(self) -> int:
        """Selected stroke width in pixels."""
        return THICKNESSES[self.thickness_index]

    @property
    def slot(self) -> ColorSlot:
        """The selected palette slot."""
        return PALETTE[self.color_index]


class PaintDocument:
    """Owns the artwork: the vector stroke list plus its rasterised cache.

    Design note — the canvas is deliberately *derived* state. ``_base`` holds
    every committed stroke, ``_stroke_buffer`` holds only the stroke currently
    being drawn, and ``_canvas`` is their composite. Because the live stroke never
    touches ``_base``, each frame only has to recomposite the few hundred pixels
    the newest samples touched, which is what keeps the frame budget flat no
    matter how full the canvas gets.
    """

    #: Maximum number of undo states retained.
    HISTORY_LIMIT: Final[int] = 100

    def __init__(self, width: int, height: int) -> None:
        self._width = width
        self._height = height
        self._base = new_layer(width, height)
        self._stroke_buffer = new_layer(width, height)
        self._canvas = new_layer(width, height)
        self._scratch = new_layer(width, height)
        self._strokes: list[Stroke] = []
        self._undo_stack: deque[list[Stroke]] = deque(maxlen=self.HISTORY_LIMIT)
        self._redo_stack: deque[list[Stroke]] = deque(maxlen=self.HISTORY_LIMIT)
        self._live_stroke: Stroke | None = None
        self._rasteriser: StrokeRasteriser | None = None
        self._live_rect: Rect | None = None
        self._content_rect: Rect | None = None
        self._seed_counter = 0

    # -- state -------------------------------------------------------------

    @property
    def canvas(self) -> np.ndarray:
        """The composited BGRA artwork layer."""
        return self._canvas

    @property
    def content_rect(self) -> Rect | None:
        """Bounding box of every committed painted pixel, or ``None`` when empty."""
        return self._content_rect

    @property
    def dirty_rect(self) -> Rect | None:
        """Bounding box of everything visible, including the in-progress stroke."""
        return union_rect(self._content_rect, self._live_rect)

    @property
    def is_drawing(self) -> bool:
        """Whether a stroke is currently open."""
        return self._live_stroke is not None

    @property
    def stroke_count(self) -> int:
        """Number of committed strokes."""
        return len(self._strokes)

    @property
    def can_undo(self) -> bool:
        """Whether an undo state is available."""
        return bool(self._undo_stack)

    @property
    def can_redo(self) -> bool:
        """Whether a redo state is available."""
        return bool(self._redo_stack)

    @property
    def is_empty(self) -> bool:
        """Whether the artwork contains nothing at all."""
        return not self._strokes and self._live_stroke is None

    # -- drawing -----------------------------------------------------------

    def begin_stroke(self, settings: BrushSettings, eraser: bool) -> None:
        """Open a new stroke using the supplied settings."""
        if self._live_stroke is not None:
            self.end_stroke(snap_shapes=False)
        self._seed_counter += 1
        thickness = int(settings.thickness * settings.eraser_scale) if eraser else settings.thickness
        self._live_stroke = Stroke(
            brush=settings.brush,
            color_index=settings.color_index,
            thickness=thickness,
            opacity=settings.opacity,
            seed=self._seed_counter,
            is_eraser=eraser,
        )
        self._rasteriser = StrokeRasteriser(self._live_stroke)
        self._live_rect = None

    def extend_stroke(self, x: float, y: float, velocity: float) -> None:
        """Append a tracked point to the open stroke and update the canvas."""
        if self._live_stroke is None or self._rasteriser is None:
            return
        self._live_stroke.points.append((x, y, velocity))
        dirty = self._rasteriser.feed(self._stroke_buffer, x, y, velocity)
        if dirty is not None:
            self._refresh_canvas(dirty)
            self._live_rect = union_rect(self._live_rect, dirty)

    def end_stroke(self, snap_shapes: bool) -> str | None:
        """Close the open stroke, optionally snapping it to ideal geometry."""
        stroke = self._live_stroke
        if stroke is None or self._rasteriser is None:
            return None
        dirty = self._rasteriser.flush(self._stroke_buffer)
        if dirty is not None:
            self._refresh_canvas(dirty)
            self._live_rect = union_rect(self._live_rect, dirty)
        self._live_stroke = None
        self._rasteriser = None

        if not stroke.points:
            self._live_rect = None
            return None

        shape_name: str | None = None
        if snap_shapes and not stroke.is_eraser:
            fit = ShapeRecogniser.identify(stroke.points)
            if fit is not None:
                shape_name = fit.kind.value
                stroke.shape_points = list(fit.points)
                stroke.shape_name = shape_name
                self._reraster_live(stroke)

        self._push_history()
        self._strokes.append(stroke)
        self._commit_live(stroke)
        return shape_name

    def cancel_stroke(self) -> None:
        """Discard the open stroke without committing it."""
        if self._live_stroke is None:
            return
        self._live_stroke = None
        self._rasteriser = None
        if self._live_rect is not None:
            self._clear_region(self._stroke_buffer, self._live_rect)
            self._copy_region(self._base, self._canvas, self._live_rect)
            self._live_rect = None

    # -- history -----------------------------------------------------------

    def undo(self) -> bool:
        """Restore the previous document state."""
        if not self._undo_stack:
            return False
        self._redo_stack.append(list(self._strokes))
        self._strokes = self._undo_stack.pop()
        self._rebuild()
        return True

    def redo(self) -> bool:
        """Re-apply a state removed by :meth:`undo`."""
        if not self._redo_stack:
            return False
        self._undo_stack.append(list(self._strokes))
        self._strokes = self._redo_stack.pop()
        self._rebuild()
        return True

    def clear(self) -> bool:
        """Erase the artwork, keeping the action undoable."""
        if not self._strokes and self._live_stroke is None:
            return False
        self.cancel_stroke()
        self._push_history()
        self._strokes = []
        self._rebuild()
        return True

    # -- export ------------------------------------------------------------

    def render_artwork(self, scale: float = 1.0) -> np.ndarray:
        """Rasterise the whole document at ``scale`` into a fresh BGRA layer.

        Because strokes are vectors, a scaled render is genuinely resolution
        independent rather than an upscaled bitmap.
        """
        if abs(scale - 1.0) < 1e-6 and self._live_stroke is None:
            return self._canvas.copy()
        width = round(self._width * scale)
        height = round(self._height * scale)
        target = new_layer(width, height)
        scratch = new_layer(width, height)
        strokes = list(self._strokes)
        if self._live_stroke is not None:
            strokes.append(self._live_stroke)
        for stroke in strokes:
            dirty = StrokeRasteriser.render_complete(stroke, scratch, scale)
            if dirty is None:
                continue
            if stroke.is_eraser:
                erase_with(target, scratch, dirty)
            else:
                composite_over(target, scratch, dirty)
            self._clear_region(scratch, dirty)
        return target

    # -- internals ---------------------------------------------------------

    def _push_history(self) -> None:
        self._undo_stack.append(list(self._strokes))
        self._redo_stack.clear()

    def _refresh_canvas(self, rect: Rect, erasing: bool | None = None) -> None:
        """Recompute ``canvas = base (+/-) stroke_buffer`` inside ``rect``."""
        if erasing is None:
            erasing = self._live_stroke is not None and self._live_stroke.is_eraser
        self._copy_region(self._base, self._canvas, rect)
        if erasing:
            erase_with(self._canvas, self._stroke_buffer, rect)
        else:
            composite_over(self._canvas, self._stroke_buffer, rect)

    def _reraster_live(self, stroke: Stroke) -> None:
        """Re-render an already-drawn stroke after its geometry was replaced."""
        previous = self._live_rect
        if previous is not None:
            self._clear_region(self._stroke_buffer, previous)
        fresh = StrokeRasteriser.render_complete(stroke, self._stroke_buffer)
        combined = union_rect(previous, fresh)
        if combined is not None:
            self._refresh_canvas(combined, erasing=stroke.is_eraser)
        self._live_rect = combined

    def _commit_live(self, stroke: Stroke) -> None:
        """Fold the finished stroke buffer into the committed base layer."""
        rect = self._live_rect
        if rect is None:
            return
        if stroke.is_eraser:
            erase_with(self._base, self._stroke_buffer, rect)
        else:
            composite_over(self._base, self._stroke_buffer, rect)
            self._content_rect = union_rect(self._content_rect, rect)
        self._copy_region(self._base, self._canvas, rect)
        self._clear_region(self._stroke_buffer, rect)
        self._live_rect = None

    def _rebuild(self) -> None:
        """Re-rasterise the whole document from its vector stroke list."""
        started = time.perf_counter()
        self._base[:] = 0
        self._stroke_buffer[:] = 0
        self._content_rect = None
        for stroke in self._strokes:
            dirty = StrokeRasteriser.render_complete(stroke, self._scratch)
            if dirty is None:
                continue
            if stroke.is_eraser:
                erase_with(self._base, self._scratch, dirty)
            else:
                composite_over(self._base, self._scratch, dirty)
                self._content_rect = union_rect(self._content_rect, dirty)
            self._clear_region(self._scratch, dirty)
        self._canvas[:] = self._base
        LOGGER.debug("Rebuilt %d strokes in %.1f ms", len(self._strokes), (time.perf_counter() - started) * 1000.0)

    @staticmethod
    def _clear_region(layer: np.ndarray, rect: Rect) -> None:
        layer[rect[1] : rect[3], rect[0] : rect[2]] = 0

    @staticmethod
    def _copy_region(source: np.ndarray, target: np.ndarray, rect: Rect) -> None:
        target[rect[1] : rect[3], rect[0] : rect[2]] = source[rect[1] : rect[3], rect[0] : rect[2]]


# =============================================================================
# SECTION 11 — Artwork export
# =============================================================================


class ExportFormat(Enum):
    """Supported artwork export targets."""

    PNG = "PNG"
    PNG_TRANSPARENT = "PNG_TRANSPARENT"
    JPG = "JPG"
    HIGH_RES = "HIGH_RES"

    @property
    def label(self) -> str:
        """Human-readable name shown in the interface."""
        return {
            ExportFormat.PNG: "PNG",
            ExportFormat.PNG_TRANSPARENT: "PNG · ALPHA",
            ExportFormat.JPG: "JPG",
            ExportFormat.HIGH_RES: "PNG · HI-RES",
        }[self]

    @property
    def suffix(self) -> str:
        """File extension including the leading dot."""
        return ".jpg" if self is ExportFormat.JPG else ".png"


class ArtworkExporter:
    """Writes the document to disk in every supported format."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    @property
    def output_dir(self) -> Path:
        """Directory artwork is written to."""
        return self._config.output_dir

    def build_image(
        self,
        document: PaintDocument,
        fmt: ExportFormat,
        backdrop: np.ndarray | None,
    ) -> Image.Image:
        """Compose the artwork into a PIL image ready to be saved."""
        scale = float(self._config.export_scale) if fmt is ExportFormat.HIGH_RES else 1.0
        artwork = document.render_artwork(scale)
        if fmt is ExportFormat.PNG_TRANSPARENT:
            rgba = cv2.cvtColor(artwork, cv2.COLOR_BGRA2RGBA)
            return Image.fromarray(rgba, mode="RGBA")
        height, width = artwork.shape[:2]
        if fmt is ExportFormat.JPG and backdrop is not None:
            canvas_bgr = cv2.resize(backdrop, (width, height), interpolation=cv2.INTER_LANCZOS4)
        else:
            canvas_bgr = np.full((height, width, 3), THEME.void, dtype=np.uint8)
        flatten_onto(canvas_bgr, artwork)
        return Image.fromarray(cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB), mode="RGB")

    def save(
        self,
        document: PaintDocument,
        fmt: ExportFormat,
        backdrop: np.ndarray | None = None,
        destination: Path | None = None,
    ) -> Path:
        """Export the artwork and return the path written."""
        image = self.build_image(document, fmt, backdrop)
        target = destination or self._auto_path(fmt)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix.lower() in {".jpg", ".jpeg"}:
            image.convert("RGB").save(target, quality=95, subsampling=0)
        else:
            image.save(target, optimize=True)
        LOGGER.info("Exported %s artwork to %s", fmt.label, target)
        return target

    def _auto_path(self, fmt: ExportFormat) -> Path:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        return self._config.output_dir / f"aivp-{stamp}-{fmt.name.lower()}{fmt.suffix}"


# =============================================================================
# SECTION 12 — Camera and hand tracking
# =============================================================================


class CameraError(RuntimeError):
    """Raised when no usable camera device could be opened."""


class ModelError(RuntimeError):
    """Raised when the hand-landmark bundle cannot be provisioned."""


class LatestSlot:
    """A one-item mailbox where a new value always replaces the old one.

    Painting must react to the newest frame, never to a queued backlog, so a
    bounded queue with drop-oldest semantics is exactly the right primitive.
    """

    __slots__ = ("_item", "_lock", "_ready")

    def __init__(self) -> None:
        self._item: object | None = None
        self._lock = threading.Lock()
        self._ready = threading.Event()

    def put(self, item: object) -> None:
        """Publish ``item``, discarding anything not yet consumed."""
        with self._lock:
            self._item = item
        self._ready.set()

    def take(self, timeout: float | None = None) -> object | None:
        """Block until an item is available, then consume it."""
        if not self._ready.wait(timeout):
            return None
        with self._lock:
            item = self._item
            self._item = None
            self._ready.clear()
        return item

    def peek(self) -> object | None:
        """Read the current item without consuming it."""
        with self._lock:
            return self._item


class CameraStream:
    """Background camera reader that always exposes the most recent frame."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._capture: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._frames_read = 0

    @property
    def frames_read(self) -> int:
        """Total frames pulled from the device since start-up."""
        return self._frames_read

    def start(self) -> None:
        """Open the device and begin reading in the background."""
        capture = self._open_capture()
        # Request MJPG before the resolution: many webcams are limited to well
        # under 30 fps at 720p on uncompressed YUY2 but reach 30-60 fps on MJPG.
        # Cameras that do not offer it simply keep their existing format.
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.camera_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.camera_height)
        capture.set(cv2.CAP_PROP_FPS, self._config.target_fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        received, frame = capture.read()
        if not received or frame is None:
            capture.release()
            raise CameraError(f"Camera {self._config.camera_index} opened but returned no frames.")
        self._capture = capture
        with self._lock:
            self._frame = cv2.flip(frame, 1)
        self._thread = threading.Thread(target=self._run, name="aivp-camera", daemon=True)
        self._thread.start()
        LOGGER.info("Camera %d streaming at %dx%d", self._config.camera_index, frame.shape[1], frame.shape[0])

    def _open_capture(self) -> cv2.VideoCapture:
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
        for backend in backends:
            capture = cv2.VideoCapture(self._config.camera_index, backend)
            if capture.isOpened():
                return capture
            capture.release()
        raise CameraError(
            f"Could not open camera index {self._config.camera_index}. "
            "Close any other application using the webcam, or set AIVP_CAMERA_INDEX."
        )

    def _run(self) -> None:
        assert self._capture is not None
        while not self._stop.is_set():
            received, frame = self._capture.read()
            if not received or frame is None:
                time.sleep(0.005)
                continue
            mirrored = cv2.flip(frame, 1)  # mirror so the user's motion feels direct
            with self._lock:
                self._frame = mirrored
                self._frames_read += 1

    def latest(self) -> np.ndarray | None:
        """Return the most recently captured (mirrored) BGR frame."""
        with self._lock:
            return self._frame

    def stop(self) -> None:
        """Stop reading and release the device."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        LOGGER.info("Camera released")


def ensure_hand_model(path: Path, on_progress: Callable[[float], None] | None = None) -> Path:
    """Return a usable hand-landmark bundle, downloading it once if required."""
    if path.is_file() and path.stat().st_size >= HAND_MODEL_MIN_BYTES:
        return path
    if path.is_file():
        LOGGER.warning("Cached model at %s looks truncated; re-downloading", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    LOGGER.info("Fetching MediaPipe hand landmarker bundle -> %s", path)
    request = urllib.request.Request(HAND_MODEL_URL, headers={"User-Agent": "ai-virtual-painter/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as sink:
            total = int(response.headers.get("Content-Length") or 0)
            written = 0
            while chunk := response.read(262_144):
                sink.write(chunk)
                written += len(chunk)
                if on_progress is not None and total:
                    on_progress(written / total)
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        temporary.unlink(missing_ok=True)
        raise ModelError(
            "Could not download the MediaPipe hand model.\n\n"
            f"{error}\n\nDownload it manually from:\n{HAND_MODEL_URL}\n"
            f"and place it at:\n{path}\n(or set AIVP_HAND_MODEL to its location)."
        ) from error
    if temporary.stat().st_size < HAND_MODEL_MIN_BYTES:
        temporary.unlink(missing_ok=True)
        raise ModelError("The downloaded hand model is incomplete; please retry.")
    temporary.replace(path)
    return path


@dataclass(frozen=True, slots=True)
class HandObservation:
    """One inference result, expressed in canvas pixel coordinates."""

    landmarks: np.ndarray  # (21, 2) float32 pixel positions
    handedness: str
    timestamp: float
    #: Wrist-to-middle-knuckle distance; used to make thresholds depth invariant.
    palm_size: float


# MediaPipe hand landmark indices used throughout the gesture engine.
WRIST: Final = 0
FINGER_TIPS: Final = (4, 8, 12, 16, 20)
FINGER_JOINTS: Final = (3, 6, 10, 14, 18)
MIDDLE_MCP: Final = 9
INDEX_TIP: Final = 8
THUMB_TIP: Final = 4
MIDDLE_TIP: Final = 12
#: Skeleton edges for the HUD wireframe overlay.
HAND_EDGES: Final[tuple[tuple[int, int], ...]] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)


class HandTracker:
    """Runs MediaPipe hand landmarking on a background thread.

    Inference is performed on a downscaled copy of the frame — landmark accuracy
    is essentially unchanged at 480 px wide while the cost drops by roughly two
    thirds, which is what leaves enough of the frame budget for the painting
    engine and the HUD.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._input = LatestSlot()
        self._observation: HandObservation | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._landmarker: mp_vision.HandLandmarker | None = None
        self._timestamp_ms = 0
        self._inference_ms = 0.0

    @property
    def inference_ms(self) -> float:
        """Duration of the most recent inference, in milliseconds."""
        return self._inference_ms

    def start(self) -> None:
        """Create the landmarker and start the inference thread."""
        model_path = ensure_hand_model(self._config.model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._thread = threading.Thread(target=self._run, name="aivp-hands", daemon=True)
        self._thread.start()
        LOGGER.info("Hand landmarker ready (model: %s)", model_path.name)

    def submit(self, frame: np.ndarray) -> None:
        """Offer a frame for inference; older unprocessed frames are dropped."""
        self._input.put(frame)

    def latest(self) -> HandObservation | None:
        """Return the most recent observation, or ``None`` if no hand is tracked."""
        with self._lock:
            return self._observation

    def _run(self) -> None:
        scale_target = self._config.inference_width
        while not self._stop.is_set():
            frame = self._input.take(timeout=0.2)
            if frame is None or self._landmarker is None:
                continue
            assert isinstance(frame, np.ndarray)
            height, width = frame.shape[:2]
            scale = scale_target / float(width)
            small = cv2.resize(frame, (scale_target, max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            self._timestamp_ms += 16
            started = time.perf_counter()
            try:
                result = self._landmarker.detect_for_video(image, self._timestamp_ms)
            except (RuntimeError, ValueError) as error:
                LOGGER.warning("Hand inference failed: %s", error)
                continue
            self._inference_ms = (time.perf_counter() - started) * 1000.0
            observation = self._to_observation(result, width, height)
            with self._lock:
                self._observation = observation

    @staticmethod
    def _to_observation(result: mp_vision.HandLandmarkerResult, width: int, height: int) -> HandObservation | None:
        hands = getattr(result, "hand_landmarks", None)
        if not hands:
            return None
        landmarks = np.asarray([(point.x * width, point.y * height) for point in hands[0]], dtype=np.float32)
        handedness_groups = getattr(result, "handedness", None) or []
        label = "HAND"
        if handedness_groups and handedness_groups[0]:
            # The frame is mirrored before inference, so MediaPipe's label is flipped
            # relative to the physical hand; report what the user actually sees.
            raw = getattr(handedness_groups[0][0], "category_name", "") or ""
            label = {"Left": "RIGHT", "Right": "LEFT"}.get(raw, raw.upper() or "HAND")
        palm = float(np.linalg.norm(landmarks[MIDDLE_MCP] - landmarks[WRIST]))
        return HandObservation(landmarks, label, time.perf_counter(), max(palm, 1.0))

    def stop(self) -> None:
        """Stop inference and release the landmarker."""
        self._stop.set()
        self._input.put(None)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        closer = getattr(self._landmarker, "close", None)
        if callable(closer):
            closer()
        self._landmarker = None
        LOGGER.info("Hand landmarker closed")


# =============================================================================
# SECTION 13 — Gesture engine
# =============================================================================


class HandPose(Enum):
    """Discrete hand poses the engine recognises."""

    NONE = "NONE"
    DRAW = "DRAW"
    SELECT = "SELECT"
    ERASE = "ERASE"
    PALM = "PALM"
    UNKNOWN = "UNKNOWN"


class GestureEvent(Enum):
    """One-shot commands triggered by transient gestures."""

    UNDO = auto()
    REDO = auto()
    CLEAR = auto()
    SAVE = auto()


@dataclass(frozen=True, slots=True)
class GestureState:
    """Everything the application needs to know about the hand this frame."""

    present: bool = False
    pose: HandPose = HandPose.NONE
    tool: Tool = Tool.IDLE
    fingers: tuple[bool, bool, bool, bool, bool] = (False,) * 5
    pointer: tuple[float, float] | None = None
    velocity: float = 0.0
    landmarks: np.ndarray | None = None
    handedness: str = ""
    palm_size: float = 0.0
    pinch_index: float = 1.0
    pinch_middle: float = 1.0
    hold_label: str = ""
    hold_progress: float = 0.0
    stationary: bool = False
    events: tuple[GestureEvent, ...] = ()


class GestureEngine:
    """Turns raw landmarks into a stable, debounced interaction state.

    Every threshold here is normalised by the palm width, so gestures behave the
    same whether the hand is near the lens or a metre away, and finger extension
    is measured radially from the wrist rather than by comparing screen Y values,
    which keeps it correct when the hand is rotated or upside down.
    """

    #: Fraction of palm width below which two fingertips count as pinched.
    PINCH_ON: Final[float] = 0.42
    PINCH_OFF: Final[float] = 0.60
    PINCH_COOLDOWN: Final[float] = 0.45
    #: Seconds a hold gesture must be sustained before it fires.
    HOLD_SECONDS: Final[float] = 2.0
    #: Radius (in pixels) within which the pointer counts as motionless.
    STILL_RADIUS: Final[float] = 14.0
    STILL_WINDOW: Final[float] = 0.35
    #: An observation older than this means tracking was lost.
    STALE_SECONDS: Final[float] = 0.45
    #: Pose must persist this long before the tool switches; kills flicker.
    POSE_DEBOUNCE: Final[float] = 0.09

    def __init__(self) -> None:
        self._pointer_filter = PointFilter(min_cutoff=1.05, beta=0.010)
        self._skeleton: np.ndarray | None = None
        self._previous_pointer: tuple[float, float] | None = None
        self._previous_time = 0.0
        self._velocity = 0.0
        self._pinch_index_active = False
        self._pinch_middle_active = False
        self._last_pinch_time = 0.0
        self._pose_candidate = HandPose.NONE
        self._pose_since = 0.0
        self._stable_pose = HandPose.NONE
        self._hold_pose: HandPose | None = None
        self._hold_started = 0.0
        self._hold_fired = False
        self._trail: deque[tuple[float, tuple[float, float]]] = deque(maxlen=48)

    def reset(self) -> None:
        """Forget all temporal state (used when tracking is lost)."""
        self._pointer_filter.reset()
        self._skeleton = None
        self._previous_pointer = None
        self._velocity = 0.0
        self._pose_candidate = HandPose.NONE
        self._stable_pose = HandPose.NONE
        self._hold_pose = None
        self._hold_fired = False
        self._trail.clear()

    def update(self, observation: HandObservation | None, now: float) -> GestureState:
        """Advance the engine by one frame and return the resulting state."""
        if observation is None or now - observation.timestamp > self.STALE_SECONDS:
            self.reset()
            return GestureState()

        landmarks = self._smooth_skeleton(observation.landmarks)
        fingers = self._extended_fingers(landmarks)
        palm = observation.palm_size
        pinch_index = float(np.linalg.norm(landmarks[THUMB_TIP] - landmarks[INDEX_TIP])) / palm
        pinch_middle = float(np.linalg.norm(landmarks[THUMB_TIP] - landmarks[MIDDLE_TIP])) / palm

        raw_pointer = (float(landmarks[INDEX_TIP][0]), float(landmarks[INDEX_TIP][1]))
        pointer = self._pointer_filter.filter(raw_pointer, now)
        self._update_velocity(pointer, now)
        self._trail.append((now, pointer))

        events: list[GestureEvent] = []
        events.extend(self._detect_pinches(pinch_index, pinch_middle, now))

        pose = self._classify_pose(fingers)
        stable = self._debounce(pose, now)
        stationary = self._is_stationary(now)
        hold_label, hold_progress, hold_event = self._update_hold(stable, stationary, now)
        if hold_event is not None:
            events.append(hold_event)

        return GestureState(
            present=True,
            pose=stable,
            tool=self._tool_for(stable),
            fingers=fingers,
            pointer=pointer,
            velocity=self._velocity,
            landmarks=landmarks,
            handedness=observation.handedness,
            palm_size=palm,
            pinch_index=pinch_index,
            pinch_middle=pinch_middle,
            hold_label=hold_label,
            hold_progress=hold_progress,
            stationary=stationary,
            events=tuple(events),
        )

    # -- internals ---------------------------------------------------------

    def _smooth_skeleton(self, landmarks: np.ndarray) -> np.ndarray:
        if self._skeleton is None or self._skeleton.shape != landmarks.shape:
            self._skeleton = landmarks.copy()
        else:
            # Light EMA on the raw landmarks; the fingertip gets its own 1-Euro
            # filter afterwards, so this only needs to steady the wireframe and
            # the pinch distances without adding perceptible lag.
            self._skeleton += (landmarks - self._skeleton) * 0.6
        return self._skeleton

    @staticmethod
    def _extended_fingers(landmarks: np.ndarray) -> tuple[bool, bool, bool, bool, bool]:
        """Radial extension test — rotation invariant, unlike a Y comparison."""
        wrist = landmarks[WRIST]
        tip_span = np.linalg.norm(landmarks[list(FINGER_TIPS)] - wrist, axis=1)
        joint_span = np.linalg.norm(landmarks[list(FINGER_JOINTS)] - wrist, axis=1)
        ratios = tip_span / np.maximum(joint_span, 1e-3)
        thresholds = np.array([1.12, 1.06, 1.06, 1.06, 1.04], dtype=np.float32)
        return tuple(bool(flag) for flag in ratios > thresholds)  # type: ignore[return-value]

    def _update_velocity(self, pointer: tuple[float, float], now: float) -> None:
        if self._previous_pointer is not None:
            delta_time = max(now - self._previous_time, 1e-3)
            instant = math.dist(pointer, self._previous_pointer) / delta_time
            self._velocity += (instant - self._velocity) * 0.35
        self._previous_pointer = pointer
        self._previous_time = now

    def _detect_pinches(self, index_ratio: float, middle_ratio: float, now: float) -> list[GestureEvent]:
        events: list[GestureEvent] = []
        cooled = now - self._last_pinch_time > self.PINCH_COOLDOWN
        # The closer pinch wins so a partially-curled hand cannot fire both.
        index_closer = index_ratio <= middle_ratio
        if index_ratio < self.PINCH_ON and not self._pinch_index_active and index_closer:
            self._pinch_index_active = True
            if cooled:
                events.append(GestureEvent.UNDO)
                self._last_pinch_time = now
        elif index_ratio > self.PINCH_OFF:
            self._pinch_index_active = False
        if middle_ratio < self.PINCH_ON and not self._pinch_middle_active and not index_closer:
            self._pinch_middle_active = True
            if cooled:
                events.append(GestureEvent.REDO)
                self._last_pinch_time = now
        elif middle_ratio > self.PINCH_OFF:
            self._pinch_middle_active = False
        return events

    @staticmethod
    def _classify_pose(fingers: tuple[bool, bool, bool, bool, bool]) -> HandPose:
        _thumb, index, middle, ring, pinky = fingers
        if index and middle and ring and pinky:
            return HandPose.PALM
        if index and middle and ring and not pinky:
            return HandPose.ERASE
        if index and middle and not ring:
            return HandPose.SELECT
        if index and not middle:
            return HandPose.DRAW
        return HandPose.UNKNOWN

    def _debounce(self, pose: HandPose, now: float) -> HandPose:
        if pose is not self._pose_candidate:
            self._pose_candidate = pose
            self._pose_since = now
        elif now - self._pose_since >= self.POSE_DEBOUNCE:
            self._stable_pose = pose
        return self._stable_pose

    def _is_stationary(self, now: float) -> bool:
        recent = [point for stamp, point in self._trail if now - stamp <= self.STILL_WINDOW]
        if len(recent) < 4:
            return False
        centre_x = sum(point[0] for point in recent) / len(recent)
        centre_y = sum(point[1] for point in recent) / len(recent)
        return max(math.dist(point, (centre_x, centre_y)) for point in recent) < self.STILL_RADIUS

    def _update_hold(self, pose: HandPose, stationary: bool, now: float) -> tuple[str, float, GestureEvent | None]:
        """Track the two timed gestures: open-palm clear and victory save."""
        if pose is HandPose.PALM:
            target: HandPose | None = HandPose.PALM
            label = "CLEAR CANVAS"
        elif pose is HandPose.SELECT and stationary:
            target = HandPose.SELECT
            label = "SAVE ARTWORK"
        else:
            target = None
            label = ""
        if target is None:
            self._hold_pose = None
            self._hold_fired = False
            return "", 0.0, None
        if self._hold_pose is not target:
            self._hold_pose = target
            self._hold_started = now
            self._hold_fired = False
        progress = clamp((now - self._hold_started) / self.HOLD_SECONDS, 0.0, 1.0)
        if progress >= 1.0 and not self._hold_fired:
            self._hold_fired = True
            event = GestureEvent.CLEAR if target is HandPose.PALM else GestureEvent.SAVE
            return label, 1.0, event
        if self._hold_fired:
            return "", 0.0, None
        return label, progress, None

    @staticmethod
    def _tool_for(pose: HandPose) -> Tool:
        return {
            HandPose.DRAW: Tool.DRAW,
            HandPose.SELECT: Tool.SELECT,
            HandPose.ERASE: Tool.ERASE,
        }.get(pose, Tool.IDLE)


# =============================================================================
# SECTION 14 — Heads-up display
# =============================================================================


class Command(Enum):
    """Every action the interface can dispatch back to the application."""

    NONE = auto()
    SELECT_BRUSH = auto()
    SELECT_COLOR = auto()
    SELECT_THICKNESS = auto()
    SET_OPACITY = auto()
    UNDO = auto()
    REDO = auto()
    CLEAR = auto()
    SAVE = auto()
    CYCLE_EXPORT = auto()
    TOGGLE_SHAPE_AI = auto()
    TOGGLE_SKELETON = auto()
    TOGGLE_MIRROR_ART = auto()
    MINIMISE = auto()
    FULLSCREEN = auto()
    CLOSE = auto()
    DRAG_WINDOW = auto()


RegionKey = tuple[Command, int]


@dataclass(frozen=True, slots=True)
class HitRegion:
    """A clickable / dwell-selectable area published by the HUD each frame."""

    command: Command
    payload: int
    rect: Rect
    label: str = ""
    #: When positive the region is treated as a circle centred in ``rect``.
    radius: float = 0.0
    dwellable: bool = True
    #: Continuous regions (sliders) report a 0..1 position instead of firing once.
    continuous: bool = False

    @property
    def key(self) -> RegionKey:
        """Stable identity used to track hover and dwell state."""
        return (self.command, self.payload)

    @property
    def centre(self) -> tuple[int, int]:
        """Geometric centre of the region."""
        return ((self.rect[0] + self.rect[2]) // 2, (self.rect[1] + self.rect[3]) // 2)

    def contains(self, x: float, y: float) -> bool:
        """Whether a point falls inside this region."""
        if self.radius > 0.0:
            return math.dist((x, y), self.centre) <= self.radius
        return self.rect[0] <= x < self.rect[2] and self.rect[1] <= y < self.rect[3]


class ToastKind(Enum):
    """Visual tone of a transient notification."""

    INFO = auto()
    SUCCESS = auto()
    WARNING = auto()
    ACCENT = auto()

    @property
    def color(self) -> Bgr:
        """Accent colour used for the toast's rule and icon."""
        return {
            ToastKind.INFO: THEME.primary,
            ToastKind.SUCCESS: THEME.mint,
            ToastKind.WARNING: THEME.amber,
            ToastKind.ACCENT: THEME.accent,
        }[self]


@dataclass(slots=True)
class Toast:
    """A transient notification with its own fade-in / fade-out envelope."""

    title: str
    detail: str
    kind: ToastKind
    created: float
    duration: float = 2.8

    def envelope(self, now: float) -> float:
        """Return 0..1 visibility for the current moment."""
        age = now - self.created
        if age < 0.0 or age > self.duration:
            return 0.0
        fade_in = clamp(age / 0.22, 0.0, 1.0)
        fade_out = clamp((self.duration - age) / 0.45, 0.0, 1.0)
        return ease_out_cubic(fade_in) * fade_out

    def expired(self, now: float) -> bool:
        """Whether the toast has finished its lifetime."""
        return now - self.created > self.duration


@dataclass(frozen=True, slots=True)
class HudModel:
    """Immutable per-frame snapshot of everything the HUD renders."""

    settings: BrushSettings
    gesture: GestureState
    tool: Tool
    fps: float
    inference_ms: float
    stroke_count: int
    can_undo: bool
    can_redo: bool
    shape_ai: bool
    show_skeleton: bool
    art_only: bool
    export_format: ExportFormat
    output_dir: Path
    hover_key: RegionKey | None
    dwell_progress: float
    menu_open: bool
    menu_origin: tuple[int, int]
    mouse: tuple[int, int] | None
    boot_progress: float
    boot_status: str
    elapsed: float


class AnimationBank:
    """A keyed pool of eased scalars, one per animated interface element."""

    def __init__(self, speed: float = 14.0) -> None:
        self._speed = speed
        self._items: dict[object, Animated] = {}

    def drive(self, key: object, target: float, delta_time: float) -> float:
        """Retarget and advance the animation for ``key``, returning its value."""
        animation = self._items.get(key)
        if animation is None:
            animation = Animated(target, self._speed)
            self._items[key] = animation
        animation.set(target)
        return animation.update(delta_time)


def build_color_chip(slot: ColorSlot, radius: int) -> np.ndarray:
    """Rasterise a palette chip, including the gradient and rainbow modes."""
    mask = SPRITES.disc(float(radius), 0.08)
    extent = mask.shape[0]
    chip = new_layer(extent, extent)
    if slot.mode is ColorMode.SOLID:
        chip[:, :, :3] = slot.color
    else:
        columns = np.linspace(0.0, 360.0, extent, dtype=np.float32)
        strip = np.array([slot.sample(float(value)) for value in columns], dtype=np.uint8)
        chip[:, :, :3] = np.repeat(strip[None, :, :], extent, axis=0)
    chip[:, :, 3] = (mask * 255.0).astype(np.uint8)
    return chip


class HudRenderer:
    """Draws the entire interface directly into the video frame.

    Every control is painted with the same raster pipeline as the artwork, which
    is what makes genuine frosted glass, additive neon and per-element motion
    possible — a stack of Tk widgets could not sit *inside* the image, only on
    top of it.
    """

    TITLE_HEIGHT: Final = 46
    DOCK_X: Final = 22
    DOCK_CELL: Final = 36
    DOCK_SWATCH: Final = (46, 24)
    #: Left edge of the top-left telemetry column; clears the brush dock.
    TELEMETRY_X: Final = 100
    RAIL_CHIP: Final = 17
    #: Frames in the looping brush-preview animation. They are rasterised lazily,
    #: at most one per frame, so changing brush never stalls the render loop.
    PREVIEW_PHASES: Final = 12
    PREVIEW_SIZE: Final = (236, 46)
    #: Boot progress at which the launch sequence stops sweeping and dissolves.
    BOOT_SETTLED: Final = 0.62
    LOGO_SIZE: Final = 148
    RADIUS_HUB: Final = 46
    RADIUS_THICKNESS: Final = 84
    RADIUS_BRUSH: Final = 136
    RADIUS_COLOR: Final = 192
    MENU_SWATCH: Final = (40, 20)

    def __init__(self, width: int, height: int) -> None:
        self._width = width
        self._height = height
        self._shade, self._chrome = self._build_atmosphere(width, height)
        self._anim = AnimationBank(speed=13.0)
        self._mode_color = AnimatedColor(THEME.primary, speed=8.0)
        self._dock_alpha = Animated(1.0, speed=7.0)
        self._menu_scale = Animated(0.0, speed=15.0)
        self._toasts: list[Toast] = []
        self._chips = {index: build_color_chip(slot, self.RAIL_CHIP) for index, slot in enumerate(PALETTE)}
        self._menu_chips = {index: build_color_chip(slot, 13) for index, slot in enumerate(PALETTE)}
        self._swatch_cache: dict[tuple, np.ndarray] = {}
        self._preview_key: tuple | None = None
        self._preview_strip: list[np.ndarray | None] = [None] * self.PREVIEW_PHASES
        self._pointer_trail: deque[tuple[float, float]] = deque(maxlen=26)
        self._veil = np.full((height, width, 3), THEME.void, dtype=np.uint8)
        self._logo = self._load_logo()

    @staticmethod
    def _load_logo() -> np.ndarray | None:
        """Load the optional application icon as a rounded BGRA sprite."""
        path = find_app_icon()
        if path is None:
            return None
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            LOGGER.warning("Could not decode the application icon at %s", path)
            return None
        extent = HudRenderer.LOGO_SIZE
        image = cv2.resize(image, (extent, extent), interpolation=cv2.INTER_AREA)
        sprite = new_layer(extent, extent)
        if image.ndim == 2:
            sprite[:, :, :3] = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            sprite[:, :, :3] = image[:, :, :3]
        existing = image[:, :, 3] if image.ndim == 3 and image.shape[2] == 4 else None
        rounded = rounded_rect_mask(extent, extent, extent // 5)
        sprite[:, :, 3] = rounded if existing is None else cv2.min(rounded, existing)
        LOGGER.info("Using application icon %s", path.name)
        return sprite

    # -- notifications -----------------------------------------------------

    def notify(self, title: str, detail: str = "", kind: ToastKind = ToastKind.INFO) -> None:
        """Queue a transient notification."""
        self._toasts.append(Toast(title.upper(), detail.upper(), kind, time.perf_counter()))
        if len(self._toasts) > 4:
            self._toasts.pop(0)

    # -- static atmosphere -------------------------------------------------

    @staticmethod
    def _build_atmosphere(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        """Bake the vignette, grain and chrome furniture into two static maps.

        Applying these as one multiply plus one add keeps the cinematic grade at
        roughly a millisecond a frame instead of the several a per-pixel float
        pass would cost.
        """
        xs = (np.linspace(-1.0, 1.0, width, dtype=np.float32)) ** 2
        ys = (np.linspace(-1.0, 1.0, height, dtype=np.float32)) ** 2
        radial = np.sqrt(ys[:, None] + xs[None, :]) / math.sqrt(2.0)
        vignette = np.clip(1.0 - radial**2.4 * 0.62, 0.42, 1.0)
        scanlines = 1.0 - (np.arange(height, dtype=np.float32) % 3 == 0) * 0.045
        shade = np.clip(vignette * scanlines[:, None], 0.0, 1.0)
        grain = np.random.default_rng(7).normal(1.0, 0.012, size=(height, width)).astype(np.float32)
        shade = np.clip(shade * grain, 0.0, 1.0)
        shade_map = np.repeat((shade * 255.0).astype(np.uint8)[:, :, None], 3, axis=2)

        chrome = np.zeros((height, width, 3), dtype=np.uint8)
        bracket, margin, thickness = 54, 18, 2
        corners = (
            ((margin, margin), (1, 1)),
            ((width - margin, margin), (-1, 1)),
            ((margin, height - margin), (1, -1)),
            ((width - margin, height - margin), (-1, -1)),
        )
        for (corner_x, corner_y), (dx, dy) in corners:
            cv2.line(chrome, (corner_x, corner_y), (corner_x + bracket * dx, corner_y), THEME.primary_deep, thickness)
            cv2.line(chrome, (corner_x, corner_y), (corner_x, corner_y + bracket * dy), THEME.primary_deep, thickness)
        for index in range(1, 24):  # measurement ticks along the top and bottom rules
            tick_x = int(width * index / 24)
            length = 9 if index % 4 else 15
            cv2.line(chrome, (tick_x, height - margin), (tick_x, height - margin - length), THEME.line, 1)
            cv2.line(chrome, (tick_x, margin), (tick_x, margin + length), THEME.line, 1)
        chrome = (chrome.astype(np.float32) * 0.55).astype(np.uint8)
        return shade_map, chrome

    # -- main entry point --------------------------------------------------

    def grade_backdrop(self, frame: np.ndarray) -> None:
        """Apply the cinematic grade to the camera image.

        This runs before the artwork is composited so that paint keeps its full
        intensity — the strokes are a digital layer over the scene, not part of
        what the lens saw, and vignetting them would mute every edge of the canvas.
        """
        if frame.shape != self._shade.shape:
            return
        cv2.multiply(frame, self._shade, dst=frame, scale=1.0 / 255.0)
        cv2.add(frame, self._chrome, dst=frame)

    def render(self, frame: np.ndarray, model: HudModel, delta_time: float) -> list[HitRegion]:
        """Paint the complete interface and return this frame's hit regions."""
        regions: list[HitRegion] = []
        if model.show_skeleton:
            self._draw_skeleton(frame, model)

        chrome_alpha = self._dock_alpha
        chrome_alpha.set(0.22 if model.tool is Tool.DRAW else 1.0)
        alpha = chrome_alpha.update(delta_time)

        self._draw_titlebar(frame, model, regions, delta_time)
        self._draw_identity(frame, model)
        self._draw_mode_capsule(frame, model, delta_time)
        self._draw_brush_dock(frame, model, regions, alpha, delta_time)
        self._draw_color_rail(frame, model, regions, alpha, delta_time)
        self._draw_right_column(frame, model, regions, alpha, delta_time)
        self._draw_preview(frame, model, alpha)
        self._draw_telemetry(frame, model, alpha)
        self._draw_radial_menu(frame, model, regions, delta_time)
        self._draw_hold_indicator(frame, model)
        self._draw_reticle(frame, model, delta_time)
        self._draw_toasts(frame)
        if model.boot_progress < 1.0:
            self._draw_boot(frame, model)
        return regions

    # -- window chrome -----------------------------------------------------

    def _draw_titlebar(self, frame: np.ndarray, model: HudModel, regions: list[HitRegion], delta_time: float) -> None:
        width = self._width
        buttons = (
            (Command.MINIMISE, "–", THEME.text_mid),
            (Command.FULLSCREEN, "⬚", THEME.text_mid),
            (Command.CLOSE, "✕", THEME.danger),
        )
        size = 30
        gap = 8
        total = len(buttons) * (size + gap)
        start_x = width - total - 14
        regions.append(HitRegion(Command.DRAG_WINDOW, 0, (0, 0, start_x - 8, self.TITLE_HEIGHT), dwellable=False))
        for index, (command, glyph, color) in enumerate(buttons):
            x = start_x + index * (size + gap)
            y = (self.TITLE_HEIGHT - size) // 2
            rect = (x, y, x + size, y + size)
            hot = model.mouse is not None and _inside(rect, model.mouse)
            intensity = self._anim.drive(("title", command), 1.0 if hot else 0.0, delta_time)
            if intensity > 0.01:
                draw_glass_panel(frame, rect, 9, THEME.glass_light, 0.55 * intensity, color, 0.6 * intensity)
            TEXT.draw(
                frame,
                glyph,
                ((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2),
                size=15,
                color=mix_bgr(THEME.text_mid, color, intensity),
                anchor="mm",
                opacity=0.75 + 0.25 * intensity,
            )
            regions.append(HitRegion(command, 0, rect, dwellable=False))

    def _draw_identity(self, frame: np.ndarray, model: HudModel) -> None:
        left = self.TELEMETRY_X
        pulse = 0.72 + 0.28 * ease_in_out_sine((math.sin(model.elapsed * 1.6) + 1.0) / 2.0)
        TEXT.draw(frame, "AIVP", (left, 12), size=30, color=THEME.text_hi, tracking=4, glow=0.45 * pulse)
        TEXT.draw(
            frame,
            "AIR CANVAS ENGINE",
            (left + 2, 44),
            role=FontRole.MONO,
            size=10,
            color=THEME.primary,
            tracking=4,
            opacity=0.8,
        )
        draw_line_aa(frame, (left, 60), (left + 170, 60), THEME.line, 1, 0.8)

    def _draw_mode_capsule(self, frame: np.ndarray, model: HudModel, delta_time: float) -> None:
        tone = {
            Tool.DRAW: PALETTE[model.settings.color_index].preview_color(),
            Tool.SELECT: THEME.primary,
            Tool.ERASE: THEME.amber,
            Tool.IDLE: THEME.text_dim,
        }[model.tool]
        self._mode_color.set(tone)
        color = self._mode_color.update(delta_time)
        label = model.tool.value if model.gesture.present else "NO HAND"
        detail = {
            Tool.DRAW: "INDEX UP · MIDDLE DOWN",
            Tool.SELECT: "RADIAL PALETTE ACTIVE",
            Tool.ERASE: "THREE FINGERS · ERASING",
            Tool.IDLE: "SHOW YOUR HAND TO BEGIN",
        }[model.tool]
        if not model.gesture.present:
            detail = "SEARCHING FOR A HAND"
        text_width = TEXT.measure(label, FontRole.DISPLAY, 19, 3)[0]
        detail_width = TEXT.measure(detail, FontRole.MONO, 10, 2)[0]
        panel_w = int(self._anim.drive("capsule", max(text_width, detail_width) + 64, delta_time))
        centre_x = self._width // 2
        rect = (centre_x - panel_w // 2, 14, centre_x + panel_w // 2, 74)
        draw_glass_panel(frame, rect, 18, THEME.glass, 0.55, color, 0.42)
        draw_soft_glow(frame, (centre_x, 44), 90, color, 0.10)
        TEXT.draw(frame, label, (centre_x, 30), size=19, color=color, tracking=3, anchor="mt", glow=0.5)
        TEXT.draw(
            frame,
            detail,
            (centre_x, 58),
            role=FontRole.MONO,
            size=10,
            color=THEME.text_mid,
            tracking=2,
            anchor="mt",
        )

    # -- docks -------------------------------------------------------------

    def _swatch(
        self, brush: BrushId, color_index: int, active: bool, size: tuple[int, int] | None = None
    ) -> np.ndarray:
        """Return a cached mini-stroke painted with the real engine."""
        extent = size or self.DOCK_SWATCH
        key = (brush, color_index, active, extent)
        swatch = self._swatch_cache.get(key)
        if swatch is None:
            swatch = render_brush_swatch(
                brush,
                color_index if active else NEUTRAL_COLOR_INDEX,
                10 if extent == self.DOCK_SWATCH else 8,
                1.0 if active else 0.75,
                extent,
            )
            self._swatch_cache[key] = swatch
        return swatch

    def _draw_brush_dock(
        self,
        frame: np.ndarray,
        model: HudModel,
        regions: list[HitRegion],
        alpha: float,
        delta_time: float,
    ) -> None:
        count = len(BrushId)
        cell = self.DOCK_CELL
        total = count * cell
        top = (self._height - total) // 2
        panel = (self.DOCK_X - 8, top - 12, self.DOCK_X + 58, top + total + 12)
        draw_glass_panel(frame, panel, 22, THEME.glass, 0.5 * alpha, THEME.line, 0.55 * alpha)
        TEXT.draw(
            frame,
            "BRUSH",
            (self.DOCK_X + 25, top - 26),
            role=FontRole.MONO,
            size=9,
            color=THEME.text_dim,
            tracking=3,
            anchor="mm",
            opacity=alpha,
        )
        for index, brush in enumerate(BrushId):
            y = top + index * cell
            rect = (self.DOCK_X - 2, y + 2, self.DOCK_X + 50, y + cell - 2)
            region = HitRegion(Command.SELECT_BRUSH, int(brush), rect, BRUSH_SPECS[brush].label)
            regions.append(region)
            active = model.settings.brush is brush
            hot = _is_hot(region, model)
            level = self._anim.drive(("brush", brush), 1.0 if active else (0.55 if hot else 0.0), delta_time)
            if level > 0.01:
                draw_glass_panel(
                    frame, rect, 12, THEME.glass_light, 0.5 * level * alpha, THEME.primary, 0.5 * level * alpha
                )
            swatch = self._swatch(brush, model.settings.color_index, active)
            blit_layer(frame, swatch, rect[0] + 3, rect[1] + (rect[3] - rect[1] - swatch.shape[0]) // 2, alpha)
            if hot and level > 0.25:  # the active brush already has the preview panel
                self._draw_tooltip(
                    frame,
                    BRUSH_SPECS[brush].label,
                    BRUSH_SPECS[brush].caption,
                    (rect[2] + 14, (rect[1] + rect[3]) // 2),
                    level * alpha,
                )

    def _draw_tooltip(
        self, frame: np.ndarray, title: str, detail: str, anchor: tuple[int, int], opacity: float
    ) -> None:
        title_w = TEXT.measure(title, FontRole.DISPLAY, 13, 2)[0]
        detail_w = TEXT.measure(detail, FontRole.MONO, 9, 1)[0] if detail else 0
        panel_w = max(title_w, detail_w) + 28
        offset = int(lerp(-18.0, 0.0, ease_out_cubic(opacity)))
        left = anchor[0] + offset
        rect = (left, anchor[1] - 21, left + panel_w, anchor[1] + 21)
        draw_glass_panel(frame, rect, 12, THEME.glass_light, 0.66 * opacity, THEME.primary, 0.35 * opacity)
        TEXT.draw(frame, title, (left + 14, anchor[1] - 12), size=13, color=THEME.text_hi, tracking=2, opacity=opacity)
        if detail:
            TEXT.draw(
                frame,
                detail,
                (left + 14, anchor[1] + 3),
                role=FontRole.MONO,
                size=9,
                color=THEME.text_mid,
                tracking=1,
                opacity=opacity * 0.9,
            )

    def _draw_color_rail(
        self,
        frame: np.ndarray,
        model: HudModel,
        regions: list[HitRegion],
        alpha: float,
        delta_time: float,
    ) -> None:
        chip = self.RAIL_CHIP
        step = chip * 2 + 12
        total = len(PALETTE) * step
        left = (self._width - total) // 2
        centre_y = self._height - 54
        panel = (left - 14, centre_y - chip - 16, left + total + 14, centre_y + chip + 16)
        draw_glass_panel(frame, panel, 26, THEME.glass, 0.5 * alpha, THEME.line, 0.5 * alpha)
        for index, slot in enumerate(PALETTE):
            centre_x = left + index * step + step // 2
            rect = (centre_x - chip - 3, centre_y - chip - 3, centre_x + chip + 3, centre_y + chip + 3)
            region = HitRegion(Command.SELECT_COLOR, index, rect, slot.name, radius=chip + 5)
            regions.append(region)
            active = model.settings.color_index == index
            hot = _is_hot(region, model)
            level = self._anim.drive(("color", index), 1.0 if active else (0.5 if hot else 0.0), delta_time)
            lift = int(level * 7)
            if level > 0.02:
                draw_soft_glow(frame, (centre_x, centre_y - lift), chip * 3, slot.preview_color(), 0.22 * level * alpha)
            blit_layer_centred(frame, self._chips[index], (centre_x, centre_y - lift), alpha)
            # Every chip keeps a faint rim so the near-black swatch stays legible
            # against the dark rail; the rim brightens on hover and selection.
            draw_ring(
                frame,
                (centre_x, centre_y - lift),
                chip + 5,
                THEME.text_hi if active else THEME.primary,
                1,
                (0.24 + 0.76 * level) * alpha,
            )
            if active:
                TEXT.draw(
                    frame,
                    slot.name,
                    (centre_x, centre_y + chip + 12),
                    role=FontRole.MONO,
                    size=9,
                    color=THEME.text_hi,
                    tracking=2,
                    anchor="mt",
                    opacity=alpha,
                )

    def _draw_right_column(
        self,
        frame: np.ndarray,
        model: HudModel,
        regions: list[HitRegion],
        alpha: float,
        delta_time: float,
    ) -> None:
        right = self._width - 26
        panel_left = right - 56
        top = 108
        panel = (panel_left - 10, top - 30, right + 10, top + 540)
        draw_glass_panel(frame, panel, 22, THEME.glass, 0.5 * alpha, THEME.line, 0.5 * alpha)
        TEXT.draw(
            frame,
            "SIZE",
            (panel_left + 23, top - 20),
            role=FontRole.MONO,
            size=9,
            color=THEME.text_dim,
            tracking=3,
            anchor="mm",
            opacity=alpha,
        )
        for index, thickness in enumerate(THICKNESSES):
            centre_y = top + 22 + index * 40
            rect = (panel_left, centre_y - 18, panel_left + 46, centre_y + 18)
            region = HitRegion(Command.SELECT_THICKNESS, index, rect, f"{thickness}PX")
            regions.append(region)
            active = model.settings.thickness_index == index
            hot = _is_hot(region, model)
            level = self._anim.drive(("size", index), 1.0 if active else (0.5 if hot else 0.0), delta_time)
            if level > 0.01:
                draw_glass_panel(
                    frame, rect, 12, THEME.glass_light, 0.5 * level * alpha, THEME.primary, 0.45 * level * alpha
                )
            dot = thickness * 0.32 + 3.0
            tint = mix_bgr(THEME.text_dim, PALETTE[model.settings.color_index].preview_color(), level)
            draw_filled_circle(frame, (panel_left + 17, centre_y), dot, tint, alpha)
            TEXT.draw(
                frame,
                str(thickness),
                (panel_left + 40, centre_y),
                role=FontRole.MONO,
                size=10,
                color=THEME.text_mid if not active else THEME.text_hi,
                anchor="rm",
                opacity=alpha,
            )

        slider_top = top + 272
        slider_rect = (panel_left + 8, slider_top, panel_left + 38, slider_top + 130)
        regions.append(HitRegion(Command.SET_OPACITY, 0, slider_rect, "OPACITY", dwellable=False, continuous=True))
        self._draw_opacity_slider(frame, slider_rect, model, alpha)

        toggles = (
            (Command.TOGGLE_SHAPE_AI, "AI", model.shape_ai),
            (Command.TOGGLE_SKELETON, "HAND", model.show_skeleton),
            (Command.TOGGLE_MIRROR_ART, "SOLO", model.art_only),
        )
        toggle_top = slider_top + 172
        for index, (command, glyph, enabled) in enumerate(toggles):
            centre_y = toggle_top + index * 34
            rect = (panel_left, centre_y - 14, panel_left + 46, centre_y + 14)
            region = HitRegion(command, 0, rect, glyph)
            regions.append(region)
            hot = _is_hot(region, model)
            level = self._anim.drive(("toggle", command), 1.0 if enabled else (0.45 if hot else 0.0), delta_time)
            color = mix_bgr(THEME.text_dim, THEME.mint, level)
            draw_rounded_outline(frame, rect, 10, color, (0.35 + 0.55 * level) * alpha)
            TEXT.draw(
                frame,
                glyph,
                ((rect[0] + rect[2]) // 2, centre_y),
                role=FontRole.MONO,
                size=10,
                color=color,
                tracking=2,
                anchor="mm",
                opacity=alpha,
            )

    def _draw_opacity_slider(self, frame: np.ndarray, rect: Rect, model: HudModel, alpha: float) -> None:
        x0, y0, x1, y1 = rect
        centre_x = (x0 + x1) // 2
        track_top, track_bottom = y0 + 10, y1 - 10
        draw_line_aa(frame, (centre_x, track_top), (centre_x, track_bottom), THEME.line, 3, 0.85 * alpha)
        value = clamp(model.settings.opacity, 0.0, 1.0)
        knob_y = int(lerp(track_bottom, track_top, value))
        tint = PALETTE[model.settings.color_index].preview_color()
        draw_line_aa(frame, (centre_x, track_bottom), (centre_x, knob_y), tint, 3, alpha)
        draw_soft_glow(frame, (centre_x, knob_y), 22, tint, 0.30 * alpha)
        draw_filled_circle(frame, (centre_x, knob_y), 7.0, THEME.text_hi, alpha)
        draw_ring(frame, (centre_x, knob_y), 10, tint, 1, alpha)
        TEXT.draw(
            frame,
            f"{int(value * 100)}%",
            (centre_x, y1 + 4),
            role=FontRole.MONO,
            size=10,
            color=THEME.text_mid,
            anchor="mt",
            tracking=1,
            opacity=alpha,
        )
        TEXT.draw(
            frame,
            "OPACITY",
            (centre_x, y0 - 12),
            role=FontRole.MONO,
            size=9,
            color=THEME.text_dim,
            tracking=3,
            anchor="mm",
            opacity=alpha,
        )

    # -- panels ------------------------------------------------------------

    def _preview_frame(self, settings: BrushSettings, elapsed: float) -> np.ndarray | None:
        """Return the current frame of the looping brush preview.

        Rasterising the swatch with the real painting engine is expensive for the
        heavier brushes, so the loop is built as a small filmstrip: at most one
        phase is rendered per displayed frame and the cycle is free thereafter.
        """
        key = (settings.brush, settings.color_index, settings.thickness, round(settings.opacity, 2))
        if key != self._preview_key:
            self._preview_key = key
            self._preview_strip = [None] * self.PREVIEW_PHASES
        index = int(elapsed * 8.0) % self.PREVIEW_PHASES
        if self._preview_strip[index] is None:
            self._preview_strip[index] = render_brush_swatch(
                settings.brush,
                settings.color_index,
                settings.thickness,
                settings.opacity,
                self.PREVIEW_SIZE,
                index / self.PREVIEW_PHASES * math.tau,
            )
        return self._preview_strip[index]

    def _draw_preview(self, frame: np.ndarray, model: HudModel, alpha: float) -> None:
        settings = model.settings
        preview = self._preview_frame(settings, model.elapsed)
        rect = (self.TELEMETRY_X, self._height - 196, self.TELEMETRY_X + 276, self._height - 96)
        draw_glass_panel(frame, rect, 20, THEME.glass, 0.52 * alpha, THEME.line, 0.5 * alpha)
        TEXT.draw(
            frame,
            BRUSH_SPECS[settings.brush].label,
            (rect[0] + 20, rect[1] + 12),
            size=15,
            color=THEME.text_hi,
            tracking=3,
            opacity=alpha,
        )
        TEXT.draw(
            frame,
            BRUSH_SPECS[settings.brush].caption.upper(),
            (rect[0] + 21, rect[1] + 32),
            role=FontRole.MONO,
            size=9,
            color=THEME.text_dim,
            tracking=1,
            opacity=alpha,
        )
        if preview is not None:
            blit_layer(frame, preview, rect[0] + 22, rect[1] + 44, alpha)

    def _draw_telemetry(self, frame: np.ndarray, model: HudModel, alpha: float) -> None:
        lines = (
            f"FPS {model.fps:5.1f}",
            f"HAND {model.inference_ms:5.1f}MS",
            f"STROKES {model.stroke_count:04d}",
            f"UNDO {'READY' if model.can_undo else '-----'}",
            f"REDO {'READY' if model.can_redo else '-----'}",
        )
        for index, line in enumerate(lines):
            TEXT.draw(
                frame,
                line,
                (self.TELEMETRY_X, 74 + index * 15),
                role=FontRole.MONO,
                size=11,
                color=THEME.text_mid if index else THEME.primary,
                tracking=1,
                opacity=0.85,
            )
        footer = f"{model.export_format.label}  ·  {model.output_dir.name.upper()}"
        TEXT.draw(
            frame,
            footer,
            (self._width - 96, self._height - 26),
            role=FontRole.MONO,
            size=10,
            color=THEME.text_dim,
            tracking=2,
            anchor="rm",
            opacity=alpha,
        )

    # -- hand visualisation ------------------------------------------------

    def _draw_skeleton(self, frame: np.ndarray, model: HudModel) -> None:
        landmarks = model.gesture.landmarks
        if landmarks is None:
            return
        rect = bounding_rect(((float(p[0]), float(p[1])) for p in landmarks), padding=8)
        if rect is None:
            return
        batch = MaskBatch(frame, rect)
        if not batch.active:
            return
        for start, end in HAND_EDGES:
            batch.line(
                (float(landmarks[start][0]), float(landmarks[start][1])),
                (float(landmarks[end][0]), float(landmarks[end][1])),
                0.32,
            )
        for index, point in enumerate(landmarks):
            batch.circle((float(point[0]), float(point[1])), 3.4 if index in FINGER_TIPS else 2.0, 0.55)
        batch.flush(THEME.primary if model.tool is not Tool.ERASE else THEME.amber)

    def _draw_hold_indicator(self, frame: np.ndarray, model: HudModel) -> None:
        gesture = model.gesture
        if gesture.hold_progress <= 0.001 or gesture.pointer is None:
            return
        clearing = gesture.hold_label.startswith("CLEAR")
        if not clearing and model.menu_open:
            return  # the palette hub already carries the save progress ring
        anchor = (self._width // 2, self._height // 2 - 40)
        color = THEME.danger if clearing else THEME.mint
        radius = 66
        draw_ring(frame, anchor, radius, THEME.line, 2, 0.6)
        draw_arc(frame, anchor, radius, -90.0, -90.0 + 360.0 * gesture.hold_progress, color, 4, 0.95)
        draw_soft_glow(frame, anchor, radius + 30, color, 0.16 * gesture.hold_progress)
        TEXT.draw(
            frame,
            gesture.hold_label,
            (anchor[0], anchor[1] + radius + 16),
            size=15,
            color=color,
            tracking=3,
            anchor="mt",
            glow=0.4,
        )
        TEXT.draw(
            frame,
            f"{int(gesture.hold_progress * 100):02d}%",
            anchor,
            role=FontRole.MONO,
            size=17,
            color=THEME.text_hi,
            anchor="mm",
        )

    def _draw_reticle(self, frame: np.ndarray, model: HudModel, delta_time: float) -> None:
        gesture = model.gesture
        if gesture.pointer is None:
            self._pointer_trail.clear()
            return
        point = (int(gesture.pointer[0]), int(gesture.pointer[1]))
        self._pointer_trail.append(gesture.pointer)
        tint = {
            Tool.DRAW: PALETTE[model.settings.color_index].preview_color(),
            Tool.SELECT: THEME.primary,
            Tool.ERASE: THEME.amber,
            Tool.IDLE: THEME.text_mid,
        }[model.tool]
        trail_rect = bounding_rect(self._pointer_trail, padding=4)
        if trail_rect is not None:
            trail = MaskBatch(frame, trail_rect)
            for index in range(1, len(self._pointer_trail)):
                weight = index / len(self._pointer_trail)
                trail.line(self._pointer_trail[index - 1], self._pointer_trail[index], 0.32 * weight**2)
            trail.flush(tint)
        if model.tool is Tool.ERASE:
            radius = int(model.settings.thickness * model.settings.eraser_scale / 2.0)
            spin = model.elapsed * 90.0
            for segment in range(6):
                start = spin + segment * 60.0
                draw_arc(frame, point, max(8, radius), start, start + 34.0, tint, 2, 0.85)
            draw_filled_circle(frame, point, 2.5, tint, 0.9)
            return
        breathe = 1.0 + 0.10 * math.sin(model.elapsed * 4.4)
        if model.tool is Tool.DRAW:
            core = max(2.0, model.settings.thickness / 2.0)
            draw_soft_glow(frame, point, int(core * 5 + 18), tint, 0.28)
            draw_filled_circle(frame, point, core, tint, 0.95)
            draw_ring(frame, point, int(core + 9 * breathe), tint, 1, 0.75)
        else:
            span = int(16 * breathe)
            draw_ring(frame, point, span, tint, 1, 0.85)
            draw_ring(frame, point, int(span * 0.42), tint, 1, 0.55)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                draw_line_aa(
                    frame,
                    (point[0] + dx * (span + 5), point[1] + dy * (span + 5)),
                    (point[0] + dx * (span + 13), point[1] + dy * (span + 13)),
                    tint,
                    1,
                    0.7,
                )
            draw_filled_circle(frame, point, 1.8, THEME.text_hi, 0.9)

    # -- radial palette ----------------------------------------------------

    def _draw_radial_menu(
        self, frame: np.ndarray, model: HudModel, regions: list[HitRegion], delta_time: float
    ) -> None:
        self._menu_scale.set(1.0 if model.menu_open else 0.0)
        scale = self._menu_scale.update(delta_time)
        if scale <= 0.02:
            return
        eased = ease_out_back(scale) if model.menu_open else scale
        origin = model.menu_origin
        opacity = clamp(scale * 1.15, 0.0, 1.0)
        spin = model.elapsed * 12.0

        draw_soft_glow(frame, origin, int(self.RADIUS_COLOR * eased), THEME.primary, 0.05 * opacity)
        span = int(self.RADIUS_COLOR * eased) + 22
        graticule = MaskBatch(frame, (origin[0] - span, origin[1] - span, origin[0] + span, origin[1] + span))
        for radius in (self.RADIUS_THICKNESS, self.RADIUS_BRUSH, self.RADIUS_COLOR):
            graticule.circle(origin, max(2, int(radius * eased)), 0.55 * opacity, thickness=1)
        for tick in range(24):  # slowly rotating outer graticule
            angle = math.radians(spin + tick * 15.0)
            inner = self.RADIUS_COLOR * eased + 6
            outer = inner + (10 if tick % 3 == 0 else 5)
            graticule.line(
                (origin[0] + math.cos(angle) * inner, origin[1] + math.sin(angle) * inner),
                (origin[0] + math.cos(angle) * outer, origin[1] + math.sin(angle) * outer),
                0.75 * opacity,
            )
        graticule.flush(THEME.line)
        draw_ring(frame, origin, max(2, int(self.RADIUS_HUB * eased)), THEME.primary, 1, 0.55 * opacity)

        self._draw_ring_items(
            frame,
            model,
            regions,
            origin,
            self.RADIUS_THICKNESS * eased,
            len(THICKNESSES),
            Command.SELECT_THICKNESS,
            18,
            opacity,
            delta_time,
            -90.0,
        )
        self._draw_ring_items(
            frame,
            model,
            regions,
            origin,
            self.RADIUS_BRUSH * eased,
            len(BrushId),
            Command.SELECT_BRUSH,
            23,
            opacity,
            delta_time,
            -90.0,
        )
        self._draw_ring_items(
            frame,
            model,
            regions,
            origin,
            self.RADIUS_COLOR * eased,
            len(PALETTE),
            Command.SELECT_COLOR,
            20,
            opacity,
            delta_time,
            -90.0,
        )
        self._draw_menu_hub(frame, model, origin, eased, opacity)

    def _draw_ring_items(
        self,
        frame: np.ndarray,
        model: HudModel,
        regions: list[HitRegion],
        origin: tuple[int, int],
        radius: float,
        count: int,
        command: Command,
        item_radius: int,
        opacity: float,
        delta_time: float,
        start_deg: float,
    ) -> None:
        """Lay ``count`` selectable items evenly around one ring of the palette."""
        if radius < 8.0:
            return
        settings = model.settings
        selected = {
            Command.SELECT_THICKNESS: settings.thickness_index,
            Command.SELECT_BRUSH: int(settings.brush),
            Command.SELECT_COLOR: settings.color_index,
        }[command]
        for index in range(count):
            angle = math.radians(start_deg + index * 360.0 / count)
            centre = (
                int(origin[0] + math.cos(angle) * radius),
                int(origin[1] + math.sin(angle) * radius),
            )
            rect = (
                centre[0] - item_radius,
                centre[1] - item_radius,
                centre[0] + item_radius,
                centre[1] + item_radius,
            )
            region = HitRegion(command, index, rect, "", radius=float(item_radius) + 4.0)
            regions.append(region)
            active = index == selected
            hot = model.hover_key == region.key
            level = self._anim.drive(("menu", command, index), 1.0 if active else (0.7 if hot else 0.0), delta_time)
            self._draw_menu_item(frame, model, command, index, centre, item_radius, level, opacity, active)
            if hot and model.dwell_progress > 0.01:
                draw_arc(
                    frame,
                    centre,
                    item_radius + 7,
                    -90.0,
                    -90.0 + 360.0 * model.dwell_progress,
                    THEME.mint,
                    3,
                    opacity,
                )

    def _draw_menu_item(
        self,
        frame: np.ndarray,
        model: HudModel,
        command: Command,
        index: int,
        centre: tuple[int, int],
        item_radius: int,
        level: float,
        opacity: float,
        active: bool,
    ) -> None:
        scale = 1.0 + 0.22 * level
        if command is Command.SELECT_COLOR:
            slot = PALETTE[index]
            if level > 0.02:
                draw_soft_glow(frame, centre, int(item_radius * 2.4), slot.preview_color(), 0.28 * level * opacity)
            blit_layer_centred(frame, self._menu_chips[index], centre, opacity)
            draw_ring(
                frame,
                centre,
                int((item_radius - 3) * scale),
                THEME.text_hi if active else THEME.line,
                1,
                (0.35 + 0.6 * level) * opacity,
            )
            return
        if command is Command.SELECT_THICKNESS:
            thickness = THICKNESSES[index]
            tint = mix_bgr(THEME.text_dim, PALETTE[model.settings.color_index].preview_color(), max(level, 0.25))
            draw_filled_circle(frame, centre, (thickness * 0.26 + 2.4) * scale, tint, opacity)
            draw_ring(frame, centre, int(item_radius * scale), THEME.line, 1, (0.3 + 0.5 * level) * opacity)
            return
        brush = BrushId(index)
        if level > 0.02:
            draw_soft_glow(frame, centre, int(item_radius * 2.2), THEME.primary, 0.22 * level * opacity)
        draw_glass_panel(
            frame,
            (centre[0] - item_radius, centre[1] - item_radius, centre[0] + item_radius, centre[1] + item_radius),
            item_radius,
            THEME.void,
            (0.55 + 0.25 * level) * opacity,
            THEME.primary if active else THEME.line,
            (0.3 + 0.6 * level) * opacity,
        )
        swatch = self._swatch(brush, model.settings.color_index, active or level > 0.4, self.MENU_SWATCH)
        blit_layer_centred(frame, swatch, centre, opacity)

    def _draw_menu_hub(
        self, frame: np.ndarray, model: HudModel, origin: tuple[int, int], eased: float, opacity: float
    ) -> None:
        radius = int(self.RADIUS_HUB * eased)
        if radius < 10:
            return
        draw_glass_panel(
            frame,
            (origin[0] - radius, origin[1] - radius, origin[0] + radius, origin[1] + radius),
            radius,
            THEME.glass,
            0.66 * opacity,
            THEME.primary,
            0.5 * opacity,
        )
        gesture = model.gesture
        if gesture.hold_progress > 0.001 and gesture.hold_label.startswith("SAVE"):
            draw_arc(frame, origin, radius - 5, -90.0, -90.0 + 360.0 * gesture.hold_progress, THEME.mint, 4, opacity)
        TEXT.draw(
            frame,
            "HOLD",
            (origin[0], origin[1] - 12),
            role=FontRole.MONO,
            size=10,
            color=THEME.text_mid,
            tracking=2,
            anchor="mm",
            opacity=opacity,
        )
        TEXT.draw(
            frame,
            "SAVE",
            (origin[0], origin[1] + 6),
            size=15,
            color=THEME.mint,
            tracking=3,
            anchor="mm",
            opacity=opacity,
            glow=0.35 * opacity,
        )

    # -- notifications & boot ---------------------------------------------

    def _draw_toasts(self, frame: np.ndarray) -> None:
        now = time.perf_counter()
        self._toasts = [toast for toast in self._toasts if not toast.expired(now)]
        base_y = 96
        for index, toast in enumerate(reversed(self._toasts)):
            envelope = toast.envelope(now)
            if envelope <= 0.01:
                continue
            title_w = TEXT.measure(toast.title, FontRole.DISPLAY, 14, 3)[0]
            detail_w = TEXT.measure(toast.detail, FontRole.MONO, 10, 1)[0] if toast.detail else 0
            panel_w = max(title_w, detail_w) + 54
            centre_x = self._width // 2
            offset = int(lerp(-16.0, 0.0, envelope))
            top = base_y + index * 62 + offset
            rect = (centre_x - panel_w // 2, top, centre_x + panel_w // 2, top + 52)
            draw_glass_panel(frame, rect, 16, THEME.glass_light, 0.7 * envelope, toast.kind.color, 0.55 * envelope)
            draw_line_aa(frame, (rect[0] + 14, top + 14), (rect[0] + 14, top + 38), toast.kind.color, 3, envelope)
            TEXT.draw(
                frame,
                toast.title,
                (rect[0] + 28, top + 12),
                size=14,
                color=THEME.text_hi,
                tracking=3,
                opacity=envelope,
                glow=0.3 * envelope,
            )
            if toast.detail:
                TEXT.draw(
                    frame,
                    toast.detail,
                    (rect[0] + 29, top + 31),
                    role=FontRole.MONO,
                    size=10,
                    color=THEME.text_mid,
                    tracking=1,
                    opacity=envelope * 0.9,
                )

    def _draw_boot(self, frame: np.ndarray, model: HudModel) -> None:
        """Paint the launch sequence: a veil, expanding rings and a title reveal."""
        progress = clamp(model.boot_progress, 0.0, 1.0)
        veil = (1.0 - ease_out_cubic(progress)) * 0.96
        if veil > 0.004 and self._veil.shape == frame.shape:
            cv2.addWeighted(self._veil, veil, frame, 1.0 - veil, 0.0, dst=frame)
        centre = (self._width // 2, self._height // 2)
        # While start-up is still running the rings keep sweeping so the wait reads
        # as a system coming online rather than as a stalled window.
        sweep = model.elapsed if progress < 1.0 else 0.0
        for ring in range(3):
            phase = ((sweep * 0.55 + ring * 0.33) % 1.0) if progress < self.BOOT_SETTLED else 1.0
            if phase >= 1.0:
                continue
            radius = int(lerp(40.0, 300.0, ease_out_cubic(phase)))
            draw_ring(frame, centre, radius, THEME.primary, 1, (1.0 - phase) * 0.7)
        fade = 1.0 - clamp((progress - self.BOOT_SETTLED) / (1.0 - self.BOOT_SETTLED), 0.0, 1.0)
        reveal = clamp(progress * 2.6, 0.0, 1.0)
        if reveal <= 0.01 or fade <= 0.01:
            return
        title_y = centre[1] - 18
        if self._logo is not None:
            title_y = centre[1] + 66
            lift = round(lerp(18.0, 0.0, ease_out_cubic(reveal)))
            logo_centre = (centre[0], centre[1] - 42 + lift)
            draw_soft_glow(frame, logo_centre, self.LOGO_SIZE, THEME.primary, 0.10 * reveal * fade)
            blit_layer_centred(frame, self._logo, logo_centre, reveal * fade)
        TEXT.draw(
            frame,
            "AI VIRTUAL PAINTER",
            (centre[0], title_y),
            size=40,
            color=THEME.text_hi,
            tracking=round(lerp(26.0, 8.0, ease_out_cubic(reveal))),
            anchor="mm",
            opacity=reveal * fade,
            glow=0.5,
        )
        TEXT.draw(
            frame,
            model.boot_status,
            (centre[0], title_y + 44),
            role=FontRole.MONO,
            size=12,
            color=THEME.primary,
            tracking=6,
            anchor="mm",
            opacity=reveal * fade,
        )


def _inside(rect: Rect, point: tuple[int, int]) -> bool:
    """Point-in-rectangle test used for mouse hover states."""
    return rect[0] <= point[0] < rect[2] and rect[1] <= point[1] < rect[3]


def _is_hot(region: HitRegion, model: HudModel) -> bool:
    """Whether a region is currently hovered by the mouse or the fingertip."""
    if model.hover_key == region.key:
        return True
    return model.mouse is not None and region.contains(*model.mouse)


# =============================================================================
# SECTION 15 — Application
# =============================================================================


def bgr_to_hex(color: Bgr) -> str:
    """Format a BGR triple as the ``#RRGGBB`` string Tk expects."""
    return f"#{color[2]:02x}{color[1]:02x}{color[0]:02x}"


class VirtualPainterApp:
    """Wires the camera, the tracker, the painting engine and the HUD together."""

    #: Seconds the fingertip must rest on a control before it activates.
    DWELL_SECONDS: Final = 0.55
    #: Refractory period after a dwell selection, so one rest fires once.
    DWELL_COOLDOWN: Final = 0.45
    #: Duration of the title reveal before the camera is ready.
    BOOT_SECONDS: Final = 1.9
    #: How far the boot sequence advances while start-up is still in progress.
    BOOT_HOLD: Final = 0.62
    #: Time taken to dissolve the boot veil once everything is online.
    BOOT_DISSOLVE: Final = 0.85
    #: Minimum fingertip travel (pixels) before a new stroke point is recorded.
    MIN_POINT_DISTANCE: Final = 1.4

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._width = config.canvas_width
        self._height = config.canvas_height
        self._document = PaintDocument(self._width, self._height)
        self._exporter = ArtworkExporter(config)
        self._hud = HudRenderer(self._width, self._height)
        self._gestures = GestureEngine()
        self._camera = CameraStream(config)
        self._tracker = HandTracker(config)

        self._settings = BrushSettings()
        self._shape_ai = True
        self._show_skeleton = True
        self._art_only = False
        self._export_format = ExportFormat.PNG

        self._compose = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        self._camera_frame: np.ndarray | None = None
        self._regions: list[HitRegion] = []
        self._hover_key: RegionKey | None = None
        self._dwell_started = 0.0
        self._dwell_blocked_until = 0.0
        self._menu_origin = (self._width // 2, self._height // 2)
        self._menu_latched = False
        self._active_tool = Tool.IDLE
        self._last_point: tuple[float, float] | None = None

        self._mouse: tuple[int, int] | None = None
        self._drag_offset: tuple[int, int] | None = None
        self._slider_drag = False
        self._display_scale = 1.0
        self._display_origin = (0, 0)

        self._frame_times: deque[float] = deque(maxlen=45)
        self._fps = 0.0
        self._last_tick = time.perf_counter()
        self._started_at = time.perf_counter()
        self._ready_at: float | None = None
        self._boot_status = "STARTING CAMERA"
        self._startup_error: BaseException | None = None
        self._closing = False
        self._fullscreen = False
        self._windowed_geometry = ""
        self._photo: ImageTk.PhotoImage | None = None
        self._photo_size = (0, 0)
        self._rgb_buffer: np.ndarray | None = None
        self._scale_buffer: np.ndarray | None = None
        self._icon: ImageTk.PhotoImage | None = None

        self._build_window()

    # -- window ------------------------------------------------------------

    def _build_window(self) -> None:
        ctk.set_appearance_mode("dark")
        ctk.set_widget_scaling(1.0)
        ctk.set_window_scaling(1.0)
        self._root = ctk.CTk()
        self._root.title("AI Virtual Painter")
        background = bgr_to_hex(THEME.void)
        self._root.configure(fg_color=background)
        screen_w = self._root.winfo_screenwidth()
        screen_h = self._root.winfo_screenheight()
        left = max(0, (screen_w - self._width) // 2)
        top = max(0, (screen_h - self._height) // 2 - 20)
        # The window is a pixel-exact video surface, so it is sized in *physical*
        # pixels. CustomTkinter multiplies requested geometry by the monitor's DPI
        # factor, which on a 125 % display would hand back a 1600x900 widget and
        # force every frame through a resample for no visual gain.
        request_w, request_h = self._physical_size(self._width, self._height)
        self._root.geometry(f"{request_w}x{request_h}+{left}+{top}")
        self._root.minsize(*self._physical_size(960, 540))
        # Frameless by design: the title bar, window buttons and every control are
        # painted inside the video frame so nothing breaks the holographic surface.
        self._root.overrideredirect(True)
        self._root.protocol("WM_DELETE_WINDOW", self.shutdown)

        self._shell = ctk.CTkFrame(self._root, fg_color=background, corner_radius=0, border_width=0)
        self._shell.pack(fill="both", expand=True)
        self._display = tk.Label(self._shell, bd=0, highlightthickness=0, bg=background, cursor="none")
        self._display.pack(fill="both", expand=True)

        self._apply_window_icon()
        self._bind_events()

    def _apply_window_icon(self) -> None:
        """Set the window icon when the optional icon file is present."""
        path = find_app_icon()
        if path is None:
            return
        try:
            self._icon = ImageTk.PhotoImage(Image.open(path).convert("RGBA").resize((64, 64)))
            self._root.iconphoto(True, self._icon)
        except (OSError, tk.TclError, ValueError) as error:
            LOGGER.warning("Could not apply the window icon: %s", error)

    def _physical_size(self, width: int, height: int) -> tuple[int, int]:
        """Convert a physical pixel size into the value CustomTkinter expects."""
        try:
            scaling = float(ctk.ScalingTracker.get_window_scaling(self._root))
        except (AttributeError, KeyError, TypeError, ValueError):  # pragma: no cover - version guard
            scaling = 1.0
        if scaling <= 0.01:
            scaling = 1.0
        return (max(1, round(width / scaling)), max(1, round(height / scaling)))

    def _bind_events(self) -> None:
        display = self._display
        display.bind("<Motion>", self._on_mouse_move)
        display.bind("<Leave>", self._on_mouse_leave)
        display.bind("<Button-1>", self._on_mouse_down)
        display.bind("<B1-Motion>", self._on_mouse_drag)
        display.bind("<ButtonRelease-1>", self._on_mouse_up)
        display.bind("<MouseWheel>", self._on_wheel)

        root = self._root
        root.bind("<Map>", self._on_map)
        for sequence, handler in (
            ("<Control-z>", lambda _event: self._undo()),
            ("<Control-y>", lambda _event: self._redo()),
            ("<Control-Shift-Z>", lambda _event: self._redo()),
            ("<Control-s>", lambda _event: self._save(ask=True)),
            ("<Control-Shift-S>", lambda _event: self._save(ask=False)),
            ("<Control-Delete>", lambda _event: self._clear()),
            ("<F11>", lambda _event: self._toggle_fullscreen()),
            ("<Escape>", lambda _event: self._on_escape()),
            ("<Tab>", lambda _event: self._cycle_export()),
            ("<Left>", lambda _event: self._step_brush(-1)),
            ("<Right>", lambda _event: self._step_brush(1)),
            ("<Up>", lambda _event: self._step_color(-1)),
            ("<Down>", lambda _event: self._step_color(1)),
            ("<bracketleft>", lambda _event: self._step_opacity(-0.05)),
            ("<bracketright>", lambda _event: self._step_opacity(0.05)),
            ("<a>", lambda _event: self._dispatch(Command.TOGGLE_SHAPE_AI, 0)),
            ("<h>", lambda _event: self._dispatch(Command.TOGGLE_SKELETON, 0)),
            ("<v>", lambda _event: self._dispatch(Command.TOGGLE_MIRROR_ART, 0)),
        ):
            root.bind(sequence, handler)
        for index in range(len(THICKNESSES)):
            root.bind(str(index + 1), lambda _event, slot=index: self._dispatch(Command.SELECT_THICKNESS, slot))
        root.focus_force()

    def _on_map(self, _event: tk.Event) -> None:
        """Re-assert the frameless style after the window is restored."""
        if not self._closing and not self._root.overrideredirect():
            self._root.after(10, lambda: self._root.overrideredirect(True))

    # -- lifecycle ---------------------------------------------------------

    def run(self) -> int:
        """Start background services, enter the main loop and return an exit code."""
        threading.Thread(target=self._startup, name="aivp-startup", daemon=True).start()
        self._root.after(16, self._tick)
        self._root.mainloop()
        if self._startup_error is not None:
            return 1
        return 0

    def _startup(self) -> None:
        """Bring the camera and the landmarker online off the UI thread."""
        try:
            self._camera.start()
            self._boot_status = "LOADING HAND MODEL"
            self._tracker.start()
        except (CameraError, ModelError, RuntimeError, OSError) as error:
            LOGGER.error("Start-up failed: %s", error)
            self._startup_error = error
            return
        self._boot_status = "TRACKING ONLINE"
        self._ready_at = time.perf_counter()
        self._hud.notify("SYSTEM ONLINE", "INDEX UP TO PAINT", ToastKind.SUCCESS)

    def _boot_progress(self, now: float) -> float:
        """Advance the boot reveal, holding it open until start-up completes.

        Provisioning the landmark model can take several seconds on a first run,
        so the sequence plays from launch and parks just short of the dissolve
        rather than leaving the window black while the model loads.
        """
        opening = clamp((now - self._started_at) / self.BOOT_SECONDS, 0.0, 1.0) * self.BOOT_HOLD
        if self._ready_at is None:
            return opening
        dissolve = clamp((now - self._ready_at) / self.BOOT_DISSOLVE, 0.0, 1.0)
        return max(opening, self.BOOT_HOLD + (1.0 - self.BOOT_HOLD) * dissolve)

    def shutdown(self) -> None:
        """Tear down every resource and close the window."""
        if self._closing:
            return
        self._closing = True
        LOGGER.info("Shutting down")
        self._tracker.stop()
        self._camera.stop()
        with contextlib.suppress(tk.TclError):
            self._root.destroy()

    # -- main loop ---------------------------------------------------------

    def _tick(self) -> None:
        if self._closing:
            return
        if self._startup_error is not None:
            self._report_fatal(self._startup_error)
            return
        started = time.perf_counter()
        delta_time = clamp(started - self._last_tick, 1e-4, 0.1)
        self._last_tick = started
        self._frame_times.append(delta_time)
        if self._frame_times:
            self._fps = len(self._frame_times) / max(sum(self._frame_times), 1e-6)

        self._acquire_frame()
        gesture = self._update_tracking(started)
        boot = self._boot_progress(started)
        if boot >= 1.0:
            self._apply_gesture_events(gesture)
            self._update_painting(gesture)
            self._update_dwell(gesture, started)
        else:
            self._document.cancel_stroke()

        self._compose_frame()
        model = self._build_model(gesture, started, boot)
        self._regions = self._hud.render(self._compose, model, delta_time)
        self._present()

        budget = 1.0 / self._config.target_fps
        elapsed = time.perf_counter() - started
        self._root.after(max(1, int((budget - elapsed) * 1000.0)), self._tick)

    def _acquire_frame(self) -> None:
        frame = self._camera.latest()
        if frame is None:
            self._compose[:] = THEME.void
            return
        self._camera_frame = frame
        self._tracker.submit(frame)

    def _update_tracking(self, now: float) -> GestureState:
        return self._gestures.update(self._tracker.latest(), now)

    def _compose_frame(self) -> None:
        frame = self._camera_frame
        if self._art_only or frame is None:
            self._compose[:] = THEME.void
        elif frame.shape[0] == self._height and frame.shape[1] == self._width:
            np.copyto(self._compose, frame)
        else:
            cv2.resize(frame, (self._width, self._height), dst=self._compose, interpolation=cv2.INTER_LINEAR)
        self._hud.grade_backdrop(self._compose)
        # Only the painted region is composited. An empty canvas costs nothing,
        # and a half-full one costs proportionally less than a full-frame blend.
        artwork = self._document.dirty_rect
        if artwork is not None:
            flatten_onto(self._compose, self._document.canvas, artwork)

    def _present(self) -> None:
        """Push the composed frame to the Tk widget.

        Both scratch buffers are reused across frames; the RGB conversion and the
        letterbox resize together cost less than a millisecond that way, leaving
        Tk's own photo upload as the only meaningful expense.
        """
        width = max(1, self._display.winfo_width())
        height = max(1, self._display.winfo_height())
        if width < 10 or height < 10:
            width, height = self._width, self._height
        scale = min(width / self._width, height / self._height)
        target = (max(1, int(self._width * scale)), max(1, int(self._height * scale)))
        self._display_scale = scale
        self._display_origin = ((width - target[0]) // 2, (height - target[1]) // 2)
        if self._rgb_buffer is None or self._rgb_buffer.shape[:2] != (target[1], target[0]):
            self._rgb_buffer = np.empty((target[1], target[0], 3), dtype=np.uint8)
            self._scale_buffer = (
                None if target == (self._width, self._height) else np.empty((target[1], target[0], 3), dtype=np.uint8)
            )
        source: np.ndarray = self._compose
        if self._scale_buffer is not None:
            cv2.resize(self._compose, target, dst=self._scale_buffer, interpolation=cv2.INTER_LINEAR)
            source = self._scale_buffer
        cv2.cvtColor(source, cv2.COLOR_BGR2RGB, dst=self._rgb_buffer)
        image = Image.frombuffer("RGB", target, self._rgb_buffer, "raw", "RGB", 0, 1)
        if self._photo is None or self._photo_size != target:
            self._photo = ImageTk.PhotoImage(image)
            self._photo_size = target
            self._display.configure(image=self._photo)
        else:
            self._photo.paste(image)

    def _report_fatal(self, error: BaseException) -> None:
        self._closing = True
        LOGGER.error("Fatal: %s", error)
        try:
            self._root.overrideredirect(False)
            self._root.withdraw()
            messagebox.showerror("AI Virtual Painter", str(error))
        finally:
            self._camera.stop()
            self._tracker.stop()
            with contextlib.suppress(tk.TclError):
                self._root.destroy()

    # -- interaction -------------------------------------------------------

    def _build_model(self, gesture: GestureState, now: float, boot: float) -> HudModel:
        menu_open = gesture.tool is Tool.SELECT and gesture.pointer is not None
        if menu_open and not self._menu_latched:
            self._menu_origin = self._clamp_menu_origin(gesture.pointer)  # type: ignore[arg-type]
            self._menu_latched = True
        elif not menu_open:
            self._menu_latched = False
        dwell = 0.0
        if self._hover_key is not None and self._dwell_started > 0.0:
            dwell = clamp((now - self._dwell_started) / self.DWELL_SECONDS, 0.0, 1.0)
        return HudModel(
            settings=self._settings,
            gesture=gesture,
            tool=gesture.tool,
            fps=self._fps,
            inference_ms=self._tracker.inference_ms,
            stroke_count=self._document.stroke_count,
            can_undo=self._document.can_undo,
            can_redo=self._document.can_redo,
            shape_ai=self._shape_ai,
            show_skeleton=self._show_skeleton,
            art_only=self._art_only,
            export_format=self._export_format,
            output_dir=self._exporter.output_dir,
            hover_key=self._hover_key,
            dwell_progress=dwell,
            menu_open=menu_open,
            menu_origin=self._menu_origin,
            mouse=self._mouse,
            boot_progress=boot,
            boot_status=self._boot_status,
            elapsed=now - self._started_at,
        )

    def _clamp_menu_origin(self, pointer: tuple[float, float]) -> tuple[int, int]:
        margin = HudRenderer.RADIUS_COLOR + 28
        return (
            int(clamp(pointer[0], margin, self._width - margin)),
            int(clamp(pointer[1], margin, self._height - margin)),
        )

    def _update_painting(self, gesture: GestureState) -> None:
        tool = gesture.tool
        pointer = gesture.pointer
        painting = tool in (Tool.DRAW, Tool.ERASE) and pointer is not None
        if not painting:
            if self._document.is_drawing:
                self._finish_stroke()
            self._active_tool = tool
            self._last_point = None
            return
        assert pointer is not None
        if not self._document.is_drawing or self._active_tool is not tool:
            if self._document.is_drawing:
                self._finish_stroke()
            self._document.begin_stroke(self._settings, eraser=tool is Tool.ERASE)
            self._last_point = None
        self._active_tool = tool
        if self._last_point is None or math.dist(pointer, self._last_point) >= self.MIN_POINT_DISTANCE:
            self._document.extend_stroke(pointer[0], pointer[1], gesture.velocity)
            self._last_point = pointer

    def _finish_stroke(self) -> None:
        shape = self._document.end_stroke(snap_shapes=self._shape_ai)
        self._last_point = None
        if shape is not None:
            self._hud.notify("SHAPE SNAPPED", shape, ToastKind.ACCENT)

    def _apply_gesture_events(self, gesture: GestureState) -> None:
        for event in gesture.events:
            if event is GestureEvent.UNDO:
                self._undo()
            elif event is GestureEvent.REDO:
                self._redo()
            elif event is GestureEvent.CLEAR:
                self._clear()
            elif event is GestureEvent.SAVE:
                self._save(ask=False)

    def _update_dwell(self, gesture: GestureState, now: float) -> None:
        pointer = gesture.pointer
        if gesture.tool is not Tool.SELECT or pointer is None:
            self._hover_key = None
            self._dwell_started = 0.0
            return
        region = self._region_at(pointer[0], pointer[1], dwellable_only=True)
        if region is None:
            self._hover_key = None
            self._dwell_started = 0.0
            return
        if region.key != self._hover_key:
            self._hover_key = region.key
            self._dwell_started = now
            return
        if now < self._dwell_blocked_until:
            self._dwell_started = now
            return
        if now - self._dwell_started >= self.DWELL_SECONDS:
            self._dispatch(region.command, region.payload)
            self._dwell_started = now
            self._dwell_blocked_until = now + self.DWELL_COOLDOWN

    def _region_at(self, x: float, y: float, dwellable_only: bool) -> HitRegion | None:
        """Return the topmost region under a point (later regions draw on top)."""
        for region in reversed(self._regions):
            if dwellable_only and not region.dwellable:
                continue
            if region.contains(x, y):
                return region
        return None

    # -- commands ----------------------------------------------------------

    def _dispatch(self, command: Command, payload: int, position: tuple[int, int] | None = None) -> None:
        """Execute an interface command."""
        if command is Command.SELECT_BRUSH:
            brush = BrushId(payload)
            if brush is not self._settings.brush:
                self._settings = replace(self._settings, brush=brush)
                self._hud.notify("BRUSH", BRUSH_SPECS[brush].label, ToastKind.INFO)
        elif command is Command.SELECT_COLOR:
            if payload != self._settings.color_index:
                self._settings = replace(self._settings, color_index=payload % len(PALETTE))
                self._hud.notify("COLOUR", PALETTE[self._settings.color_index].name, ToastKind.INFO)
        elif command is Command.SELECT_THICKNESS:
            if payload != self._settings.thickness_index:
                self._settings = replace(self._settings, thickness_index=payload % len(THICKNESSES))
                self._hud.notify("SIZE", f"{self._settings.thickness} PX", ToastKind.INFO)
        elif command is Command.SET_OPACITY and position is not None:
            self._apply_opacity_from(position)
        elif command is Command.UNDO:
            self._undo()
        elif command is Command.REDO:
            self._redo()
        elif command is Command.CLEAR:
            self._clear()
        elif command is Command.SAVE:
            self._save(ask=False)
        elif command is Command.CYCLE_EXPORT:
            self._cycle_export()
        elif command is Command.TOGGLE_SHAPE_AI:
            self._shape_ai = not self._shape_ai
            self._hud.notify("SHAPE AI", "ENABLED" if self._shape_ai else "DISABLED", ToastKind.INFO)
        elif command is Command.TOGGLE_SKELETON:
            self._show_skeleton = not self._show_skeleton
            self._hud.notify("HAND WIREFRAME", "ON" if self._show_skeleton else "OFF", ToastKind.INFO)
        elif command is Command.TOGGLE_MIRROR_ART:
            self._art_only = not self._art_only
            self._hud.notify("CANVAS", "ARTWORK ONLY" if self._art_only else "CAMERA BLEND", ToastKind.INFO)
        elif command is Command.MINIMISE:
            self._minimise()
        elif command is Command.FULLSCREEN:
            self._toggle_fullscreen()
        elif command is Command.CLOSE:
            self.shutdown()

    def _undo(self) -> None:
        self._document.cancel_stroke()
        if self._document.undo():
            self._hud.notify("UNDO", f"{self._document.stroke_count} STROKES", ToastKind.INFO)
        else:
            self._hud.notify("NOTHING TO UNDO", "", ToastKind.WARNING)

    def _redo(self) -> None:
        self._document.cancel_stroke()
        if self._document.redo():
            self._hud.notify("REDO", f"{self._document.stroke_count} STROKES", ToastKind.INFO)
        else:
            self._hud.notify("NOTHING TO REDO", "", ToastKind.WARNING)

    def _clear(self) -> None:
        if self._document.clear():
            self._hud.notify("CANVAS CLEARED", "UNDO RESTORES IT", ToastKind.WARNING)
        else:
            self._hud.notify("CANVAS ALREADY EMPTY", "", ToastKind.WARNING)

    def _cycle_export(self) -> None:
        formats = list(ExportFormat)
        self._export_format = formats[(formats.index(self._export_format) + 1) % len(formats)]
        self._hud.notify("EXPORT FORMAT", self._export_format.label, ToastKind.INFO)

    def _step_brush(self, direction: int) -> None:
        self._dispatch(Command.SELECT_BRUSH, (int(self._settings.brush) + direction) % len(BrushId))

    def _step_color(self, direction: int) -> None:
        self._dispatch(Command.SELECT_COLOR, (self._settings.color_index + direction) % len(PALETTE))

    def _step_opacity(self, delta: float) -> None:
        value = clamp(self._settings.opacity + delta, 0.05, 1.0)
        self._settings = replace(self._settings, opacity=value)
        self._hud.notify("OPACITY", f"{int(value * 100)}%", ToastKind.INFO)

    def _apply_opacity_from(self, position: tuple[int, int]) -> None:
        region = next((item for item in self._regions if item.command is Command.SET_OPACITY), None)
        if region is None:
            return
        top, bottom = region.rect[1] + 10, region.rect[3] - 10
        value = clamp((bottom - position[1]) / max(bottom - top, 1), 0.05, 1.0)
        self._settings = replace(self._settings, opacity=value)

    def _save(self, ask: bool) -> None:
        if self._document.is_empty:
            self._hud.notify("NOTHING TO EXPORT", "PAINT SOMETHING FIRST", ToastKind.WARNING)
            return
        destination: Path | None = None
        if ask:
            chosen = filedialog.asksaveasfilename(
                parent=self._root,
                title="Export artwork",
                defaultextension=self._export_format.suffix,
                initialdir=str(self._exporter.output_dir),
                initialfile=f"aivp-{time.strftime('%Y%m%d-%H%M%S')}{self._export_format.suffix}",
                filetypes=[("PNG image", "*.png"), ("JPEG image", "*.jpg")],
            )
            if not chosen:
                return
            destination = Path(chosen)
        started = time.perf_counter()
        try:
            written = self._exporter.save(self._document, self._export_format, self._camera_frame, destination)
        except (OSError, ValueError) as error:
            LOGGER.error("Export failed: %s", error)
            self._hud.notify("EXPORT FAILED", str(error)[:48], ToastKind.WARNING)
            return
        elapsed = (time.perf_counter() - started) * 1000.0
        LOGGER.info("Export completed in %.0f ms", elapsed)
        self._hud.notify("ARTWORK SAVED", written.name, ToastKind.SUCCESS)

    # -- window commands ---------------------------------------------------

    def _minimise(self) -> None:
        self._root.overrideredirect(False)
        self._root.iconify()

    def _toggle_fullscreen(self) -> None:
        if self._fullscreen:
            self._root.geometry(self._windowed_geometry or f"{self._width}x{self._height}+80+60")
            self._fullscreen = False
        else:
            self._windowed_geometry = self._root.geometry()
            screen = self._physical_size(self._root.winfo_screenwidth(), self._root.winfo_screenheight())
            self._root.geometry(f"{screen[0]}x{screen[1]}+0+0")
            self._fullscreen = True
        self._hud.notify("DISPLAY", "FULLSCREEN" if self._fullscreen else "WINDOWED", ToastKind.INFO)

    def _on_escape(self) -> None:
        if self._fullscreen:
            self._toggle_fullscreen()
        else:
            self.shutdown()

    # -- mouse -------------------------------------------------------------

    def _to_canvas(self, x: int, y: int) -> tuple[int, int]:
        scale = max(self._display_scale, 1e-6)
        return (
            int((x - self._display_origin[0]) / scale),
            int((y - self._display_origin[1]) / scale),
        )

    def _on_mouse_move(self, event: tk.Event) -> None:
        self._mouse = self._to_canvas(event.x, event.y)

    def _on_mouse_leave(self, _event: tk.Event) -> None:
        self._mouse = None

    def _on_mouse_down(self, event: tk.Event) -> None:
        point = self._to_canvas(event.x, event.y)
        self._mouse = point
        region = self._region_at(point[0], point[1], dwellable_only=False)
        if region is None:
            return
        if region.command is Command.DRAG_WINDOW:
            self._drag_offset = (
                event.x_root - self._root.winfo_x(),
                event.y_root - self._root.winfo_y(),
            )
            return
        if region.continuous:
            self._slider_drag = True
            self._dispatch(region.command, region.payload, point)
            return
        self._dispatch(region.command, region.payload, point)

    def _on_mouse_drag(self, event: tk.Event) -> None:
        point = self._to_canvas(event.x, event.y)
        self._mouse = point
        if self._drag_offset is not None and not self._fullscreen:
            self._root.geometry(f"+{event.x_root - self._drag_offset[0]}+{event.y_root - self._drag_offset[1]}")
        elif self._slider_drag:
            self._apply_opacity_from(point)

    def _on_mouse_up(self, _event: tk.Event) -> None:
        self._drag_offset = None
        self._slider_drag = False

    def _on_wheel(self, event: tk.Event) -> None:
        step = 1 if getattr(event, "delta", 0) > 0 else -1
        self._dispatch(
            Command.SELECT_THICKNESS,
            (self._settings.thickness_index + step) % len(THICKNESSES),
        )


# =============================================================================
# SECTION 16 — Entry point
# =============================================================================


def main() -> int:
    """Configure logging, build the application and run it."""
    configure_logging()
    config = AppConfig.from_environment()
    LOGGER.info("AI Virtual Painter starting (output -> %s)", config.output_dir)
    try:
        app = VirtualPainterApp(config)
    except tk.TclError as error:
        LOGGER.error("No display available: %s", error)
        return 1
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
