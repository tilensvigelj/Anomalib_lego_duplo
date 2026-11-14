#!/usr/bin/env python3
"""Deterministic RGB background-subtraction preprocessing for Duplo blocks (Person C - j26).

Takes an RGB photo of a Duplo block and a background reference (computed from a photo set of
the empty background without the Duplo block), compensates for global lighting/exposure shifts,
subtracts the background, enforces a single continuous Duplo block shape, and outputs a
canonical 256x256 RGB crop + mask for Anomalib.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import logging
from pathlib import Path
import threading
from typing import Any, Sequence

import cv2
import numpy as np

try:
    from scipy.ndimage import binary_fill_holes as _scipy_binary_fill_holes
except ImportError:  # pragma: no cover
    _scipy_binary_fill_holes = None

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass(frozen=True)
class PreprocessConfig:
    """Deterministic configuration for RGB background-subtraction Duplo preprocessing.

    Attributes:
        CHROMA_THRESH: CIE Lab chroma difference threshold `hypot(da, db)` vs background.
        LUMA_BRIGHT_THRESH: Positive CIE Lab lightness difference `dL` (`L_fg - L_bg`) threshold.
        LUMA_DARK_THRESH: Negative CIE Lab lightness difference `-dL` (`L_bg - L_fg`) threshold,
            set high enough to reject diffuse cast shadows while catching dark/black bricks.
        GRAD_DIFF_THRESH: Sobel gradient magnitude difference threshold to catch low-contrast
            brick edges/studs (e.g. light brick on white mat).
        BLUR_KERNEL: Odd Gaussian kernel size for suppressing camera sensor / JPEG noise.
        MIN_AREA_PX: Minimum area (in camera pixels) for a valid Duplo brick.
        MAX_AREA_PX: Maximum area (in camera pixels); rejects arms/hands/clutter.
        KERNEL: Elliptical structuring-element diameter (px) for morphological cleaning.
        CLOSE_KERNEL: Larger closing kernel (px) to merge the brick into one continuous shape.
        FILL_CONVEX_HULL: If True, fills concavities/gaps inside the brick's convex hull where
            pixels moderately differ from the background so the block is one continuous shape.
        DILATE_PX: Safety-margin radial dilation (px) so studs, edges, and chips are never clipped.
        BG_FILL: Constant RGB background fill tuple `(R, G, B)` in `[0, 255]`.
        FEATHER_SIGMA: Gaussian sigma (px) for smooth edge transition into `BG_FILL`.
        MARGIN: Fractional margin added on each side of the square bounding box (`0.10` = 10%).
        OUT_SIZE: Final square output resolution `(OUT_SIZE, OUT_SIZE)` in pixels.
        COLOR_FORMAT: Channel order of input `color` and `bg_color` (`"RGB"` or `"BGR"`).
            Output `processed_rgb` is always `"RGB"`.
        CANONICALIZE_ROTATION: If True, rotates the crop via `cv2.minAreaRect` so the long axis
            is horizontal and resolves the 180-degree ambiguity deterministically.
        REJECT_BORDER_TOUCH: If True, returns `valid=False` when the brick touches the image edge.
        MAKE_DEBUG_OVERLAY: If True, renders `info["debug_overlay"]` with mask edges and bbox.
    """

    CHROMA_THRESH: float = 10.0
    LUMA_BRIGHT_THRESH: float = 14.0
    LUMA_DARK_THRESH: float = 42.0
    GRAD_DIFF_THRESH: float = 18.0
    BLUR_KERNEL: int = 5
    MIN_AREA_PX: int = 800
    MAX_AREA_PX: int = 120_000
    KERNEL: int = 5
    CLOSE_KERNEL: int = 7
    FILL_CONVEX_HULL: bool = True
    DILATE_PX: int = 2
    BG_FILL: tuple[int, int, int] = (128, 128, 128)
    FEATHER_SIGMA: float = 1.0
    MARGIN: float = 0.10
    OUT_SIZE: int = 256
    COLOR_FORMAT: str = "RGB"
    CANONICALIZE_ROTATION: bool = True
    REJECT_BORDER_TOUCH: bool = True
    MAKE_DEBUG_OVERLAY: bool = True
    MAX_SEG_DIM: int = 640
    REMOVE_GLARE: bool = True
    GLARE_MIN_L: float = 212.0
    GLARE_PROMINENCE: float = 16.0
    GLARE_MAX_SAT: float = 48.0
    GLARE_INPAINT_RADIUS: int = 4

    def to_json(self, path: str | Path) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data["BG_FILL"] = list(self.BG_FILL)
        out_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> PreprocessConfig:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if "preprocess" in raw and isinstance(raw["preprocess"], dict):
            raw = raw["preprocess"]
        if "BG_FILL" in raw:
            raw["BG_FILL"] = tuple(int(v) for v in raw["BG_FILL"])
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in raw.items() if k in valid_keys}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Background Photo-Set Calibration (Empty Scene Without Duplo Block)
# ---------------------------------------------------------------------------


def compute_background_model(bg_frames: Sequence[np.ndarray]) -> np.ndarray:
    """Compute a noise-free per-pixel median RGB background image from a set of empty-scene photos.

    Using the temporal median across multiple background photos removes sensor noise, flickering
    shadows, and transient artifacts.
    """
    if not bg_frames:
        raise ValueError("bg_frames must contain at least one background image.")

    first_shape = bg_frames[0].shape
    if len(first_shape) != 3 or first_shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB background frames, got shape={first_shape}.")

    stack = np.stack([np.asarray(f, dtype=np.float32) for f in bg_frames], axis=0)
    median_rgb = np.median(stack, axis=0)
    return np.clip(np.rint(median_rgb), 0, 255).astype(np.uint8)


def calibrate_background_from_folder(
    bg_dir: str | Path,
    output_path: str | Path | None = "data/calibration/background_mat.png",
    max_frames: int = 50,
    color_format: str = "RGB",
) -> np.ndarray:
    """Load empty-background photos from `bg_dir`, compute the median background, and save it.

    Returns the calibrated background in `color_format` (`"RGB"` by default).
    """
    folder = Path(bg_dir)
    if not folder.is_dir():
        raise FileNotFoundError(f"Background folder not found: {folder}")

    paths = [
        p for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ][:max_frames]
    if not paths:
        raise FileNotFoundError(f"No background photos ({IMAGE_EXTENSIONS}) found in {folder}")

    frames: list[np.ndarray] = []
    target_hw: tuple[int, int] | None = None
    for p in paths:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        if target_hw is None:
            target_hw = bgr.shape[:2]
        elif bgr.shape[:2] != target_hw:
            bgr = cv2.resize(bgr, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if color_format.upper() == "RGB" else bgr
        frames.append(img)

    if not frames:
        raise ValueError(f"Could not decode any valid images from {folder}")

    bg_model = compute_background_model(frames)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() == ".npy":
            np.save(str(out), bg_model)
        else:
            save_bgr = (
                cv2.cvtColor(bg_model, cv2.COLOR_RGB2BGR)
                if color_format.upper() == "RGB"
                else bg_model
            )
            cv2.imwrite(str(out), save_bgr)
        logger.info("Saved calibrated background (%d photos) -> %s", len(frames), out)

    return bg_model


def load_background_model(
    path_or_dir: str | Path,
    color_format: str = "RGB",
) -> np.ndarray:
    """Load a background model from a folder of photos, a single `.png`/`.jpg`, or a `.npy` file."""
    p = Path(path_or_dir)
    if p.is_dir():
        return calibrate_background_from_folder(p, output_path=None, color_format=color_format)
    if not p.is_file():
        raise FileNotFoundError(f"Background reference not found: {p}")
    if p.suffix.lower() == ".npy":
        arr = np.load(str(p))
        return np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read background image: {p}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if color_format.upper() == "RGB" else bgr


# ---------------------------------------------------------------------------
# Internal Deterministic Vision Helpers
# ---------------------------------------------------------------------------


def _ellipse_kernel(diameter: int) -> np.ndarray:
    size = max(1, int(diameter))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _fast_dilate_ellipse_bool(mask: np.ndarray, diameter: int) -> np.ndarray:
    """Exact O(N) Euclidean dilation replacing slow O(N * D^2) large-kernel `MORPH_ELLIPSE`."""
    d = max(1, int(diameter))
    if d < 15:
        u8 = (mask.astype(np.uint8) * 255) if mask.dtype == bool else mask
        return cv2.dilate(u8, _ellipse_kernel(d), iterations=1) > 0
    fg = mask > 0
    if not np.any(fg):
        return np.zeros(mask.shape, dtype=bool)
    if np.all(fg):
        return np.ones(mask.shape, dtype=bool)
    inv_u8 = np.where(fg, 0, 1).astype(np.uint8)
    dist = cv2.distanceTransform(inv_u8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    radius = (d // 2) + 1e-3
    return dist <= radius


@dataclass
class _PreparedBackground:
    bg_rgb: np.ndarray
    bg_f: np.ndarray
    bmask: np.ndarray
    bg_med: np.ndarray
    bg_lab: np.ndarray
    bg_lab3: np.ndarray
    bg_p: np.ndarray
    inv_bg_p: np.ndarray
    norm_bg_p: np.ndarray
    bg_p3: np.ndarray
    norm_bg_p3: np.ndarray
    s_bg: np.ndarray
    g_bg_l: np.ndarray


_BG_CACHE_LOCK = threading.Lock()
_BG_CACHE: dict[tuple[Any, ...], _PreparedBackground] = {}


def _get_prepared_bg(bg_rgb: np.ndarray, k_blur: int) -> _PreparedBackground:
    """Precompute and cache static background Lab/gradient/norm tensors for realtime (<20ms) frames."""
    h, w, _ = bg_rgb.shape
    flat = bg_rgb.reshape(-1)
    sig = (
        id(bg_rgb),
        h,
        w,
        int(k_blur),
        int(flat[0]),
        int(flat[len(flat) // 2]),
        int(flat[-1]),
    )
    with _BG_CACHE_LOCK:
        cached = _BG_CACHE.get(sig)
        if cached is not None and cached.bg_rgb is bg_rgb:
            return cached

    border = max(8, min(h, w) // 16)
    bmask = np.zeros((h, w), dtype=bool)
    bmask[:border, :] = True
    bmask[-border:, :] = True
    bmask[:, :border] = True
    bmask[:, -border:] = True

    bg_f = bg_rgb.astype(np.float32)
    bg_med = np.median(bg_f[bmask], axis=0)
    bg_smooth = cv2.GaussianBlur(bg_rgb, (k_blur, k_blur), 0) if k_blur > 1 else bg_rgb
    bg_sharp = cv2.GaussianBlur(bg_rgb, (3, 3), 0)

    bg_lab = cv2.cvtColor(bg_smooth, cv2.COLOR_RGB2LAB).astype(np.float32)
    bg_lab3 = cv2.cvtColor(bg_sharp, cv2.COLOR_RGB2LAB).astype(np.float32)

    bg_smooth_f = bg_smooth.astype(np.float32)
    bg_p = bg_smooth_f + 8.0
    inv_bg_p = 1.0 / bg_p
    norm_bg_p = np.linalg.norm(bg_p, axis=2)

    bg_p3 = bg_sharp.astype(np.float32) + 8.0
    norm_bg_p3 = np.linalg.norm(bg_p3, axis=2)

    s_bg = (
        255.0
        * (np.max(bg_smooth_f, axis=2) - np.min(bg_smooth_f, axis=2))
        / np.maximum(np.max(bg_smooth_f, axis=2), 65.0)
    )
    g_bg_l = _grad_magnitude(bg_lab3[:, :, 0])

    prep = _PreparedBackground(
        bg_rgb=bg_rgb,
        bg_f=bg_f,
        bmask=bmask,
        bg_med=bg_med,
        bg_lab=bg_lab,
        bg_lab3=bg_lab3,
        bg_p=bg_p,
        inv_bg_p=inv_bg_p,
        norm_bg_p=norm_bg_p,
        bg_p3=bg_p3,
        norm_bg_p3=norm_bg_p3,
        s_bg=s_bg,
        g_bg_l=g_bg_l,
    )
    with _BG_CACHE_LOCK:
        if len(_BG_CACHE) >= 4:
            _BG_CACHE.clear()
        _BG_CACHE[sig] = prep
    return prep


def _fill_interior_holes(mask_u8: np.ndarray) -> np.ndarray:
    """Fill all interior topological holes in a binary `{0, 255}` uint8 mask."""
    if not np.any(mask_u8):
        return np.zeros_like(mask_u8, dtype=np.uint8)
    if _scipy_binary_fill_holes is not None:
        return (_scipy_binary_fill_holes(mask_u8 > 0).astype(np.uint8)) * 255

    h, w = mask_u8.shape
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded[1 : h + 1, 1 : w + 1] = np.where(mask_u8 > 0, 255, 0)
    flood = padded.copy()
    ff_mask = np.zeros((h + 4, w + 4), dtype=np.uint8)
    cv2.floodFill(flood, ff_mask, seedPoint=(0, 0), newVal=128)
    interior_holes = flood[1 : h + 1, 1 : w + 1] == 0
    return np.where((mask_u8 > 0) | interior_holes, 255, 0).astype(np.uint8)


def _keep_largest_component(mask_u8: np.ndarray) -> tuple[np.ndarray, int]:
    """Keep the single largest 8-connected component (one continuous Duplo block)."""
    binary = (mask_u8 > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask_u8, dtype=np.uint8), 0

    areas = stats[1:, cv2.CC_STAT_AREA]
    best_idx = 1 + int(np.argmax(areas))
    best_area = int(stats[best_idx, cv2.CC_STAT_AREA])
    largest = np.where(labels == best_idx, 255, 0).astype(np.uint8)
    return largest, best_area


def _convex_hull_mask(mask_u8: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask_u8, dtype=np.uint8)
    hull = cv2.convexHull(np.vstack(contours))
    hull_mask = np.zeros_like(mask_u8, dtype=np.uint8)
    cv2.fillConvexPoly(hull_mask, hull, 255)
    return hull_mask


def _normalize_color_input(color: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    if not isinstance(color, np.ndarray):
        raise TypeError("Expected numpy.ndarray image.")
    if color.ndim != 3 or color.shape[2] != 3 or color.dtype != np.uint8:
        raise ValueError(f"Expected HxWx3 uint8 image, got shape={color.shape}, dtype={color.dtype}.")
    fmt = cfg.COLOR_FORMAT.upper().strip()
    if fmt == "RGB":
        return color.copy()
    if fmt == "BGR":
        return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    raise ValueError(f"COLOR_FORMAT must be 'RGB' or 'BGR', got {cfg.COLOR_FORMAT!r}.")


def _synthesize_border_background(color_rgb: np.ndarray) -> np.ndarray:
    """Fallback when no background photo set is provided: build a smooth pad image from borders."""
    h, w, _ = color_rgb.shape
    border = max(8, min(h, w) // 16)
    mask = np.zeros((h, w), dtype=bool)
    mask[:border, :] = True
    mask[-border:, :] = True
    mask[:, :border] = True
    mask[:, -border:] = True
    med_rgb = np.median(color_rgb[mask], axis=0).astype(np.uint8)
    return np.full_like(color_rgb, med_rgb, dtype=np.uint8)


def _align_foreground_lighting(
    fg_rgb: np.ndarray,
    bg_rgb: np.ndarray,
    prep_bg: _PreparedBackground | None = None,
) -> np.ndarray:
    """Two-pass spatial lighting alignment: global border gain + smooth 2D mat illumination field.

    Compensates for camera auto-exposure, auto-white-balance, and spatial lamp/vignetting shifts
    between the background reference set and the live frame without fitting to the brick or shadow.
    """
    h, w, _ = fg_rgb.shape
    fg_f = fg_rgb.astype(np.float32)
    if prep_bg is not None:
        bmask = prep_bg.bmask
        bg_f = prep_bg.bg_f
        bg_med = prep_bg.bg_med
    else:
        border = max(8, min(h, w) // 16)
        bmask = np.zeros((h, w), dtype=bool)
        bmask[:border, :] = True
        bmask[-border:, :] = True
        bmask[:, :border] = True
        bmask[:, -border:] = True
        bg_f = bg_rgb.astype(np.float32)
        bg_med = np.median(bg_f[bmask], axis=0)

    fg_med = np.median(fg_f[bmask], axis=0)
    g0 = np.clip(bg_med / np.maximum(fg_med, 1.0), 0.30, 3.30).astype(np.float32)
    fg_p1 = np.clip(fg_f * g0[None, None, :], 0.0, 255.0)

    max_channel_diff = np.max(np.abs(fg_p1 - bg_f), axis=2)
    inliers = (max_channel_diff < 14.0).astype(np.float32)
    if float(np.mean(inliers)) >= 0.15 and min(h, w) >= 64:
        ds_w = max(16, w // 4)
        ds_h = max(16, h // 4)
        w_ds = cv2.resize(inliers, (ds_w, ds_h), interpolation=cv2.INTER_AREA)
        fg_ds = cv2.resize(fg_f * inliers[:, :, None], (ds_w, ds_h), interpolation=cv2.INTER_AREA)
        bg_ds = cv2.resize(bg_f * inliers[:, :, None], (ds_w, ds_h), interpolation=cv2.INTER_AREA)

        sigma = max(6.0, min(ds_h, ds_w) * 0.20)
        w_blur = cv2.GaussianBlur(w_ds, (0, 0), sigmaX=sigma, sigmaY=sigma)
        fg_blur = cv2.GaussianBlur(fg_ds, (0, 0), sigmaX=sigma, sigmaY=sigma)
        bg_blur = cv2.GaussianBlur(bg_ds, (0, 0), sigmaX=sigma, sigmaY=sigma)

        prior = 0.05
        gain_ds = (bg_blur + prior * bg_med[None, None, :]) / np.maximum(
            fg_blur + prior * fg_med[None, None, :], 1.0
        )
        conf = (w_blur / (w_blur + prior))[:, :, None]
        gain_ds = conf * gain_ds + (1.0 - conf) * g0[None, None, :]
        gain_map = cv2.resize(gain_ds, (w, h), interpolation=cv2.INTER_LINEAR)
        gain_map = np.clip(gain_map, 0.30, 3.30)
        aligned = fg_f * gain_map
    else:
        aligned = fg_p1

    return np.clip(np.rint(aligned), 0, 255).astype(np.uint8)


def _grad_magnitude(gray_f32: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray_f32, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f32, cv2.CV_32F, 0, 1, ksize=3)
    return np.hypot(gx, gy)


def _segment_duplo_from_background(
    color_rgb: np.ndarray,
    bg_rgb: np.ndarray,
    cfg: PreprocessConfig,
) -> tuple[np.ndarray, int]:
    """Segment Duplo block while differentiating ground shadows from shadows on the block.

    Physical separation of Ground Shadows vs. Block Shadows:
      1. Neutral cast ground shadows: darker than the calibrated mat (`d_l <= -4`) with BOTH
         low illumination-invariant spectral angle AND low Lab chroma difference.
      2. Colored bounce-light ground shadows (under overhangs or beside bright walls):
         - Penumbra bounce fringe connects smoothly from `neutral_ground` without crossing any
           sharp chromatic/luminance brick silhouette barrier (`wall_barrier`).
         - Deep occluded ground umbra (`d_l < -42`, `max_ratio < 0.79`, non-ink-black) plus its
           contiguous ground transition collar floods from `outer_ground` and stops dead at the
           sharp overhang rim cliff (`umbra_barrier`).
      3. Self-shaded faces ON the Duplo block (shaded vertical walls, protruding studs, curved
         walls, Fresnel-sheen rims): sit outside `ground_mask`, retain reflectance ratio
         `max_ratio >= 0.77` (`d_l > -50`), and align with the lit brick's dominant `(da, db)`
         chromaticity vector (`hue_cos > 0.68`).
    """
    h, w, _ = color_rgb.shape
    k_blur = max(1, int(cfg.BLUR_KERNEL) | 1)
    prep_bg = _get_prepared_bg(bg_rgb, k_blur)
    fg_aligned = _align_foreground_lighting(color_rgb, bg_rgb, prep_bg=prep_bg)

    if k_blur > 1:
        fg_smooth = cv2.GaussianBlur(fg_aligned, (k_blur, k_blur), 0)
    else:
        fg_smooth = fg_aligned

    # Sharp 3x3 scale for accurate 1px silhouette edge barriers and watershed snapping
    fg_sharp = cv2.GaussianBlur(fg_aligned, (3, 3), 0)
    fg_lab3 = cv2.cvtColor(fg_sharp, cv2.COLOR_RGB2LAB).astype(np.float32)
    bg_lab3 = prep_bg.bg_lab3

    fg_lab = cv2.cvtColor(fg_smooth, cv2.COLOR_RGB2LAB).astype(np.float32)
    bg_lab = prep_bg.bg_lab

    d_l = fg_lab[:, :, 0] - bg_lab[:, :, 0]
    d_a = fg_lab[:, :, 1] - bg_lab[:, :, 1]
    d_b = fg_lab[:, :, 2] - bg_lab[:, :, 2]
    chroma_diff = np.hypot(d_a, d_b)

    d_a3 = fg_lab3[:, :, 1] - bg_lab3[:, :, 1]
    d_b3 = fg_lab3[:, :, 2] - bg_lab3[:, :, 2]
    chroma_diff3 = np.hypot(d_a3, d_b3)

    fg_f = fg_smooth.astype(np.float32)
    fg_p = fg_f + 8.0
    bg_p = prep_bg.bg_p
    max_ratio = np.max(fg_p * prep_bg.inv_bg_p, axis=2)

    dot = np.sum(fg_p * bg_p, axis=2)
    norms = np.linalg.norm(fg_p, axis=2) * prep_bg.norm_bg_p
    spec_angle = np.degrees(np.arccos(np.clip(dot / np.maximum(norms, 1e-6), -1.0, 1.0)))

    fg_p3 = fg_sharp.astype(np.float32) + 8.0
    bg_p3 = prep_bg.bg_p3
    dot3 = np.sum(fg_p3 * bg_p3, axis=2)
    norms3 = np.linalg.norm(fg_p3, axis=2) * prep_bg.norm_bg_p3
    spec_angle3 = np.degrees(np.arccos(np.clip(dot3 / np.maximum(norms3, 1e-6), -1.0, 1.0)))

    s_fg = (
        255.0
        * (np.max(fg_f, axis=2) - np.min(fg_f, axis=2))
        / np.maximum(np.max(fg_f, axis=2), 65.0)
    )
    s_diff = s_fg - prep_bg.s_bg

    # Sharp edge gradients (luminance, Lab chromaticity difference, spectral angle)
    g_l = _grad_magnitude(fg_lab3[:, :, 0])
    grad_excess = np.maximum(0.0, g_l - prep_bg.g_bg_l)
    g_col = np.hypot(_grad_magnitude(d_a3), _grad_magnitude(d_b3))
    g_ang = _grad_magnitude(spec_angle3)

    min_spec = max(3.5, min(float(cfg.CHROMA_THRESH), 12.0) * 0.42)

    # 1. Lit foreground core (unambiguously non-shadow faces of the Duplo block)
    is_lit_ch = (
        (chroma_diff > cfg.CHROMA_THRESH)
        & (d_l > -24.0)
        & (max_ratio > 0.86)
        & (spec_angle > min_spec)
    )
    is_brighter = (d_l > cfg.LUMA_BRIGHT_THRESH) & (max_ratio > 1.04)
    is_structural_edge = (
        (grad_excess > cfg.GRAD_DIFF_THRESH)
        & (d_l > -4.0)
        & (max_ratio > 0.94)
    )
    lit_core = is_lit_ch | is_brighter | is_structural_edge

    # Estimate dominant brick chromaticity unit vector (ua, ub) when a colored brick is present
    lit_count = int(np.count_nonzero(lit_core))
    brick_chroma = 0.0
    ua, ub = 0.0, 0.0
    if lit_count >= 150:
        med_da = float(np.median(d_a[lit_core]))
        med_db = float(np.median(d_b[lit_core]))
        brick_chroma = float(np.hypot(med_da, med_db))
        if brick_chroma >= 10.0:
            ua = med_da / brick_chroma
            ub = med_db / brick_chroma

    if brick_chroma >= 10.0:
        chroma_proj = d_a * ua + d_b * ub
        hue_cos = chroma_proj / np.maximum(chroma_diff, 1e-4)
    else:
        chroma_proj = chroma_diff
        hue_cos = np.ones_like(chroma_diff)

    # 2. Differentiate Ground Shadows vs Block Shadows:
    # 2a. Clean mat + neutral gray cast shadow on the ground (BOTH low angle AND low chroma)
    is_clean_mat = (
        (chroma_diff < min(8.0, cfg.CHROMA_THRESH * 0.65))
        & (np.abs(d_l) < min(15.0, cfg.LUMA_BRIGHT_THRESH * 0.75))
        & (grad_excess < (cfg.GRAD_DIFF_THRESH * 0.80))
    )
    is_neutral_ground_shadow = (
        (d_l <= -4.0)
        & (max_ratio < 0.98)
        & (spec_angle < 4.5)
        & (chroma_diff < min(8.0, cfg.CHROMA_THRESH * 0.68))
    )

    is_deep_dark_raw = ((-d_l) > (cfg.LUMA_DARK_THRESH * 1.15)) & (max_ratio < 0.46)
    if lit_count < cfg.MIN_AREA_PX:
        is_neutral_ground_shadow = is_neutral_ground_shadow & (~is_deep_dark_raw)

    neutral_ground = is_clean_mat | is_neutral_ground_shadow

    # 2b. Colored bounce-light ground shadow (under overhangs or beside chromatic walls):
    neg_dl = np.maximum(0.0, -d_l)
    is_deep_umbra_cand = (d_l < -42.0) & (max_ratio < 0.79) & (~is_deep_dark_raw)

    is_penumbra_bounce_cand = (
        (d_l <= -10.0)
        & (max_ratio < 0.95)
        & (chroma_diff < (6.2 + 0.22 * neg_dl))
        & (spec_angle < (4.0 + 0.14 * neg_dl))
    )

    # Barrier 1 (for penumbra bounce flood): stops at any sharp chromatic/luma brick wall silhouette.
    chromatic_brick_near = (
        cv2.dilate(
            ((chroma_diff > 10.5) & (max_ratio >= 0.82) & (d_l > -40.0)).astype(np.uint8) * 255,
            _ellipse_kernel(5),
            iterations=1,
        )
        > 0
    )
    wall_edge = (
        ((g_l > 24.0) & chromatic_brick_near)
        | ((g_col > 8.5) & (g_ang > 4.2) & (g_l > 11.0))
        | ((g_col > 10.5) & (g_ang > 5.4))
        | ((g_l + 1.6 * g_col + 2.2 * g_ang) > 48.0)
    )
    wall_barrier = cv2.dilate(wall_edge.astype(np.uint8) * 255, _ellipse_kernel(3), iterations=1) > 0

    penumbra_domain = ((neutral_ground | (is_penumbra_bounce_cand & (~lit_core))) & (~wall_barrier)).astype(np.uint8)
    num_p, p_labels = cv2.connectedComponents(penumbra_domain, connectivity=4)
    if num_p > 1 and np.any(neutral_ground & (~wall_barrier)):
        valid_p_ids = np.unique(p_labels[neutral_ground & (~wall_barrier)])
        valid_p_ids = valid_p_ids[valid_p_ids > 0]
        penumbra_bounce = np.isin(p_labels, valid_p_ids)
    else:
        penumbra_bounce = np.zeros((h, w), dtype=bool)

    outer_ground = neutral_ground | penumbra_bounce

    # Barrier 2 (overhang rim & shaded-wall barrier): sharp luminance or chromatic cliff bordering the brick
    overhang_brick_near = (
        cv2.dilate(
            (
                ((max_ratio >= 0.85) & (d_l > -38.0) & (chroma_diff > 14.0))
                | ((max_ratio >= 0.82) & (d_l > -41.0) & (chroma_diff > 20.0))
            ).astype(np.uint8)
            * 255,
            _ellipse_kernel(5),
            iterations=1,
        )
        > 0
    )
    overhang_rim_edge = overhang_brick_near & (
        (g_l > 22.0) | ((g_col > 9.5) & (g_ang > 4.8))
    )
    umbra_barrier = cv2.dilate(overhang_rim_edge.astype(np.uint8) * 255, _ellipse_kernel(3), iterations=1) > 0

    outer_touch = cv2.dilate(outer_ground.astype(np.uint8) * 255, _ellipse_kernel(3), iterations=1) > 0
    umbra_domain = ((outer_touch | (is_deep_umbra_cand & (~lit_core))) & (~umbra_barrier)).astype(np.uint8)
    num_u, u_labels = cv2.connectedComponents(umbra_domain, connectivity=4)
    if num_u > 1 and np.any(outer_touch & (~umbra_barrier)):
        valid_u_ids = np.unique(u_labels[outer_touch & (~umbra_barrier)])
        valid_u_ids = valid_u_ids[valid_u_ids > 0]
        deep_umbra_ground = np.isin(u_labels, valid_u_ids) & is_deep_umbra_cand
    else:
        deep_umbra_ground = np.zeros((h, w), dtype=bool)

    # 2c. Overhang umbra penumbra collar: around any detected `deep_umbra_ground`, include its
    # contiguous ground transition halo (`d_l < -18, max_ratio < 0.89`) up to `umbra_barrier`.
    if np.any(deep_umbra_ground):
        umbra_halo_zone = _fast_dilate_ellipse_bool(deep_umbra_ground, 111)
        collar_cand = (
            umbra_halo_zone
            & (~lit_core)
            & (~is_deep_dark_raw)
            & (d_l < -18.0)
            & (max_ratio < 0.89)
            & (chroma_diff < (6.8 + 0.30 * neg_dl))
        )
        collar_seed = deep_umbra_ground & (~umbra_barrier)
        collar_domain = ((collar_seed | collar_cand) & (~umbra_barrier)).astype(np.uint8)
        num_c, c_labels = cv2.connectedComponents(collar_domain, connectivity=4)
        if num_c > 1 and np.any(collar_seed):
            valid_c_ids = np.unique(c_labels[collar_seed])
            valid_c_ids = valid_c_ids[valid_c_ids > 0]
            umbra_collar_ground = np.isin(c_labels, valid_c_ids)
        else:
            umbra_collar_ground = np.zeros((h, w), dtype=bool)
    else:
        umbra_collar_ground = np.zeros((h, w), dtype=bool)

    ground_mask = outer_ground | deep_umbra_ground | umbra_collar_ground

    # 3. Shaded faces ON the Duplo block (self-shaded vertical walls, studs, curved faces):
    # Must either be genuinely shaded darker than the mat (`d_l < -14.5`) or strongly chromatic,
    # preventing faint mat interreflection glow (`d_l ~ -10, ch ~ 8..10`) outside bright walls from being seeded.
    shaded_chroma_min = max(9.0, min(float(cfg.CHROMA_THRESH), 12.5) * 0.75)
    if brick_chroma >= 10.0:
        is_shaded_wall = (
            (~ground_mask)
            & (hue_cos > 0.68)
            & (chroma_proj > shaded_chroma_min)
            & (spec_angle > 5.4)
            & (max_ratio >= 0.77)
            & (d_l > -50.0)
            & ((d_l < -14.5) | (chroma_proj > max(12.0, cfg.CHROMA_THRESH)))
        )
    else:
        is_shaded_wall = (
            (~ground_mask)
            & (chroma_diff > max(10.5, cfg.CHROMA_THRESH * 0.85))
            & (spec_angle > max(6.5, min_spec * 1.25))
            & (s_diff > 50.0)
            & (max_ratio >= 0.75)
        )

    is_deep_dark = is_deep_dark_raw & (~ground_mask)

    # Use a 3px opening kernel on `sure_fg` so narrow ~8-12px protruding studs are never shaved off
    seed_open_k = _ellipse_kernel(min(int(cfg.KERNEL), 3))
    core_close_k = _ellipse_kernel(max(cfg.KERNEL, 9))
    fg_union = lit_core | is_shaded_wall | is_deep_dark
    sure_fg_u8 = cv2.morphologyEx(fg_union.astype(np.uint8) * 255, cv2.MORPH_OPEN, seed_open_k)
    sure_fg_u8 = cv2.morphologyEx(sure_fg_u8, cv2.MORPH_CLOSE, core_close_k)

    main_core, core_area = _keep_largest_component(sure_fg_u8)
    if core_area == 0:
        return main_core, 0

    # Include nearby sibling core islands of the same brick (e.g. studs split by a shaded crease
    # or a dark marker stroke across the brick) before watershed boundary snapping.
    proximity_zone = _fast_dilate_ellipse_bool(main_core > 0, 35)
    num_lbl, lbls, _, _ = cv2.connectedComponentsWithStats((sure_fg_u8 > 0).astype(np.uint8), connectivity=8)
    fg_comp = main_core.copy()
    for lbl_id in range(1, num_lbl):
        comp_mask = lbls == lbl_id
        if np.any(comp_mask & proximity_zone):
            fg_comp[comp_mask] = 255

    fg_comp = _fill_interior_holes(fg_comp)
    fg_seed = cv2.erode(fg_comp, _ellipse_kernel(3), iterations=1)
    if not np.any(fg_seed):
        fg_seed = fg_comp.copy()

    # 4. Sure BG seed from `ground_mask` outside `fg_comp`
    bg_raw = ground_mask & (fg_comp == 0)
    bg_seed = cv2.erode(bg_raw.astype(np.uint8) * 255, _ellipse_kernel(3), iterations=1)
    if not _touches_image_border(fg_seed):
        bg_seed[0, :] = 255
        bg_seed[-1, :] = 255
        bg_seed[:, 0] = 255
        bg_seed[:, -1] = 255

    # 5. Marker-controlled Watershed on sharp 3x3 [0.95 * L, 10.0 * SpecAngle, 5.0 * ChromaDiff]
    c0 = np.clip(np.rint(fg_lab3[:, :, 0] * 0.95), 0, 255).astype(np.uint8)
    c1 = np.clip(np.rint(spec_angle3 * 10.0), 0, 255).astype(np.uint8)
    c2 = np.clip(np.rint(chroma_diff3 * 5.0), 0, 255).astype(np.uint8)
    ws_img = np.stack([c0, c1, c2], axis=2)

    markers = np.zeros((h, w), dtype=np.int32)
    markers[bg_seed > 0] = 1
    markers[fg_seed > 0] = 2
    cv2.watershed(ws_img, markers)

    ws_fg = np.where(markers == 2, 255, 0).astype(np.uint8)
    open_k = _ellipse_kernel(min(int(cfg.KERNEL), 3))
    close_k = _ellipse_kernel(max(cfg.KERNEL, cfg.CLOSE_KERNEL))
    cleaned = cv2.morphologyEx(ws_fg, cv2.MORPH_OPEN, open_k)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_k)

    largest, area = _keep_largest_component(cleaned)
    if area == 0:
        return largest, 0
    largest = _fill_interior_holes(largest)

    # 6. Recover specular-sheen / low-contrast brick patches strictly inside convex hull (NEVER ground shadows!)
    if cfg.FILL_CONVEX_HULL:
        hull_u8 = _convex_hull_mask(largest)
        non_shadow_lit_fg = (
            (~ground_mask)
            & (d_l >= -6.0)
            & (max_ratio >= 0.94)
            & (
                (chroma_diff > (cfg.CHROMA_THRESH * 0.55))
                | (d_l > (cfg.LUMA_BRIGHT_THRESH * 0.55))
                | (grad_excess > (cfg.GRAD_DIFF_THRESH * 0.75))
            )
        )
        shaded_hull_fg = (
            (~ground_mask)
            & (brick_chroma >= 10.0)
            & (hue_cos > 0.65)
            & (chroma_proj > 7.0)
            & (spec_angle > 4.4)
            & (max_ratio >= 0.78)
            & (d_l > -44.0)
            & ((d_l < -14.5) | (chroma_proj > max(11.5, cfg.CHROMA_THRESH * 0.95)))
        )
        grown = np.where(
            (largest > 0) | ((hull_u8 > 0) & (non_shadow_lit_fg | shaded_hull_fg)),
            255,
            0,
        ).astype(np.uint8)
        grown = cv2.morphologyEx(grown, cv2.MORPH_CLOSE, _ellipse_kernel(5))
        grown = _fill_interior_holes(grown)
        largest, _ = _keep_largest_component(np.maximum(largest, grown))

    return largest, int(np.count_nonzero(largest))


def _touches_image_border(mask_u8: np.ndarray) -> bool:
    return bool(
        np.any(mask_u8[0, :])
        or np.any(mask_u8[-1, :])
        or np.any(mask_u8[:, 0])
        or np.any(mask_u8[:, -1])
    )


def _apply_background_fill(
    color_rgb: np.ndarray,
    mask_u8: np.ndarray,
    cfg: PreprocessConfig,
) -> np.ndarray:
    bg = np.array(cfg.BG_FILL, dtype=np.uint8)
    clean_fg = color_rgb.copy()
    clean_fg[mask_u8 == 0] = bg
    if cfg.FEATHER_SIGMA <= 0.0:
        return clean_fg

    alpha = cv2.GaussianBlur(
        mask_u8.astype(np.float32) * (1.0 / 255.0),
        ksize=(0, 0),
        sigmaX=float(cfg.FEATHER_SIGMA),
        sigmaY=float(cfg.FEATHER_SIGMA),
        borderType=cv2.BORDER_CONSTANT,
    )
    blended = (
        alpha[:, :, None] * clean_fg.astype(np.float32)
        + (1.0 - alpha[:, :, None]) * bg.astype(np.float32)[None, None, :]
    )
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8)


def _square_crop_and_resize(
    fg_rgb: np.ndarray,
    mask_u8: np.ndarray,
    cfg: PreprocessConfig,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    h_img, w_img = mask_u8.shape
    x, y, w, h = cv2.boundingRect(mask_u8)
    base_side = max(w, h)
    side = max(1, int(np.rint(base_side * (1.0 + 2.0 * cfg.MARGIN))))
    cx = x + w * 0.5
    cy = y + h * 0.5
    x0 = int(np.floor(cx - side * 0.5))
    y0 = int(np.floor(cy - side * 0.5))
    x1, y1 = x0 + side, y0 + side

    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w_img, x1), min(h_img, y1)

    square_rgb = np.empty((side, side, 3), dtype=np.uint8)
    square_rgb[:, :] = np.array(cfg.BG_FILL, dtype=np.uint8)
    square_mask = np.zeros((side, side), dtype=np.uint8)

    if sx1 > sx0 and sy1 > sy0:
        dx0, dy0 = sx0 - x0, sy0 - y0
        dx1, dy1 = dx0 + (sx1 - sx0), dy0 + (sy1 - sy0)
        square_rgb[dy0:dy1, dx0:dx1] = fg_rgb[sy0:sy1, sx0:sx1]
        square_mask[dy0:dy1, dx0:dx1] = mask_u8[sy0:sy1, sx0:sx1]

    out_size = int(cfg.OUT_SIZE)
    resized_rgb = cv2.resize(square_rgb, (out_size, out_size), interpolation=cv2.INTER_AREA)
    resized_mask_raw = cv2.resize(square_mask, (out_size, out_size), interpolation=cv2.INTER_AREA)
    resized_mask = np.where(resized_mask_raw >= 128, 255, 0).astype(np.uint8)
    return resized_rgb, resized_mask, (x0, y0, x1, y1)


def _should_flip_180_rgb(rot_mask_u8: np.ndarray, rot_rgb: np.ndarray) -> bool:
    """Resolve 180-degree ambiguity deterministically using stud/edge gradient & luminance moments."""
    ys, xs = np.nonzero(rot_mask_u8 > 0)
    if xs.size == 0:
        return False
    mx, my = float(np.mean(xs)), float(np.mean(ys))

    gray = cv2.cvtColor(rot_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    grad = _grad_magnitude(gray)
    # Studs side has stronger circular edge energy and highlight luminance
    weights = (grad[ys, xs] + 0.25 * gray[ys, xs] + 1.0).astype(np.float64)
    w_sum = float(np.sum(weights))
    wx = float(np.sum(xs * weights) / w_sum)
    wy = float(np.sum(ys * weights) / w_sum)
    if abs(wy - my) > 0.35:
        return wy > my
    return wx < mx


def _canonicalize_and_crop(
    fg_rgb: np.ndarray,
    mask_u8: np.ndarray,
    cfg: PreprocessConfig,
) -> tuple[np.ndarray, np.ndarray, float, tuple[int, int, int, int]]:
    ys, xs = np.nonzero(mask_u8 > 0)
    pts = np.column_stack((xs, ys)).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)

    e0 = box[1] - box[0]
    e1 = box[3] - box[0]
    long_vec = e0 if (np.hypot(e0[0], e0[1]) >= np.hypot(e1[0], e1[1])) else e1
    if long_vec[0] < 0 or (long_vec[0] == 0 and long_vec[1] < 0):
        long_vec = -long_vec

    base_angle_deg = float(np.degrees(np.arctan2(long_vec[1], long_vec[0])))
    _, _, w, h = cv2.boundingRect(mask_u8)
    patch_rad = int(np.ceil(np.hypot(w, h) * (0.5 + cfg.MARGIN))) + 8
    patch_side = 2 * patch_rad + 1

    rcx, rcy = int(np.rint(rect[0][0])), int(np.rint(rect[0][1]))
    px0, py0 = rcx - patch_rad, rcy - patch_rad
    px1, py1 = px0 + patch_side, py0 + patch_side

    h_img, w_img = mask_u8.shape
    sx0, sy0 = max(0, px0), max(0, py0)
    sx1, sy1 = min(w_img, px1), min(h_img, py1)

    bg_tuple = tuple(int(v) for v in cfg.BG_FILL)
    patch_rgb = np.full((patch_side, patch_side, 3), bg_tuple, dtype=np.uint8)
    patch_mask = np.zeros((patch_side, patch_side), dtype=np.uint8)

    if sx1 > sx0 and sy1 > sy0:
        dx0, dy0 = sx0 - px0, sy0 - py0
        dx1, dy1 = dx0 + (sx1 - sx0), dy0 + (sy1 - sy0)
        patch_rgb[dy0:dy1, dx0:dx1] = fg_rgb[sy0:sy1, sx0:sx1]
        patch_mask[dy0:dy1, dx0:dx1] = mask_u8[sy0:sy1, sx0:sx1]

    center = (float(patch_rad), float(patch_rad))
    rot_mat = cv2.getRotationMatrix2D(center, base_angle_deg, 1.0)
    rot_rgb = cv2.warpAffine(
        patch_rgb,
        rot_mat,
        (patch_side, patch_side),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=bg_tuple,
    )
    rot_mask = cv2.warpAffine(
        patch_mask,
        rot_mat,
        (patch_side, patch_side),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    total_angle = base_angle_deg
    if _should_flip_180_rgb(rot_mask, rot_rgb):
        rot_rgb = np.ascontiguousarray(rot_rgb[::-1, ::-1])
        rot_mask = np.ascontiguousarray(rot_mask[::-1, ::-1])
        total_angle += 180.0
        if total_angle > 180.0:
            total_angle -= 360.0

    rot_rgb = _apply_background_fill(rot_rgb, rot_mask, replace(cfg, FEATHER_SIGMA=0.0))
    cropped_rgb, cropped_mask, local_crop_box = _square_crop_and_resize(rot_rgb, rot_mask, cfg)
    global_crop_box = (
        px0 + local_crop_box[0],
        py0 + local_crop_box[1],
        px0 + local_crop_box[2],
        py0 + local_crop_box[3],
    )
    return cropped_rgb, cropped_mask, float(total_angle), global_crop_box


def remove_specular_glare(
    crop_rgb: np.ndarray,
    mask_u8: np.ndarray | None,
    cfg: PreprocessConfig = PreprocessConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    """Detect and remove specular lamp glare/reflections inside the Duplo brick mask.

    Uses the Dichromatic Reflection Model on plastic (`I = I_diffuse + I_specular`):
    1. Computes masked normalized local luminance (`L_local`) strictly inside the eroded brick
       mask so the neutral gray `(128, 128, 128)` background border never biases edge pixels.
    2. Flags specular glare cores that are simultaneously:
       - high in absolute brightness (`V >= GLARE_MIN_L` and above brick median lightness),
       - a local luminance peak (`L - L_local >= GLARE_PROMINENCE`, so flat white bricks are untouched),
       - desaturated/bleached toward white (`S <= GLARE_MAX_SAT` or `< 0.55 * median_S` at `V >= 225`),
       ensuring dark and colored marker strokes are 100% preserved.
    3. Expands `2 px` into the bright glare halo and reconstructs diffuse plastic appearance via
       Telea fast-marching inpainting with smooth `1 px` Gaussian feathering.
    """
    h, w, _ = crop_rgb.shape
    empty_glare = np.zeros((h, w), dtype=np.uint8)
    if not cfg.REMOVE_GLARE:
        return crop_rgb, empty_glare

    if mask_u8 is None:
        bg_arr = np.array(cfg.BG_FILL, dtype=np.int16)
        fg_raw = (np.max(np.abs(crop_rgb.astype(np.int16) - bg_arr), axis=2) > 6).astype(np.uint8) * 255
    else:
        fg_raw = np.where(mask_u8 > 0, 255, 0).astype(np.uint8)

    # Erode 3px inside the brick silhouette so the feathered BG_FILL border is never touched
    fg_interior_u8 = cv2.erode(fg_raw, _ellipse_kernel(5), iterations=1)
    fg = fg_interior_u8 > 0
    brick_area = int(np.count_nonzero(fg))
    if brick_area < 64:
        return crop_rgb, empty_glare

    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    luma = lab[:, :, 0].astype(np.float32)

    # Masked normalized Gaussian blur (sigma=9) computes local diffuse neighborhood strictly inside brick
    fg_f = fg.astype(np.float32)
    num_blur = cv2.GaussianBlur(luma * fg_f, (0, 0), sigmaX=9.0, sigmaY=9.0)
    den_blur = cv2.GaussianBlur(fg_f, (0, 0), sigmaX=9.0, sigmaY=9.0)
    l_local = num_blur / np.maximum(den_blur, 1e-4)
    prominence = luma - l_local

    med_l = float(np.median(luma[fg]))
    med_s = float(np.median(sat[fg]))

    bleached_white = sat <= float(cfg.GLARE_MAX_SAT)
    bleached_chromatic = (med_s >= 70.0) & (sat <= (med_s * 0.55)) & (val >= 225.0)

    glare_core = (
        fg
        & (val >= float(cfg.GLARE_MIN_L))
        & (luma >= max(202.0, min(245.0, med_l + 10.0)))
        & (prominence >= float(cfg.GLARE_PROMINENCE))
        & (bleached_white | bleached_chromatic)
    )
    if not np.any(glare_core):
        return crop_rgb, empty_glare

    # Filter connected components: drop 1px sensor noise and reject broad uniform lit faces (>22% of brick)
    max_glare_area = max(16, int(round(brick_area * 0.22)))
    num_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(glare_core.astype(np.uint8), connectivity=8)
    filtered_core = np.zeros((h, w), dtype=np.uint8)
    for idx in range(1, num_lbl):
        area_i = int(stats[idx, cv2.CC_STAT_AREA])
        if 2 <= area_i <= max_glare_area:
            filtered_core[lbls == idx] = 255

    if not np.any(filtered_core):
        return crop_rgb, empty_glare

    # Expand 2px into bright glare halo so Telea inpainting samples true diffuse plastic color, not bright rim
    halo_cand = fg & (prominence >= 5.0) & (val >= (float(cfg.GLARE_MIN_L) - 28.0))
    dilated = cv2.dilate(filtered_core, _ellipse_kernel(5), iterations=1) > 0
    glare_mask = np.where(
        (filtered_core > 0) | (dilated & halo_cand),
        255,
        0,
    ).astype(np.uint8)

    inpaint_r = max(2, int(cfg.GLARE_INPAINT_RADIUS))
    repaired_bgr = cv2.inpaint(
        cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR),
        glare_mask,
        inpaintRadius=float(inpaint_r),
        flags=cv2.INPAINT_TELEA,
    )
    repaired_rgb = cv2.cvtColor(repaired_bgr, cv2.COLOR_BGR2RGB)

    # Smooth 1px alpha feather on the repaired glare boundary for invisible transitions
    alpha = cv2.GaussianBlur(glare_mask.astype(np.float32) * (1.0 / 255.0), (0, 0), sigmaX=1.0, sigmaY=1.0)
    alpha = alpha * fg_f
    blended = (
        alpha[:, :, None] * repaired_rgb.astype(np.float32)
        + (1.0 - alpha[:, :, None]) * crop_rgb.astype(np.float32)
    )
    out_rgb = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
    return out_rgb, glare_mask


def create_debug_overlay(
    color_rgb: np.ndarray,
    full_mask_u8: np.ndarray,
    info: dict[str, Any],
) -> np.ndarray:
    overlay = color_rgb.copy()
    valid = bool(info.get("valid", False))
    edge_color = (0, 255, 0) if valid else (255, 64, 64)

    if np.any(full_mask_u8):
        contours, _ = cv2.findContours(full_mask_u8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, edge_color, thickness=2, lineType=cv2.LINE_AA)
        x, y, w, h = info.get("bbox", (0, 0, 0, 0))
        if w > 0 and h > 0:
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 200, 255), thickness=1)

    label = (
        f"VALID={valid} ({info.get('reason', '')}) "
        f"area={info.get('area', 0)}px "
        f"rot={info.get('rotation_angle', 0.0):.1f}deg"
    )
    cv2.putText(overlay, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, edge_color, 2, cv2.LINE_AA)
    return overlay


# ---------------------------------------------------------------------------
# Public Pure Preprocessing Function (RGB Only)
# ---------------------------------------------------------------------------


def preprocess(
    color: np.ndarray,
    bg_color: np.ndarray | None = None,
    cfg: PreprocessConfig = PreprocessConfig(),
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Pure RGB background-subtraction Duplo preprocessor.

    Args:
        color: `HxWx3` `uint8` photo containing the Duplo block (RGB or BGR per `cfg.COLOR_FORMAT`).
        bg_color: `HxWx3` `uint8` empty-background reference image computed from the background
            photo set via `compute_background_model()`. If `None`, estimates background from the
            image border.
        cfg: Frozen `PreprocessConfig`.

    Returns:
        `(processed_rgb, mask, info)` where `processed_rgb` is `(OUT_SIZE, OUT_SIZE, 3)` RGB `uint8`
        and `mask` is `(OUT_SIZE, OUT_SIZE)` binary `{0, 255}` `uint8`.
    """
    if cfg.COLOR_FORMAT.upper().strip() == "RGB" and isinstance(color, np.ndarray) and color.ndim == 3 and color.shape[2] == 3 and color.dtype == np.uint8:
        color_rgb = color
    else:
        color_rgb = _normalize_color_input(color, cfg)

    if bg_color is None:
        bg_rgb = _synthesize_border_background(color_rgb)
    elif (
        cfg.COLOR_FORMAT.upper().strip() == "RGB"
        and isinstance(bg_color, np.ndarray)
        and bg_color.ndim == 3
        and bg_color.shape[:2] == color_rgb.shape[:2]
        and bg_color.shape[2] == 3
        and bg_color.dtype == np.uint8
    ):
        # Keep identity (`bg_rgb is bg_color`) so `_get_prepared_bg` hits O(1) cache on realtime frames!
        bg_rgb = bg_color
    else:
        bg_rgb = _normalize_color_input(bg_color, cfg)
        if bg_rgb.shape[:2] != color_rgb.shape[:2]:
            bg_rgb = cv2.resize(
                bg_rgb, (color_rgb.shape[1], color_rgb.shape[0]), interpolation=cv2.INTER_AREA
            )

    out_size = int(cfg.OUT_SIZE)
    empty_rgb = np.full((out_size, out_size, 3), np.array(cfg.BG_FILL, dtype=np.uint8), dtype=np.uint8)
    empty_mask = np.zeros((out_size, out_size), dtype=np.uint8)
    h_img, w_img, _ = color_rgb.shape

    def _fail(
        reason: str,
        fmask: np.ndarray | None = None,
        area: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        m = fmask if fmask is not None else np.zeros((h_img, w_img), dtype=np.uint8)
        bbox = cv2.boundingRect(m) if np.any(m) else (0, 0, 0, 0)
        info_d: dict[str, Any] = {
            "valid": False,
            "reason": reason,
            "bbox": tuple(int(v) for v in bbox),
            "crop_box": (0, 0, 0, 0),
            "area": int(area),
            "rotation_angle": 0.0,
            "full_mask": m,
            "debug_overlay": None,
        }
        if cfg.MAKE_DEBUG_OVERLAY:
            info_d["debug_overlay"] = create_debug_overlay(color_rgb, m, info_d)
        return empty_rgb, empty_mask, info_d

    # Fast pyramid segmentation for high-res HD/4K cameras (> MAX_SEG_DIM), while cropping at full sharpness!
    max_dim = max(h_img, w_img)
    if cfg.MAX_SEG_DIM and cfg.MAX_SEG_DIM > 0 and max_dim > cfg.MAX_SEG_DIM:
        seg_scale = float(cfg.MAX_SEG_DIM) / float(max_dim)
        sw = max(64, int(round(w_img * seg_scale)))
        sh = max(64, int(round(h_img * seg_scale)))
        color_seg = cv2.resize(color_rgb, (sw, sh), interpolation=cv2.INTER_AREA)
        bg_seg = cv2.resize(bg_rgb, (sw, sh), interpolation=cv2.INTER_AREA)
        seg_cfg = replace(
            cfg,
            MIN_AREA_PX=max(64, int(round(cfg.MIN_AREA_PX * seg_scale * seg_scale))),
            MAX_AREA_PX=max(256, int(round(cfg.MAX_AREA_PX * seg_scale * seg_scale))),
        )
        mask_seg, comp_area_seg = _segment_duplo_from_background(color_seg, bg_seg, seg_cfg)
        if comp_area_seg == 0:
            return _fail("no_component")
        mask_up = cv2.resize(mask_seg, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
        mask_u8 = np.where(mask_up >= 128, 255, 0).astype(np.uint8)
        comp_area = int(np.count_nonzero(mask_u8))
    else:
        mask_u8, comp_area = _segment_duplo_from_background(color_rgb, bg_rgb, cfg)
    if comp_area == 0:
        return _fail("no_component")
    if comp_area < cfg.MIN_AREA_PX:
        return _fail("area_too_small", fmask=mask_u8, area=comp_area)

    # Safety margin dilation so studs, edges, and marker marks near the edge are never cut off
    if cfg.DILATE_PX > 0:
        dil_kernel = _ellipse_kernel(2 * int(cfg.DILATE_PX) + 1)
        mask_u8 = cv2.dilate(mask_u8, dil_kernel, iterations=1)
        mask_u8 = _fill_interior_holes(mask_u8)

    final_area = int(np.count_nonzero(mask_u8))
    if final_area > cfg.MAX_AREA_PX:
        return _fail("area_too_large", fmask=mask_u8, area=final_area)
    if cfg.REJECT_BORDER_TOUCH and _touches_image_border(mask_u8):
        return _fail("touches_border", fmask=mask_u8, area=final_area)

    fg_rgb = _apply_background_fill(color_rgb, mask_u8, cfg)
    if cfg.CANONICALIZE_ROTATION:
        processed_rgb, cropped_mask, rot_angle, crop_box = _canonicalize_and_crop(
            fg_rgb, mask_u8, cfg
        )
    else:
        processed_rgb, cropped_mask, crop_box = _square_crop_and_resize(fg_rgb, mask_u8, cfg)
        rot_angle = 0.0

    processed_rgb, glare_mask = remove_specular_glare(processed_rgb, cropped_mask, cfg)

    bbox = cv2.boundingRect(mask_u8)
    info: dict[str, Any] = {
        "valid": True,
        "reason": "ok",
        "bbox": tuple(int(v) for v in bbox),
        "crop_box": tuple(int(v) for v in crop_box),
        "area": final_area,
        "rotation_angle": float(rot_angle),
        "glare_pixels_removed": int(np.count_nonzero(glare_mask > 0)),
        "full_mask": mask_u8,
        "debug_overlay": None,
    }
    if cfg.MAKE_DEBUG_OVERLAY:
        info["debug_overlay"] = create_debug_overlay(color_rgb, mask_u8, info)
    return processed_rgb, cropped_mask, info


# ---------------------------------------------------------------------------
# Batch Dataset Processing & Acceptance Test Helpers
# ---------------------------------------------------------------------------


def discover_rgb_images(folder: Path) -> list[tuple[Path, str]]:
    """Find RGB photos in `folder` (or `<folder>/color` / `<folder>/rgb`)."""
    for sub_name in ("color", "rgb"):
        sub = folder / sub_name
        if sub.is_dir():
            imgs = [
                (p, p.stem)
                for p in sorted(sub.iterdir())
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ]
            if imgs:
                return imgs

    return [
        (p, p.stem)
        for p in sorted(folder.iterdir())
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and not p.name.endswith("_depth.png")
    ]


def get_rotations_rgb(processed_rgb: np.ndarray, angles: Sequence[int] = (0, 180)) -> list[np.ndarray]:
    outputs: list[np.ndarray] = []
    for deg in angles:
        norm_deg = int(deg) % 360
        if norm_deg % 90 != 0:
            raise ValueError(f"Only multiples of 90 degrees are lossless; got {deg}.")
        k = (360 - norm_deg) // 90
        outputs.append(np.ascontiguousarray(np.rot90(processed_rgb, k=k % 4)))
    return outputs


def process_dataset_batch(
    input_dir: str | Path,
    output_dir: str | Path,
    view: str = "mat",
    bg_path_or_dir: str | Path | None = None,
    default_split: str = "train/good",
    cfg: PreprocessConfig = PreprocessConfig(),
    save_masks: bool = True,
    save_debug: bool = True,
) -> dict[str, Any]:
    """Batch-subtract background from Duplo photos in `input_dir` and write 256x256 crops to `output_dir`.

    Automatically looks for background photos in:
    1. Explicit `bg_path_or_dir` (file or folder of background photos)
    2. `<input_dir>/background/`
    3. `data/calibration/<view>/` or `data/calibration/background_<view>.png`
    """
    in_root = Path(input_dir)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    run_cfg = replace(cfg, COLOR_FORMAT="RGB", MAKE_DEBUG_OVERLAY=bool(save_debug))

    # Resolve background reference photo set
    bg_rgb: np.ndarray | None = None
    candidates: list[Path] = []
    if bg_path_or_dir is not None:
        candidates.append(Path(bg_path_or_dir))
    candidates.extend(
        [
            in_root / "background",
            Path("data/calibration") / f"background_{view}.png",
            Path("data/calibration") / f"background_{view}.npy",
            Path("data/calibration") / view,
            Path("data/calibration/mat_A"),
        ]
    )
    for cand in candidates:
        if cand.is_file() or (cand.is_dir() and discover_rgb_images(cand)):
            bg_rgb = load_background_model(cand, color_format="RGB")
            break

    # Discover splits (excluding `background/`)
    split_dirs: list[tuple[Path, str]] = []
    if discover_rgb_images(in_root) and in_root.name != "background":
        split_dirs.append((in_root, default_split))
    else:
        for sub in sorted(p for p in in_root.rglob("*") if p.is_dir()):
            if "background" in sub.parts or sub.name in {"color", "rgb", "depth"}:
                continue
            if discover_rgb_images(sub):
                split_dirs.append((sub, sub.relative_to(in_root).as_posix()))

    if not split_dirs:
        raise FileNotFoundError(f"No Duplo photos found under {in_root}")

    summary: dict[str, Any] = {
        "view": view,
        "output_dir": str(out_root),
        "used_background_reference": bg_rgb is not None,
        "total_inputs": 0,
        "valid_saved": 0,
        "rejected": 0,
        "rejection_reasons": {},
        "splits": {},
    }

    for folder_path, split_rel in split_dirs:
        out_img_dir = out_root / split_rel
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_mask_dir = (out_root / "masks" / split_rel) if save_masks else None
        out_dbg_dir = (out_root / "debug" / split_rel) if save_debug else None
        if out_mask_dir is not None:
            out_mask_dir.mkdir(parents=True, exist_ok=True)
        if out_dbg_dir is not None:
            out_dbg_dir.mkdir(parents=True, exist_ok=True)

        split_valid, split_rejected = 0, 0
        for c_path, stem in discover_rgb_images(folder_path):
            summary["total_inputs"] += 1
            c_bgr = cv2.imread(str(c_path), cv2.IMREAD_COLOR)
            if c_bgr is None:
                summary["rejected"] += 1
                split_rejected += 1
                continue
            c_rgb = cv2.cvtColor(c_bgr, cv2.COLOR_BGR2RGB)
            proc_rgb, proc_mask, info = preprocess(c_rgb, bg_rgb, run_cfg)

            if out_dbg_dir is not None and info["debug_overlay"] is not None:
                cv2.imwrite(
                    str(out_dbg_dir / f"{stem}_debug.png"),
                    cv2.cvtColor(info["debug_overlay"], cv2.COLOR_RGB2BGR),
                )
            if not info["valid"]:
                r = str(info["reason"])
                summary["rejected"] += 1
                split_rejected += 1
                summary["rejection_reasons"][r] = summary["rejection_reasons"].get(r, 0) + 1
                logger.warning("Rejected %s/%s: %s", split_rel, stem, r)
                continue

            cv2.imwrite(str(out_img_dir / f"{stem}.png"), cv2.cvtColor(proc_rgb, cv2.COLOR_RGB2BGR))
            if out_mask_dir is not None:
                cv2.imwrite(str(out_mask_dir / f"{stem}.png"), proc_mask)
            summary["valid_saved"] += 1
            split_valid += 1

        summary["splits"][split_rel] = {"valid": split_valid, "rejected": split_rejected}

    run_cfg.to_json(out_root / "preprocess_config.json")
    (out_root / "preprocess_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def run_acceptance_test(
    frames: Sequence[tuple[np.ndarray, str]],
    bg_color: np.ndarray,
    output_dir: str | Path,
    cfg: PreprocessConfig = PreprocessConfig(),
    same_pose: bool = True,
    max_area_variation_pct: float = 5.0,
) -> dict[str, Any]:
    """Run RGB background subtraction across ~20 lighting conditions and verify mask stability."""
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    eval_cfg = replace(cfg, COLOR_FORMAT="RGB", MAKE_DEBUG_OVERLAY=True)

    areas: list[int] = []
    valid_flags: list[bool] = []
    tiles: list[np.ndarray] = []

    for idx, (color_rgb, name) in enumerate(frames):
        proc_rgb, proc_mask, info = preprocess(color_rgb, bg_color, eval_cfg)
        valid_flags.append(bool(info["valid"]))
        if info["valid"]:
            areas.append(int(info["area"]))
        cv2.imwrite(str(out_root / f"{idx:02d}_{name}_crop.png"), cv2.cvtColor(proc_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_root / f"{idx:02d}_{name}_mask.png"), proc_mask)
        if info["debug_overlay"] is not None:
            dbg = info["debug_overlay"]
            cv2.imwrite(str(out_root / f"{idx:02d}_{name}_debug.png"), cv2.cvtColor(dbg, cv2.COLOR_RGB2BGR))
            dbg_thumb = cv2.resize(dbg, (cfg.OUT_SIZE, cfg.OUT_SIZE), interpolation=cv2.INTER_AREA)
            tiles.append(np.hstack([dbg_thumb, proc_rgb]))

    if tiles:
        cols = 4
        rows = (len(tiles) + cols - 1) // cols
        th, tw, _ = tiles[0].shape
        sheet = np.full((rows * th, cols * tw, 3), 40, dtype=np.uint8)
        for i, tile in enumerate(tiles):
            r, c = divmod(i, cols)
            sheet[r * th : (r + 1) * th, c * tw : (c + 1) * tw] = tile
        cv2.imwrite(str(out_root / "contact_sheet.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

    mean_area = float(np.mean(areas)) if areas else 0.0
    spread_pct = (100.0 * float(max(areas) - min(areas)) / mean_area) if areas else float("inf")
    cv_pct = (100.0 * float(np.std(areas)) / mean_area) if areas else float("inf")
    all_valid = len(valid_flags) > 0 and all(valid_flags)
    passed = all_valid and ((not same_pose) or (spread_pct <= max_area_variation_pct))

    report = {
        "num_frames": len(frames),
        "num_valid": int(sum(valid_flags)),
        "all_valid": all_valid,
        "mean_area_px": mean_area,
        "min_area_px": int(min(areas)) if areas else 0,
        "max_area_px": int(max(areas)) if areas else 0,
        "area_max_min_spread_pct": spread_pct,
        "area_cv_pct": cv_pct,
        "max_allowed_variation_pct": max_area_variation_pct,
        "passed": passed,
    }
    (out_root / "acceptance_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report
