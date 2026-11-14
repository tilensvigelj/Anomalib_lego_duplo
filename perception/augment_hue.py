#!/usr/bin/env python3
"""Hue-shift augmentation (rotate hue, preserve lightness) + Lab color-deviation second check.

Implements two brick-generalization helpers:
1. `hue_shift_crop`: Generates plausible unseen brick colors from existing `train/good` crops
   while keeping `BG_FILL` neutral gray untouched and preserving CIE/HLS lightness.
2. `compute_color_deviation`: Training-free second opinion comparing each pixel's Lab chroma
   (`a*`, `b*`) and darkness against the brick's median interior color to corroborate marker ink
   on unseen brick colors.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


def hue_shift_crop(
    crop_rgb: np.ndarray,
    mask_u8: np.ndarray | None = None,
    hue_delta_deg: float = 60.0,
    bg_fill: tuple[int, int, int] = (128, 128, 128),
) -> np.ndarray:
    """Rotate hue by `hue_delta_deg` inside the brick mask while keeping lightness constant."""
    if mask_u8 is None:
        diff = np.max(np.abs(crop_rgb.astype(np.int16) - np.array(bg_fill, dtype=np.int16)), axis=2)
        fg = diff > 6
    else:
        fg = mask_u8 > 0

    hls = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HLS).astype(np.int32)
    shift = int(round(hue_delta_deg * 0.5)) % 180
    chromatic_fg = fg & (hls[:, :, 2] >= 25)
    hls[chromatic_fg, 0] = (hls[chromatic_fg, 0] + shift) % 180

    shifted_rgb = cv2.cvtColor(hls.astype(np.uint8), cv2.COLOR_HLS2RGB)
    out_rgb = crop_rgb.copy()
    out_rgb[chromatic_fg] = shifted_rgb[chromatic_fg]
    return out_rgb


def compute_color_deviation(
    crop_rgb: np.ndarray,
    mask_u8: np.ndarray | None = None,
    chroma_thresh: float = 18.0,
    dark_thresh: float = 35.0,
    min_region_px: int = 6,
    bg_fill: tuple[int, int, int] = (128, 128, 128),
) -> dict[str, Any]:
    """Compare interior pixels against the brick's median Lab color to corroborate marker strokes."""
    if mask_u8 is None:
        diff = np.max(np.abs(crop_rgb.astype(np.int16) - np.array(bg_fill, dtype=np.int16)), axis=2)
        fg_u8 = (diff > 8).astype(np.uint8) * 255
    else:
        fg_u8 = np.where(mask_u8 > 0, 255, 0).astype(np.uint8)

    _, _, bw, bh = cv2.boundingRect(fg_u8)
    erode_diam = max(17, int(round(min(bw, bh) * 0.16)) | 1)
    eroded = cv2.erode(
        fg_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_diam, erode_diam)),
        iterations=1,
    )
    interior = eroded > 0
    if int(np.count_nonzero(interior)) < 64:
        return {"flagged": False, "max_blob_px": 0, "deviation_score": 0.0}

    lab = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    med_lab = np.median(lab[interior], axis=0)

    d_l = lab[:, :, 0] - med_lab[0]
    d_a = lab[:, :, 1] - med_lab[1]
    d_b = lab[:, :, 2] - med_lab[2]
    chroma_dev = np.hypot(d_a, d_b)
    dark_dev = np.maximum(0.0, -d_l)

    outlier = interior & ((chroma_dev > chroma_thresh) | (dark_dev > dark_thresh))
    outlier_u8 = (outlier.astype(np.uint8)) * 255
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(outlier_u8, connectivity=8)
    max_blob = int(np.max(stats[1:, cv2.CC_STAT_AREA])) if num_labels > 1 else 0

    raw_strength = float(
        np.percentile(chroma_dev[interior] + 0.5 * dark_dev[interior], 99.0)
    )
    norm_score = float(np.clip(raw_strength / max(1.0, chroma_thresh * 2.0), 0.0, 1.0))
    return {
        "flagged": bool(max_blob >= int(min_region_px)),
        "max_blob_px": max_blob,
        "deviation_score": round(norm_score, 4),
    }


def augment_good_folder(
    processed_dir: Path,
    hue_steps_deg: Sequence[float] = (60.0, 120.0, 180.0, 240.0, 300.0),
) -> int:
    """Create hue-shifted copies inside `data/processed/mat/train/good/`."""
    good_dir = processed_dir / "train" / "good"
    mask_dir = processed_dir / "masks" / "train" / "good"
    if not good_dir.is_dir():
        raise FileNotFoundError(f"Missing folder: {good_dir}")

    originals = [
        p for p in sorted(good_dir.glob("*.png")) if "_hue" not in p.stem
    ]
    created = 0
    for img_path in originals:
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        m_path = mask_dir / img_path.name
        mask_u8 = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE) if m_path.is_file() else None

        for deg in hue_steps_deg:
            aug_rgb = hue_shift_crop(rgb, mask_u8=mask_u8, hue_delta_deg=float(deg))
            out_name = f"{img_path.stem}_hue{int(round(deg)):03d}.png"
            cv2.imwrite(str(good_dir / out_name), cv2.cvtColor(aug_rgb, cv2.COLOR_RGB2BGR))
            if mask_u8 is not None and mask_dir.is_dir():
                cv2.imwrite(str(mask_dir / out_name), mask_u8)
            created += 1
    return created


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hue-shift augment data/processed/mat/train/good.")
    parser.add_argument("--processed-dir", default="data/processed/mat")
    parser.add_argument(
        "--angles",
        type=float,
        nargs="+",
        default=[60.0, 120.0, 180.0, 240.0, 300.0],
        help="Hue rotation offsets in degrees.",
    )
    args = parser.parse_args(argv)
    n = augment_good_folder(Path(args.processed_dir), hue_steps_deg=args.angles)
    print(f"Generated {n} hue-shifted good crops in {Path(args.processed_dir) / 'train/good'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
