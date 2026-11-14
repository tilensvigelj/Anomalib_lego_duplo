#!/usr/bin/env python3
"""CLI to subtract background from single-camera raw RGB Duplo photos (`data/raw/mat/`) into `data/processed/mat/`."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Sequence

from .duplo_rgbd_preprocess import (
    PreprocessConfig,
    calibrate_background_from_folder,
    process_dataset_batch,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Subtract background photo set from Duplo RGB photos into data/processed/mat."
    )
    parser.add_argument("--raw-dir", default="data/raw/mat", help="Input raw photo folder (default: data/raw/mat).")
    parser.add_argument("--out-dir", default="data/processed/mat", help="Output crop folder (default: data/processed/mat).")
    parser.add_argument(
        "--bg-dir",
        default=None,
        help="Folder of empty-background photos without the Duplo block (defaults to <raw-dir>/background).",
    )
    parser.add_argument(
        "--save-bg",
        default="data/calibration/background_mat.png",
        help="Path to save the computed median background PNG.",
    )
    parser.add_argument(
        "--config",
        default="configs/defect_config.json",
        help="Config JSON containing 'preprocess' parameters.",
    )
    parser.add_argument(
        "--no-rotation",
        action="store_true",
        help="Disable minAreaRect horizontal rotation canonicalization.",
    )
    parser.add_argument(
        "--allow-border-touch",
        action="store_true",
        help="Allow masks that touch the image border (useful for tight close-up crops).",
    )
    args = parser.parse_args(argv)

    cfg = (
        PreprocessConfig.from_json(args.config)
        if Path(args.config).is_file()
        else PreprocessConfig()
    )
    if args.no_rotation:
        cfg = replace(cfg, CANONICALIZE_ROTATION=False)
    if args.allow_border_touch:
        cfg = replace(cfg, REJECT_BORDER_TOUCH=False)

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    bg_source: Path | None = Path(args.bg_dir) if args.bg_dir else None
    if bg_source is None and (raw_dir / "background").is_dir():
        bg_source = raw_dir / "background"

    if bg_source is not None and bg_source.is_dir():
        save_bg_path = Path(args.save_bg)
        try:
            calibrate_background_from_folder(bg_source, output_path=save_bg_path)
            bg_source = save_bg_path
        except FileNotFoundError:
            pass

    summary = process_dataset_batch(
        input_dir=raw_dir,
        output_dir=out_dir,
        bg_path_or_dir=bg_source,
        cfg=cfg,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
