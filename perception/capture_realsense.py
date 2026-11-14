#!/usr/bin/env python3
"""RGB photo capture utility for single-camera background photo sets & Duplo photos (`data/raw/mat/`).

Supports both standard OpenCV webcams (`--camera-index`) and RealSense RGB streams (`--serial`).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .duplo_rgbd_preprocess import calibrate_background_from_folder


def capture_rgb_frames(
    out_folder: Path,
    count: int = 10,
    interval_sec: float = 0.5,
    prefix: str = "img",
    camera_index: int = 0,
    serial: str | None = None,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
) -> Path:
    out_folder.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())

    if serial:
        import pyrealsense2 as rs  # UNVERIFIED locally without camera attached

        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(str(serial))
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        pipeline.start(cfg)
        try:
            for _ in range(15):
                pipeline.wait_for_frames()
            for i in range(count):
                frames = pipeline.wait_for_frames()
                c_frame = frames.get_color_frame()
                if not c_frame:
                    continue
                bgr = np.asanyarray(c_frame.get_data())
                out_path = out_folder / f"{prefix}_{ts}_{i:04d}.png"
                cv2.imwrite(str(out_path), bgr)
                print(f"[{i + 1}/{count}] Saved {out_path}")
                if i + 1 < count and interval_sec > 0:
                    time.sleep(interval_sec)
        finally:
            pipeline.stop()
        return out_folder

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    try:
        for _ in range(10):
            cap.read()
        for i in range(count):
            ok, bgr = cap.read()
            if not ok or bgr is None:
                raise RuntimeError(f"Failed to read frame from camera index {camera_index}")
            out_path = out_folder / f"{prefix}_{ts}_{i:04d}.png"
            cv2.imwrite(str(out_path), bgr)
            print(f"[{i + 1}/{count}] Saved {out_path}")
            if i + 1 < count and interval_sec > 0:
                time.sleep(interval_sec)
    finally:
        cap.release()
    return out_folder


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture single-camera RGB background & Duplo photos.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_bg = sub.add_parser("background", help="Capture empty-background photos (no Duplo block).")
    p_bg.add_argument("--raw-root", default="data/raw/mat", help="Raw camera dataset directory.")
    p_bg.add_argument("--count", type=int, default=15, help="Number of empty background photos.")
    p_bg.add_argument("--interval", type=float, default=0.2)
    p_bg.add_argument("--camera-index", type=int, default=0)
    p_bg.add_argument("--serial", default=None, help="Optional RealSense serial number.")

    p_cap = sub.add_parser("capture", help="Capture Duplo RGB photos into data/raw/mat/<split>/")
    p_cap.add_argument("--raw-root", default="data/raw/mat", help="Raw camera dataset directory.")
    p_cap.add_argument("--split", default="train/good", help="train/good, test/good, or test/defect")
    p_cap.add_argument("--count", type=int, default=10)
    p_cap.add_argument("--interval", type=float, default=0.8)
    p_cap.add_argument("--prefix", default="duplo")
    p_cap.add_argument("--camera-index", type=int, default=0)
    p_cap.add_argument("--serial", default=None, help="Optional RealSense serial number.")

    args = parser.parse_args(argv)
    if args.cmd == "background":
        bg_dir = Path(args.raw_root) / "background"
        capture_rgb_frames(
            out_folder=bg_dir,
            count=args.count,
            interval_sec=args.interval,
            prefix="bg",
            camera_index=args.camera_index,
            serial=args.serial,
        )
        out_ref = Path("data/calibration/background_mat.png")
        calibrate_background_from_folder(bg_dir, output_path=out_ref)
        print(f"Computed median background reference -> {out_ref}")
        return 0

    if args.cmd == "capture":
        target = Path(args.raw_root) / args.split
        capture_rgb_frames(
            out_folder=target,
            count=args.count,
            interval_sec=args.interval,
            prefix=args.prefix,
            camera_index=args.camera_index,
            serial=args.serial,
        )
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
