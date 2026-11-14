#!/usr/bin/env python3
"""CLI & visual inspector for evaluating a single-camera Duplo image (`'perfect'` vs `'defective'`).

Examples:
    # Evaluate a raw or preprocessed Duplo photo and save a visual heatmap panel:
    python -m anomaly.evaluate \
        --image data/raw/mat/test/defect/sample.png \
        --save-vis anomaly/results/eval_vis.png

    # Print ONLY 'perfect' or 'defective' (exact VLA output string):
    python -m anomaly.evaluate --image data/raw/mat/test/good/sample.png --verdict-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from .detector import BrickAnomalyEvaluator


def _colorize_map(amap: np.ndarray | None, size: int = 256) -> np.ndarray:
    if amap is None or amap.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    arr = np.asarray(amap, dtype=np.float32)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.shape[:2] != (size, size):
        arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_LINEAR)
    lo, hi = float(np.min(arr)), float(np.max(arr))
    norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0) if (hi - lo) > 1e-6 else np.clip(arr, 0.0, 1.0)
    bgr = cv2.applyColorMap(np.round(norm * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def render_evaluation_panel(eval_result: dict[str, Any], out_path: str | Path) -> Path:
    """Save `[256x256 Crop | Heatmap | Blended Overlay]` + binary `'perfect'`/`'defective'` banner."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sz = 256

    crop = eval_result.get("crop_rgb")
    if crop is None:
        crop = np.full((sz, sz, 3), 128, dtype=np.uint8)
    elif crop.shape[:2] != (sz, sz):
        crop = cv2.resize(crop, (sz, sz), interpolation=cv2.INTER_AREA)

    heat = _colorize_map(eval_result.get("anomaly_map"), size=sz)
    blend = cv2.addWeighted(crop, 0.58, heat, 0.42, 0.0)
    strip = np.hstack([crop, heat, blend])

    verdict = str(eval_result.get("verdict", "defective"))
    banner_col = (35, 165, 65) if verdict == "perfect" else (210, 45, 55)
    header = np.full((64, strip.shape[1], 3), (28, 30, 36), dtype=np.uint8)
    cv2.rectangle(header, (0, 0), (strip.shape[1] - 1, 63), banner_col, 2)

    title = f"VLA OUTPUT: {verdict.upper()}   |   norm_score={eval_result.get('normalized_score', 0.0):.3f} (thresh=1.000)"
    sub = f"model={eval_result.get('model', '')}   device={eval_result.get('device', '')}   latency={eval_result.get('latency_ms', 0.0):.1f}ms"
    cv2.putText(header, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, banner_col, 2, cv2.LINE_AA)
    cv2.putText(header, sub, (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1, cv2.LINE_AA)

    panel = np.vstack([header, strip])
    cv2.imwrite(str(out), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a single-camera Duplo image ('perfect' | 'defective').")
    parser.add_argument("--image", required=True, help="Path to raw or 256x256 RGB Duplo image.")
    parser.add_argument(
        "--model",
        default="auto",
        help="Export name(s) under anomaly/exports/ (default: 'auto' runs PatchCore + PaDiM in parallel).",
    )
    parser.add_argument("--single-model", action="store_true", help="Run single model instead of ensemble.")
    parser.add_argument("--exports-root", default="./anomaly/exports")
    parser.add_argument("--rules", default="./configs/rules.yaml")
    parser.add_argument("--config", default="./configs/defect_config.json")
    parser.add_argument("--calibration-dir", default="./data/calibration")
    parser.add_argument("--device", default=None)
    parser.add_argument("--force-crop", action="store_true")
    parser.add_argument("--verdict-only", action="store_true", help="Print ONLY 'perfect' or 'defective'.")
    parser.add_argument("--save-vis", default=None, help="Optional path to save visual inspection PNG.")
    args = parser.parse_args(argv)

    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")
    color_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    evaluator = BrickAnomalyEvaluator(
        export_spec=args.model,
        exports_root=args.exports_root,
        rules_path=args.rules,
        defect_cfg_path=args.config,
        calibration_dir=args.calibration_dir,
        device=args.device,
        enable_ensemble=not args.single_model,
    )

    need_vis = bool(args.save_vis)
    res = evaluator.evaluate(
        color_rgb=color_rgb,
        force_crop=args.force_crop,
        include_arrays=need_vis,
    )
    if need_vis:
        vis_out = render_evaluation_panel(res, args.save_vis)
        res["saved_vis"] = str(vis_out)

    for k in ("crop_rgb", "anomaly_map", "debug_overlay"):
        res.pop(k, None)

    if args.verdict_only:
        print(res["verdict"])
    else:
        print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
