#!/usr/bin/env python3
"""Verify single-camera OpenVINO model(s), benchmark parallel latency, and calibrate single `threshold`.

Binary classification rule:
- `score >= threshold` -> `"defective"`
- `score <  threshold` -> `"perfect"`
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .config import (
    load_model_config,
    load_rules,
    save_export_calibration,
    update_rules_threshold,
)
from .detector import (
    EnsembleAnomalyDetector,
    SingleModelDetector,
    discover_exports,
    required_majority_votes,
    resolve_model_device,
)


def _list_sampled_pngs(folder: Path, max_images: int | None = 300) -> list[Path]:
    if not folder.is_dir():
        return []
    files = sorted(folder.glob("*.png"))
    if max_images and max_images > 0 and len(files) > max_images:
        idx = np.linspace(0, len(files) - 1, max_images, dtype=int)
        return [files[int(i)] for i in idx]
    return files


def _score_folder(
    detector: SingleModelDetector,
    folder: Path,
    max_images: int | None = 300,
) -> tuple[list[float], list[float]]:
    scores: list[float] = []
    latencies: list[float] = []
    for p in _list_sampled_pngs(folder, max_images=max_images):
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = detector.score_crop(rgb, mask_u8=None, threshold=0.5)
        scores.append(float(res["score"]))
        latencies.append(float(res["latency_ms"]))
    return scores, latencies


def calibrate_single_threshold(good_scores: Sequence[float], defect_scores: Sequence[float]) -> tuple[float, float, float | None]:
    """Compute optimal single `threshold` maximizing balanced accuracy (Youden's J = TPR - FPR).

    Robust against single mislabeled or hidden-side outliers in `test/defect` (which previously
    pulled `min(defect_scores)` below most `good` bricks and caused >70% false-defective rate).
    """
    pos_defects = [float(d) for d in defect_scores if float(d) > 1e-5]
    if pos_defects and good_scores:
        g_arr = np.sort(np.asarray(good_scores, dtype=np.float64))
        d_arr = np.sort(np.asarray(pos_defects, dtype=np.float64))

        # If classes are cleanly separable with <= 10% FPR, keep clean midpoint; otherwise maximize Youden's J
        min_def = float(d_arr[0])
        fpr_at_min_def = float(np.mean(g_arr >= min_def))
        if fpr_at_min_def <= 0.10:
            good_below = g_arr[g_arr < min_def]
            rec_thresh = 0.5 * (float(good_below[-1]) + min_def) if good_below.size > 0 else max(min_def * 0.98, 1e-4)
        else:
            all_vals = np.unique(np.concatenate([g_arr, d_arr]))
            if all_vals.size >= 2:
                mids = 0.5 * (all_vals[:-1] + all_vals[1:])
                candidates = np.concatenate([[all_vals[0] * 0.98], mids, [min(all_vals[-1] * 1.02, 1.05)]])
            else:
                candidates = all_vals

            best_obj = -1e9
            rec_thresh = float(np.percentile(g_arr, 90))
            for cand in candidates:
                tpr = float(np.mean(d_arr >= cand))
                tnr = float(np.mean(g_arr < cand))
                # Maximize balanced accuracy 0.5*(TPR + TNR), penalizing >20% false-positive rate on clean bricks
                fpr = 1.0 - tnr
                penalty = 0.35 * max(0.0, fpr - 0.20)
                obj = 0.5 * (tpr + tnr) - penalty
                if obj > best_obj:
                    best_obj = obj
                    rec_thresh = float(cand)

        fpr_at_thresh = float(np.mean(g_arr >= rec_thresh))
        thresh_5fpr = float(np.percentile(g_arr, 95))
        recall_5fpr = float(np.mean(np.asarray(defect_scores, dtype=np.float64) >= thresh_5fpr))
        return round(float(rec_thresh), 6), round(fpr_at_thresh, 4), round(recall_5fpr, 4)

    g_max = float(np.max(good_scores))
    if g_max >= 0.999:
        rec_thresh = min(max(float(np.percentile(good_scores, 98)), 0.50), 0.95)
    else:
        rec_thresh = max(g_max * 1.05, g_max + 1e-4, 0.01)
    return round(float(rec_thresh), 6), 0.0, (0.0 if defect_scores else None)


def evaluate_single_model(
    export_name: str,
    device: str = "AUTO_INTEL",
    processed_dir: str | Path = "./data/processed/mat",
    exports_root: str | Path = "./anomaly/exports",
    cache_dir: str | Path = "./anomaly/ov_cache",
    config_path: str | Path = "./configs/defect_config.json",
    rules_path: str | Path = "./configs/rules.yaml",
    update_rules: bool = False,
    update_default: bool = True,
) -> dict:
    mcfg = load_model_config(config_path)
    rules = load_rules(rules_path)
    xml_path = Path(exports_root) / export_name / "weights" / "openvino" / "model.xml"
    if not xml_path.is_file():
        raise FileNotFoundError(f"OpenVINO IR not found at {xml_path}. Run `python -m anomaly.export` first.")

    detector = SingleModelDetector(
        xml_path=xml_path,
        model_name=export_name,
        preferred_device=device,
        cache_dir=cache_dir,
        chroma_thresh=mcfg.color_dev_chroma_thresh,
        dark_thresh=mcfg.color_dev_dark_thresh,
        min_defect_area_px=mcfg.min_defect_area_px,
        rules=rules,
    )

    base = Path(processed_dir)
    good_scores, good_lat = _score_folder(detector, base / "test" / "good", max_images=300)
    if not good_scores:
        good_scores, good_lat = _score_folder(detector, base / "train" / "good", max_images=300)
    defect_scores, def_lat = _score_folder(detector, base / "test" / "defect", max_images=None)

    all_lat = good_lat + def_lat
    if not good_scores:
        raise RuntimeError(f"No processed images found under {base} to verify!")

    rec_thresh, fpr_at_100, recall_5fpr = calibrate_single_threshold(good_scores, defect_scores)

    save_export_calibration(
        Path(exports_root) / export_name,
        threshold=rec_thresh,
        extra={
            "model": export_name,
            "num_good_scored": len(good_scores),
            "num_defect_scored": len(defect_scores),
            "fpr_at_100pct_recall": fpr_at_100,
        },
    )
    if update_rules:
        update_rules_threshold(
            threshold=rec_thresh,
            rules_path=rules_path,
            model_name=export_name,
            update_default=update_default,
        )

    report = {
        "model": export_name,
        "preferred_device": detector.preferred_device,
        "active_device": detector.active_device,
        "num_good_scored": len(good_scores),
        "num_defect_scored": len(defect_scores),
        "recommended_threshold": rec_thresh,
        "fpr_at_100pct_recall": fpr_at_100,
        "recall_at_5pct_fpr": recall_5fpr,
        "latency_p50_ms": round(float(np.percentile(all_lat, 50)), 2) if all_lat else 0.0,
        "latency_p95_ms": round(float(np.percentile(all_lat, 95)), 2) if all_lat else 0.0,
        "updated_rules_yaml": bool(update_rules),
    }
    print(json.dumps(report, indent=2))
    return report


def evaluate_ensemble_sweep(
    export_spec: str = "auto",
    device: str = "AUTO_INTEL",
    processed_dir: str | Path = "./data/processed/mat",
    exports_root: str | Path = "./anomaly/exports",
    cache_dir: str | Path = "./anomaly/ov_cache",
    config_path: str | Path = "./configs/defect_config.json",
    rules_path: str | Path = "./configs/rules.yaml",
    update_rules: bool = False,
) -> dict:
    """Calibrate each available model (`PatchCore` + `PaDiM` + `EfficientAd`) and verify 2/3 majority voting."""
    names = discover_exports(exports_root, requested=export_spec, enable_ensemble=True)
    if not names:
        raise FileNotFoundError(f"No exported OpenVINO models found under {exports_root}")

    if len(names) == 1:
        return evaluate_single_model(
            export_name=names[0],
            device=device,
            processed_dir=processed_dir,
            exports_root=exports_root,
            cache_dir=cache_dir,
            config_path=config_path,
            rules_path=rules_path,
            update_rules=update_rules,
            update_default=True,
        )

    member_reports: list[dict] = []
    model_thresholds: dict[str, float] = {}
    for idx, name in enumerate(names):
        rep = evaluate_single_model(
            export_name=name,
            device=device,
            processed_dir=processed_dir,
            exports_root=exports_root,
            cache_dir=cache_dir,
            config_path=config_path,
            rules_path=rules_path,
            update_rules=update_rules,
            # Default top-level `threshold` tracks primary model (PatchCore = idx 0)
            update_default=(idx == 0),
        )
        member_reports.append(rep)
        model_thresholds[name] = float(rep["recommended_threshold"])

    mcfg = load_model_config(config_path)
    rules = load_rules(rules_path)
    ens_threads = max(2, (os.cpu_count() or 8) // len(names))
    sub_detectors = [
        SingleModelDetector(
            xml_path=Path(exports_root) / name / "weights" / "openvino" / "model.xml",
            model_name=name,
            preferred_device=device,
            cache_dir=cache_dir,
            chroma_thresh=mcfg.color_dev_chroma_thresh,
            dark_thresh=mcfg.color_dev_dark_thresh,
            min_defect_area_px=mcfg.min_defect_area_px,
            cpu_threads=ens_threads,
            rules=rules,
        )
        for name in names
    ]
    ensemble = EnsembleAnomalyDetector(sub_detectors, exports_root=exports_root)

    base = Path(processed_dir)
    ens_good_verdicts: list[str] = []
    ens_def_verdicts: list[str] = []
    ens_latencies: list[float] = []
    default_t = model_thresholds[names[0]]

    for split_rel, bucket, cap in (
        ("test/good", ens_good_verdicts, 300),
        ("train/good", ens_good_verdicts, 300),
        ("test/defect", ens_def_verdicts, None),
    ):
        folder = base / split_rel
        if not folder.is_dir() or (split_rel == "train/good" and ens_good_verdicts):
            continue
        for p in _list_sampled_pngs(folder, max_images=cap):
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            res = ensemble.score_crop(
                cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                mask_u8=None,
                threshold=default_t,
                model_thresholds=model_thresholds,
            )
            bucket.append(str(res["verdict"]))
            ens_latencies.append(float(res["latency_ms"]))

    false_def = sum(1 for v in ens_good_verdicts if v == "defective")
    caught_def = sum(1 for v in ens_def_verdicts if v == "defective")
    ens_summary = {
        "mode": "ensemble_verification",
        "ensemble_model": ensemble.model_name,
        "ensemble_size": len(names),
        "voting_rule": f"{required_majority_votes(len(names))}/{len(names)}_majority",
        "preferred_devices": ensemble.preferred_device,
        "active_devices": ensemble.active_device,
        "parallel_execution": True,
        "num_good_scored": len(ens_good_verdicts),
        "num_defect_scored": len(ens_def_verdicts),
        "ensemble_perfect_rate_on_good": (
            round(1.0 - (false_def / len(ens_good_verdicts)), 4) if ens_good_verdicts else 1.0
        ),
        "ensemble_false_defective_fpr": (
            round(false_def / len(ens_good_verdicts), 4) if ens_good_verdicts else 0.0
        ),
        "ensemble_defect_recall": (
            round(caught_def / len(ens_def_verdicts), 4) if ens_def_verdicts else None
        ),
        "latency_p50_ms": round(float(np.percentile(ens_latencies, 50)), 2) if ens_latencies else None,
        "latency_p95_ms": round(float(np.percentile(ens_latencies, 95)), 2) if ens_latencies else None,
        "members": member_reports,
    }
    print(json.dumps(ens_summary, indent=2))
    return ens_summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify OpenVINO export(s) & calibrate single binary threshold.")
    parser.add_argument(
        "--export-name",
        default="auto",
        help="Model export name, 'a+b+c' tuple, or 'auto' (default: calibrates & combines PatchCore + PaDiM + EfficientAd)",
    )
    parser.add_argument(
        "--single-model",
        action="store_true",
        help="Disable automatic PatchCore + PaDiM + EfficientAd ensemble calibration when --export-name is 'auto'",
    )
    parser.add_argument(
        "--device",
        default="AUTO_INTEL",
        help="OpenVINO device ('AUTO_INTEL' routes EfficientAd->NPU and PatchCore/PaDiM->GPU; or force 'CPU'/'GPU'/'NPU')",
    )
    parser.add_argument("--processed-dir", default="./data/processed/mat")
    parser.add_argument("--exports-root", default="./anomaly/exports")
    parser.add_argument("--rules", default="./configs/rules.yaml")
    parser.add_argument("--update-rules", action="store_true", help="Write calibrated single threshold into configs/rules.yaml")
    args = parser.parse_args(argv)

    if args.single_model and args.export_name == "auto":
        names = discover_exports(args.exports_root, requested="auto", enable_ensemble=False)
        if not names:
            raise FileNotFoundError(f"No exports found under {args.exports_root}")
        evaluate_single_model(
            export_name=names[0],
            device=args.device,
            processed_dir=args.processed_dir,
            exports_root=args.exports_root,
            rules_path=args.rules,
            update_rules=args.update_rules,
        )
        return 0

    evaluate_ensemble_sweep(
        export_spec=args.export_name,
        device=args.device,
        processed_dir=args.processed_dir,
        exports_root=args.exports_root,
        rules_path=args.rules,
        update_rules=args.update_rules,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
