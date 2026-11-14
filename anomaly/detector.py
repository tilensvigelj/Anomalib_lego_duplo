"""Single-camera OpenVINO anomaly detector (`PatchCore` + `PaDiM` parallel ensemble).

Outputs a strict binary verdict:
- `"perfect"`   (`normalized_score < 1.0`, i.e. `score < threshold`)
- `"defective"` (`normalized_score >= 1.0`, i.e. `score >= threshold`, or invalid preprocess crop)
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import replace
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Literal, Sequence

from anomalib.deploy import OpenVINOInferencer
import cv2
import numpy as np
import openvino as ov

from perception.augment_hue import compute_color_deviation
from perception.duplo_rgbd_preprocess import (
    PreprocessConfig,
    _get_prepared_bg,
    load_background_model,
    preprocess,
)

from .config import (
    load_export_calibration,
    load_model_config,
    load_rules,
    parse_threshold_value,
)

BinaryVerdict = Literal["perfect", "defective"]
NORM_DEFECT_THRESHOLD = 1.0
ENSEMBLE_FAMILIES = ("patchcore", "padim")
_VERSION_RE = re.compile(r"_v(\d+)(?:_|$)")
_SPLIT_SPEC_RE = re.compile(r"[+,]")


def _model_family(export_name: str) -> str:
    low = export_name.lower()
    for fam in ("mh_patchcore", "patchcore", "padim", "efficient_ad", "found_ad"):
        if low.startswith(fam):
            return fam
    return low.split("_")[0]


def _export_version(export_name: str) -> int:
    m = _VERSION_RE.search(export_name)
    return int(m.group(1)) if m else 0


def discover_exports(
    exports_root: str | Path,
    requested: str | Sequence[str] | None = "auto",
    enable_ensemble: bool = True,
) -> list[str]:
    """Resolve one or more OpenVINO export names (`PatchCore` + `PaDiM`) for the single camera.

    - If `requested` is `"a+b"` or `"a,b"` or a list, returns those explicit exports.
    - If `requested` is `None` or `"auto"` and `enable_ensemble=True`, automatically selects
      the highest-version `patchcore` AND highest-version `padim` exports so both models
      run together in parallel.
    """
    exp_root = Path(exports_root)
    if not exp_root.is_dir():
        return []

    if isinstance(requested, (list, tuple)):
        explicit = [str(x).strip() for x in requested if str(x).strip()]
        return explicit

    if isinstance(requested, str):
        req = requested.strip()
        if req and req.lower() != "auto":
            parts = [p.strip() for p in _SPLIT_SPEC_RE.split(req) if p.strip()]
            return parts

    valid_dirs = [
        d.name
        for d in exp_root.iterdir()
        if d.is_dir() and (d / "weights" / "openvino" / "model.xml").is_file()
    ]
    if not valid_dirs:
        return []

    if not enable_ensemble:
        valid_dirs.sort(
            key=lambda n: (
                1 if _model_family(n) == "patchcore" else 0,
                _export_version(n),
                n,
            )
        )
        return [valid_dirs[-1]]

    chosen: list[str] = []
    for fam in ENSEMBLE_FAMILIES:
        fam_cands = [n for n in valid_dirs if _model_family(n) == fam]
        if fam_cands:
            fam_cands.sort(key=lambda n: (_export_version(n), n))
            chosen.append(fam_cands[-1])

    if chosen:
        return chosen

    valid_dirs.sort(key=lambda n: (_export_version(n), n))
    return [valid_dirs[-1]]


def resolve_effective_threshold(threshold: float) -> float:
    """Resolve positive single decision threshold (`score >= threshold` -> `'defective'`)."""
    t = float(threshold)
    return t if t > 0.0 else 0.50


def normalize_anomaly_score(raw_score: float, threshold: float) -> float:
    """Map raw model score so `raw_score == threshold` equals `1.0` (`>= 1.0` is `'defective'`)."""
    t = resolve_effective_threshold(threshold)
    return float(max(0.0, float(raw_score) / max(t, 1e-6)))


def encode_heatmap_png_b64(anomaly_map: np.ndarray) -> str:
    """Convert float anomaly map to a compact colorized PNG encoded as base64."""
    amap = np.asarray(anomaly_map, dtype=np.float32)
    if amap.ndim > 2:
        amap = np.squeeze(amap)
    lo, hi = float(np.min(amap)), float(np.max(amap))
    if hi - lo > 1e-6:
        norm = np.clip((amap - lo) / (hi - lo), 0.0, 1.0)
    else:
        norm = np.clip(amap, 0.0, 1.0)
    u8 = np.round(norm * 255.0).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    ok, buf = cv2.imencode(".png", colored_bgr)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


_SHARED_OV_CORE: ov.Core | None = None
_SHARED_OV_LOCK = threading.Lock()


def _get_shared_ov_core(cache_dir: str | Path | None = None) -> ov.Core:
    """Return process-wide shared `ov.Core()` so parallel models share one TBB task arena."""
    global _SHARED_OV_CORE
    with _SHARED_OV_LOCK:
        if _SHARED_OV_CORE is None:
            _SHARED_OV_CORE = ov.Core()
        if cache_dir is not None:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            try:
                _SHARED_OV_CORE.set_property({"CACHE_DIR": str(cache_dir)})
            except Exception:
                pass
        return _SHARED_OV_CORE


def _prep_input_tensor(crop_rgb: np.ndarray) -> np.ndarray:
    """Convert `HxWx3` RGB image into contiguous `[1, 3, 256, 256]` float32 `[0, 1]` tensor."""
    if crop_rgb.shape[:2] != (256, 256):
        crop_rgb = cv2.resize(crop_rgb, (256, 256), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(OpenVINOInferencer.pre_process(crop_rgb), dtype=np.float32)


class SingleModelDetector:
    """Wraps one exported OpenVINO model (`PatchCore` or `PaDiM`) with async C++ InferRequest support."""

    def __init__(
        self,
        xml_path: str | Path,
        model_name: str,
        preferred_device: str = "NPU",
        cache_dir: str | Path = "./anomaly/ov_cache",
        chroma_thresh: float = 18.0,
        dark_thresh: float = 35.0,
        min_defect_area_px: int = 6,
        cpu_threads: int | None = None,
    ) -> None:
        self.xml_path = Path(xml_path)
        self.model_name = model_name
        self.chroma_thresh = chroma_thresh
        self.dark_thresh = dark_thresh
        self.min_defect_area_px = min_defect_area_px

        core = _get_shared_ov_core(cache_dir)
        ov_model = core.read_model(str(self.xml_path))
        available = set(core.available_devices)

        candidates: list[str] = [preferred_device]
        if preferred_device == "NPU":
            candidates.extend(["HETERO:NPU,CPU", "GPU", "CPU"])
        elif preferred_device != "CPU":
            candidates.append("CPU")

        last_err: Exception | None = None
        self.active_device = "CPU"
        self.compiled_model: ov.CompiledModel | None = None
        self.input_name: str = "input"
        self._output_names: list[str] = []

        for dev in candidates:
            base_dev = dev.split(":")[-1].split(",")[0]
            if dev not in {"AUTO", "CPU"} and base_dev not in available:
                continue
            compile_cfg: dict[str, Any] = {}
            if dev == "CPU":
                compile_cfg["PERFORMANCE_HINT"] = "LATENCY"
                compile_cfg["NUM_STREAMS"] = "1"
                if cpu_threads is not None and cpu_threads > 0:
                    compile_cfg["INFERENCE_NUM_THREADS"] = int(cpu_threads)
                    compile_cfg["ENABLE_CPU_PINNING"] = "NO"
            try:
                try:
                    cmodel = core.compile_model(model=ov_model, device_name=dev, config=compile_cfg)
                except Exception:
                    cmodel = core.compile_model(model=ov_model, device_name=dev)

                self.compiled_model = cmodel
                self.input_name = cmodel.input(0).any_name
                self._output_names = [
                    (next(iter(out.names)) if out.names else f"out_{idx}")
                    for idx, out in enumerate(cmodel.outputs)
                ]
                self._infer_req = cmodel.create_infer_request()
                self._req_lock = threading.Lock()
                dummy = _prep_input_tensor(np.full((256, 256, 3), 128, dtype=np.uint8))
                handle = self.start_async_tensor(dummy)
                _ = self.finish_async(handle)
                self.active_device = dev
                break
            except Exception as exc:  # pragma: no cover - NPU fallback guard
                last_err = exc
                self.compiled_model = None
                continue

        if self.compiled_model is None:
            raise RuntimeError(f"Failed to load {self.xml_path} on {candidates}: {last_err}")

    @staticmethod
    def _parse_pred_dict(pred_dict: Mapping[str, Any]) -> tuple[float, np.ndarray]:
        """Extract continuous `(pred_score, anomaly_map)`, ignoring binary `pred_label` / `pred_mask`."""
        score: float | None = None
        amap: np.ndarray | None = None
        for name, val in pred_dict.items():
            arr = np.asarray(val).squeeze()
            lname = name.lower()
            if "score" in lname and "label" not in lname:
                score = float(arr.reshape(-1)[0])
            elif "map" in lname and "mask" not in lname:
                amap = arr.astype(np.float32)

        if amap is None:
            candidates_2d = [
                np.asarray(v).squeeze()
                for k, v in pred_dict.items()
                if "mask" not in k.lower()
                and np.asarray(v).squeeze().ndim == 2
                and np.asarray(v).dtype.kind == "f"
            ]
            amap = candidates_2d[0].astype(np.float32) if candidates_2d else np.zeros((256, 256), dtype=np.float32)
        if score is None:
            candidates_0d = [
                float(np.asarray(v).reshape(-1)[0])
                for k, v in pred_dict.items()
                if "label" not in k.lower()
                and np.asarray(v).size == 1
                and np.asarray(v).dtype.kind == "f"
            ]
            score = candidates_0d[0] if candidates_0d else float(np.max(amap))
        return score, amap

    def start_async_tensor(self, inp_tensor: np.ndarray) -> tuple[ov.InferRequest, bool, float]:
        """Launch non-blocking C++ OpenVINO inference on `inp_tensor` (`[1, 3, 256, 256]` float32)."""
        assert self.compiled_model is not None
        t0 = time.perf_counter()
        owns_primary = self._req_lock.acquire(blocking=False)
        req = self._infer_req if owns_primary else self.compiled_model.create_infer_request()
        req.start_async({self.input_name: inp_tensor}, share_inputs=True)
        return req, owns_primary, t0

    def finish_async(self, handle: tuple[ov.InferRequest, bool, float]) -> tuple[float, np.ndarray, float]:
        """Wait for async C++ OpenVINO inference to complete and return `(score, amap, elapsed_ms)`."""
        req, owns_primary, t0 = handle
        try:
            req.wait()
            pred_dict = {
                name: req.get_output_tensor(idx).data.copy()
                for idx, name in enumerate(self._output_names)
            }
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        finally:
            if owns_primary:
                self._req_lock.release()
        score, amap = self._parse_pred_dict(pred_dict)
        return score, amap, elapsed_ms

    def _predict_single(self, crop_rgb: np.ndarray) -> tuple[float, np.ndarray]:
        handle = self.start_async_tensor(_prep_input_tensor(crop_rgb))
        score, amap, _ = self.finish_async(handle)
        return score, amap

    def score_crop(
        self,
        crop_rgb: np.ndarray,
        mask_u8: np.ndarray | None,
        threshold: float,
        return_map: bool = False,
        include_heatmap_b64: bool = False,
    ) -> dict[str, Any]:
        """Score a 256x256 RGB crop against single `threshold` -> `'perfect'` or `'defective'`."""
        t0 = time.perf_counter()
        inp_tensor = _prep_input_tensor(crop_rgb)
        handle = self.start_async_tensor(inp_tensor)
        color_check = compute_color_deviation(
            crop_rgb,
            mask_u8=mask_u8,
            chroma_thresh=self.chroma_thresh,
            dark_thresh=self.dark_thresh,
            min_region_px=self.min_defect_area_px,
        )
        score, amap, _ = self.finish_async(handle)

        thresh_eff = resolve_effective_threshold(threshold)
        norm_score = normalize_anomaly_score(score, thresh_eff)
        verdict: BinaryVerdict = "defective" if score >= thresh_eff else "perfect"

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        out: dict[str, Any] = {
            "valid": True,
            "score": round(float(score), 4),
            "normalized_score": round(float(norm_score), 4),
            "threshold": round(float(thresh_eff), 6),
            "verdict": verdict,
            "model": self.model_name,
            "device": self.active_device,
            "latency_ms": latency_ms,
            "color_deviation": color_check,
        }
        if include_heatmap_b64:
            out["heatmap"] = encode_heatmap_png_b64(amap)
        if return_map:
            out["anomaly_map"] = amap
        return out


def fuse_ensemble_predictions(
    members: Sequence[Mapping[str, Any]],
    color_check: Mapping[str, Any],
) -> tuple[float, BinaryVerdict]:
    """Combine `PatchCore` + `PaDiM` single-threshold predictions into `'perfect'` or `'defective'`.

    - `PatchCore` (weight `0.65`) is the high-precision local nearest-neighbor primary detector.
    - `PaDiM` (weight `0.35`) is the spatial Mahalanobis secondary detector.
    - Returns `'defective'` when:
      1. Both models flag `'defective'`, OR
      2. Primary (`PatchCore`) flags `'defective'`, OR
      3. Weighted normalized score `>= 1.0`, OR
      4. Secondary (`PaDiM`) flags `'defective'` AND primary (`PatchCore`) also shows elevated score (`>= 0.75`).
    - Otherwise returns `'perfect'` (preventing `PaDiM` clean-edge spikes or stud/number Lab blobs from
      falsely rejecting clean bricks).
    """
    _ = color_check
    if not members:
        return 1.0, "defective"
    if len(members) == 1:
        m0 = members[0]
        ns0 = float(m0["normalized_score"])
        v0: BinaryVerdict = "defective" if str(m0["verdict"]) == "defective" else "perfect"
        return ns0, v0

    raw_weights: list[float] = []
    for m in members:
        fam = _model_family(str(m.get("model", "")))
        raw_weights.append(0.65 if fam in {"patchcore", "mh_patchcore"} else 0.35)
    w_sum = sum(raw_weights) or 1.0
    weights = [w / w_sum for w in raw_weights]

    norms = [float(m["normalized_score"]) for m in members]
    capped_norms = [min(s, 1.5) for s in norms]
    verdicts = [str(m["verdict"]) for m in members]
    weighted_norm = float(sum(w * s for w, s in zip(weights, capped_norms)))

    primary_idx = next(
        (i for i, m in enumerate(members) if _model_family(str(m.get("model", ""))) in {"patchcore", "mh_patchcore"}),
        0,
    )
    primary_norm = norms[primary_idx]
    primary_verdict = verdicts[primary_idx]
    num_defective = sum(1 for v in verdicts if v == "defective")

    if num_defective >= 2 or primary_verdict == "defective" or weighted_norm >= NORM_DEFECT_THRESHOLD:
        def_max = max((s for s, v in zip(norms, verdicts) if v == "defective"), default=weighted_norm)
        return max(weighted_norm, def_max, NORM_DEFECT_THRESHOLD), "defective"

    if num_defective == 1 and primary_norm >= 0.75:
        return max(weighted_norm, NORM_DEFECT_THRESHOLD), "defective"

    return min(weighted_norm, 0.9999), "perfect"


class EnsembleAnomalyDetector:
    """Runs `PatchCore` + `PaDiM` IN PARALLEL on the single camera crop and returns `'perfect'` or `'defective'`."""

    def __init__(
        self,
        detectors: Sequence[SingleModelDetector],
        exports_root: str | Path = "./anomaly/exports",
    ) -> None:
        if not detectors:
            raise ValueError("EnsembleAnomalyDetector requires at least one SingleModelDetector.")
        self.detectors = list(detectors)
        self.exports_root = Path(exports_root)
        self.model_name = "+".join(d.model_name for d in self.detectors)
        devices = sorted({d.active_device for d in self.detectors})
        self.active_device = "+".join(devices)
        self.calib_cache: dict[str, float | None] = {
            d.model_name: load_export_calibration(self.exports_root / d.model_name)
            for d in self.detectors
        }

    def _resolve_member_threshold(
        self,
        model_name: str,
        default_threshold: float,
        model_thresholds: Mapping[str, Any] | None,
    ) -> float:
        if isinstance(model_thresholds, Mapping) and model_name in model_thresholds:
            val = parse_threshold_value(model_thresholds[model_name], default=0.0)
            if val > 0.0:
                return resolve_effective_threshold(val)
        cal = self.calib_cache.get(model_name)
        if cal is not None and cal > 0.0:
            return resolve_effective_threshold(cal)
        return resolve_effective_threshold(default_threshold)

    def score_crop(
        self,
        crop_rgb: np.ndarray,
        mask_u8: np.ndarray | None,
        threshold: float,
        return_map: bool = False,
        include_heatmap_b64: bool = False,
        model_thresholds: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run all ensemble models (`PatchCore` + `PaDiM`) concurrently on `crop_rgb`."""
        if len(self.detectors) == 1:
            d0 = self.detectors[0]
            t0_thresh = self._resolve_member_threshold(d0.model_name, threshold, model_thresholds)
            single = d0.score_crop(
                crop_rgb=crop_rgb,
                mask_u8=mask_u8,
                threshold=t0_thresh,
                return_map=return_map,
                include_heatmap_b64=include_heatmap_b64,
            )
            single["parallel_execution"] = False
            single["ensemble_size"] = 1
            single["ensemble_members"] = [
                {
                    "model": d0.model_name,
                    "device": d0.active_device,
                    "score": single["score"],
                    "normalized_score": single["normalized_score"],
                    "threshold": single["threshold"],
                    "verdict": single["verdict"],
                    "latency_ms": single["latency_ms"],
                }
            ]
            return single

        t0 = time.perf_counter()
        d_ref = self.detectors[0]

        # 1. Pre-normalize `[1, 3, 256, 256]` float32 tensor ONCE for all models.
        inp_tensor = _prep_input_tensor(crop_rgb)
        # 2. Kick off ALL OpenVINO models (`PatchCore` + `PaDiM`) simultaneously in C++ (`start_async`).
        async_handles = [det.start_async_tensor(inp_tensor) for det in self.detectors]
        # 3. Run Lab `compute_color_deviation` concurrently in Python while OpenVINO executes in C++.
        color_check = compute_color_deviation(
            crop_rgb,
            mask_u8=mask_u8,
            chroma_thresh=d_ref.chroma_thresh,
            dark_thresh=d_ref.dark_thresh,
            min_region_px=d_ref.min_defect_area_px,
        )
        # 4. Collect completed C++ predictions.
        raw_results = [det.finish_async(h) for det, h in zip(self.detectors, async_handles)]

        members_info: list[dict[str, Any]] = []
        norm_maps: list[tuple[float, np.ndarray]] = []

        for det, (raw_s, amap_i, mlat) in zip(self.detectors, raw_results):
            thresh_i = self._resolve_member_threshold(det.model_name, threshold, model_thresholds)
            ns_i = normalize_anomaly_score(raw_s, thresh_i)
            v_i: BinaryVerdict = "defective" if raw_s >= thresh_i else "perfect"

            members_info.append(
                {
                    "model": det.model_name,
                    "device": det.active_device,
                    "score": round(float(raw_s), 4),
                    "normalized_score": round(float(ns_i), 4),
                    "threshold": round(float(thresh_i), 6),
                    "verdict": v_i,
                    "latency_ms": mlat,
                }
            )
            w_i = 0.65 if _model_family(det.model_name) in {"patchcore", "mh_patchcore"} else 0.35
            if return_map or include_heatmap_b64:
                lo_m, hi_m = float(np.min(amap_i)), float(np.max(amap_i))
                scaled_m = (amap_i - lo_m) / (hi_m - lo_m) if (hi_m - lo_m) > 1e-6 else np.clip(amap_i, 0.0, 1.0)
                norm_maps.append((w_i, scaled_m.astype(np.float32)))

        fused_norm, fused_verdict = fuse_ensemble_predictions(members_info, color_check)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        out: dict[str, Any] = {
            "valid": True,
            "score": round(float(fused_norm), 4),
            "normalized_score": round(float(fused_norm), 4),
            "threshold": NORM_DEFECT_THRESHOLD,
            "verdict": fused_verdict,
            "model": self.model_name,
            "device": self.active_device,
            "parallel_execution": True,
            "ensemble_size": len(self.detectors),
            "ensemble_members": members_info,
            "latency_ms": latency_ms,
            "color_deviation": color_check,
        }

        if return_map or include_heatmap_b64:
            w_total = sum(w for w, _ in norm_maps) or 1.0
            blended_map = sum((w / w_total) * m for w, m in norm_maps)
            if include_heatmap_b64:
                out["heatmap"] = encode_heatmap_png_b64(blended_map)
            if return_map:
                out["anomaly_map"] = blended_map

        return out


class BrickAnomalyEvaluator:
    """Single-camera Duplo brick evaluator returning `'perfect'` or `'defective'`."""

    def __init__(
        self,
        export_spec: str | Sequence[str] | None = "auto",
        exports_root: str | Path = "./anomaly/exports",
        rules_path: str | Path = "./configs/rules.yaml",
        defect_cfg_path: str | Path = "./configs/defect_config.json",
        calibration_dir: str | Path = "./data/calibration",
        cache_dir: str | Path = "./anomaly/ov_cache",
        device: str | None = None,
        enable_ensemble: bool = True,
    ) -> None:
        self.exports_root = Path(exports_root)
        self.rules_path = Path(rules_path)
        self.defect_cfg_path = Path(defect_cfg_path)
        self.prep_cfg = (
            PreprocessConfig.from_json(defect_cfg_path)
            if self.defect_cfg_path.is_file()
            else PreprocessConfig()
        )

        rules = load_rules(self.rules_path)
        preferred_device = device or str(rules.get("anomaly_device", "CPU"))
        mcfg = load_model_config(defect_cfg_path)

        names = discover_exports(self.exports_root, requested=export_spec, enable_ensemble=enable_ensemble)
        if not names:
            raise FileNotFoundError(f"No OpenVINO exports found under {self.exports_root}")

        cpu_threads = max(2, (os.cpu_count() or 8) // len(names)) if len(names) > 1 else None
        sub_detectors = [
            SingleModelDetector(
                xml_path=self.exports_root / name / "weights" / "openvino" / "model.xml",
                model_name=name,
                preferred_device=preferred_device,
                cache_dir=cache_dir,
                chroma_thresh=mcfg.color_dev_chroma_thresh,
                dark_thresh=mcfg.color_dev_dark_thresh,
                min_defect_area_px=mcfg.min_defect_area_px,
                cpu_threads=cpu_threads,
            )
            for name in names
        ]
        self.detector = EnsembleAnomalyDetector(sub_detectors, exports_root=self.exports_root)

        cal_root = Path(calibration_dir)
        self.bg_rgb: np.ndarray | None = None
        for cand in (
            cal_root / "background_mat.png",
            cal_root / "background_mat.npy",
            cal_root / "mat_A",
            Path("data/raw/mat/background"),
        ):
            if cand.is_file() or (
                cand.is_dir() and (any(cand.glob("*.png")) or any(cand.glob("*.jpg")))
            ):
                try:
                    self.bg_rgb = load_background_model(cand, color_format="RGB")
                    _get_prepared_bg(self.bg_rgb, max(1, int(self.prep_cfg.BLUR_KERNEL) | 1))
                    break
                except Exception:
                    continue

    def evaluate(
        self,
        color_rgb: np.ndarray,
        mask_u8: np.ndarray | None = None,
        force_crop: bool = False,
        rules: Mapping[str, Any] | None = None,
        include_arrays: bool = False,
    ) -> dict[str, Any]:
        """Preprocess (if full camera frame) and classify brick as `'perfect'` or `'defective'`."""
        active_rules = dict(rules) if rules is not None else load_rules(self.rules_path)
        default_thresh = parse_threshold_value(active_rules.get("threshold", 0.50), default=0.50)

        needs_crop = bool(force_crop) or (
            color_rgb.shape[:2] != (self.prep_cfg.OUT_SIZE, self.prep_cfg.OUT_SIZE)
        )
        crop_rgb = color_rgb
        if needs_crop:
            t_prep = time.perf_counter()
            rt_cfg = replace(self.prep_cfg, MAKE_DEBUG_OVERLAY=bool(include_arrays))
            crop_rgb, mask_u8, info = preprocess(color_rgb, self.bg_rgb, rt_cfg)
            prep_ms = round((time.perf_counter() - t_prep) * 1000.0, 2)
            if not info["valid"]:
                out_fail: dict[str, Any] = {
                    "valid": False,
                    "score": 1.0,
                    "normalized_score": 1.0,
                    "threshold": NORM_DEFECT_THRESHOLD,
                    "verdict": "defective",
                    "model": "preprocess_gate",
                    "device": "CPU",
                    "latency_ms": prep_ms,
                    "preprocess_ms": prep_ms,
                    "reason": str(info["reason"]),
                }
                if include_arrays:
                    out_fail["crop_rgb"] = crop_rgb
                    out_fail["anomaly_map"] = np.zeros(
                        (self.prep_cfg.OUT_SIZE, self.prep_cfg.OUT_SIZE), dtype=np.float32
                    )
                    out_fail["debug_overlay"] = info.get("debug_overlay")
                return out_fail
        else:
            prep_ms = 0.0
            info = {"debug_overlay": None}

        res = self.detector.score_crop(
            crop_rgb=crop_rgb,
            mask_u8=mask_u8,
            threshold=default_thresh,
            return_map=include_arrays,
            include_heatmap_b64=False,
            model_thresholds=active_rules.get("model_thresholds"),
        )
        amap = res.pop("anomaly_map", None)
        res["preprocess_ms"] = prep_ms
        res["latency_ms"] = round(float(res["latency_ms"]) + prep_ms, 2)
        if include_arrays:
            res["crop_rgb"] = crop_rgb
            res["anomaly_map"] = amap
            res["debug_overlay"] = info.get("debug_overlay")
        return res

    def classify(
        self,
        color_rgb: np.ndarray,
        mask_u8: np.ndarray | None = None,
        force_crop: bool = False,
    ) -> BinaryVerdict:
        """Return strictly `'perfect'` or `'defective'`."""
        res = self.evaluate(color_rgb=color_rgb, mask_u8=mask_u8, force_crop=force_crop, include_arrays=False)
        return "defective" if res["verdict"] == "defective" else "perfect"
