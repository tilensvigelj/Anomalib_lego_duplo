"""Shared configuration loader for `configs/defect_config.json` and `configs/rules.yaml`."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import yaml

DEFAULT_DEFECT_CONFIG = Path("configs/defect_config.json")
DEFAULT_RULES_YAML = Path("configs/rules.yaml")


@dataclass
class ModelConfig:
    model_type: str = "patchcore"
    version: int = 1
    image_sensitivity: float = 0.50
    pixel_sensitivity: float = 0.50
    openvino_precision: str = "fp16"
    color_dev_chroma_thresh: float = 18.0
    color_dev_dark_thresh: float = 35.0
    min_defect_area_px: int = 6


def load_model_config(config_path: str | Path = DEFAULT_DEFECT_CONFIG) -> ModelConfig:
    path = Path(config_path)
    if not path.is_file():
        return ModelConfig()
    raw = json.loads(path.read_text(encoding="utf-8"))
    params = raw.get("model")
    if not isinstance(params, Mapping):
        params = raw.get("views", {}).get("mat", {})
    valid = {f.name for f in ModelConfig.__dataclass_fields__.values()}
    return ModelConfig(**{k: v for k, v in params.items() if k in valid})


def parse_threshold_value(val: Any, default: float = 0.5) -> float:
    """Extract a single positive threshold float from scalar or dict config entries."""
    if isinstance(val, (int, float)):
        v = float(val)
        return v if v > 0.0 else float(default)
    if isinstance(val, Mapping):
        for key in ("threshold", "high", "low"):
            if key in val and float(val[key]) > 0.0:
                return float(val[key])
    return float(default)


def load_rules(rules_path: str | Path = DEFAULT_RULES_YAML) -> dict[str, Any]:
    path = Path(rules_path)
    if not path.is_file():
        return {
            "verdict_to_task": {
                "perfect": "put the brick in the blue output bin",
                "defective": "put the brick in the pink bin",
            },
            "threshold": 0.50,
            "anomaly_device": "CPU",
        }
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def update_rules_threshold(
    threshold: float,
    rules_path: str | Path = DEFAULT_RULES_YAML,
    model_name: str | None = None,
    update_default: bool = True,
) -> None:
    """Update single `threshold` and optional `model_thresholds.<model_name>` in `rules.yaml`."""
    path = Path(rules_path)
    rules = load_rules(path)
    rules.pop("thresholds", None)
    rules.pop("max_extra_looks", None)

    t_val = round(float(threshold), 6)
    if update_default or "threshold" not in rules:
        rules["threshold"] = t_val
    if model_name:
        model_map = rules.get("model_thresholds")
        if not isinstance(model_map, dict):
            model_map = {}
        model_map[str(model_name)] = t_val
        rules["model_thresholds"] = model_map

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(rules, sort_keys=False), encoding="utf-8")


def save_export_calibration(
    export_dir: str | Path,
    threshold: float,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Persist per-model calibrated single `threshold` inside `anomaly/exports/<export_name>/calibration.json`."""
    out_path = Path(export_dir) / "calibration.json"
    payload: dict[str, Any] = {
        "threshold": round(float(threshold), 6),
    }
    if extra:
        payload.update(extra)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def load_export_calibration(export_dir: str | Path) -> float | None:
    """Load single `threshold` from `anomaly/exports/<export_name>/calibration.json` if available."""
    cal_path = Path(export_dir) / "calibration.json"
    if not cal_path.is_file():
        return None
    try:
        raw = json.loads(cal_path.read_text(encoding="utf-8"))
        val = parse_threshold_value(raw, default=0.0)
        return val if val > 0.0 else None
    except Exception:
        return None
