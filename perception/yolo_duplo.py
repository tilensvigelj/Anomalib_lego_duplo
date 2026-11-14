#!/usr/bin/env python3
"""Roboflow Serverless YOLO Object Detection (`duplo-model/1`) integration for Duplo bricks.

Uses `inference_sdk` (>=1.5.0) with strict `Authorization: Bearer` header transport
(`api_key_transport="header"`, never in the query string) and loads the API key from
the `ROBOFLOW_API_KEY` environment variable (or a local `.env` file).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import threading
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from inference_sdk import InferenceConfiguration, InferenceHTTPClient

logger = logging.getLogger(__name__)

DEFAULT_ROBOFLOW_API_URL = "https://serverless.roboflow.com"
DEFAULT_DUPLO_MODEL_ID = "duplo-model/1"
ENV_API_KEY_NAMES = ("ROBOFLOW_API_KEY",)


@dataclass(frozen=True)
class YoloDetection:
    """Single Duplo bounding-box prediction in pixel coordinates `(x1, y1, x2, y2)`."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_name: str
    raw: dict[str, Any]

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def xywh(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.width, self.height)


_ENV_LOADED = False
_CLIENT_LOCK = threading.Lock()
_CLIENT_CACHE: dict[tuple[str, str, float | None, float | None], InferenceHTTPClient] = {}


def _load_dotenv_if_present() -> None:
    """Load `ROBOFLOW_API_KEY` from `.env` in cwd or project root if not already in `os.environ`."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True

    proj_root = Path(__file__).resolve().parents[1]
    for env_path in (Path(".env"), proj_root / ".env"):
        if not env_path.is_file():
            continue
        try:
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[len("export ") :].strip()
                k, v = line.split("=", 1)
                key = k.strip()
                val = v.strip().strip('"').strip("'")
                if key and val and key not in os.environ:
                    os.environ[key] = val
        except OSError:
            continue


def get_roboflow_api_key(explicit_key: str | None = None) -> str | None:
    """Resolve the Roboflow API key from `explicit_key`, `os.environ`, or `.env`."""
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()
    for name in ENV_API_KEY_NAMES:
        val = os.environ.get(name, "").strip()
        if val:
            return val
    _load_dotenv_if_present()
    for name in ENV_API_KEY_NAMES:
        val = os.environ.get(name, "").strip()
        if val:
            return val
    return None


def get_roboflow_client(
    api_key: str | None = None,
    api_url: str = DEFAULT_ROBOFLOW_API_URL,
    confidence_threshold: float | None = None,
    iou_threshold: float | None = None,
) -> InferenceHTTPClient:
    """Return a cached thread-safe `InferenceHTTPClient` configured with `api_key_transport='header'`."""
    resolved_key = get_roboflow_api_key(api_key)
    if not resolved_key:
        raise RuntimeError(
            "Roboflow API key not found. Set the ROBOFLOW_API_KEY environment variable "
            "(from https://app.roboflow.com/settings/api) or place ROBOFLOW_API_KEY=... in .env."
        )

    conf_val = float(confidence_threshold) if confidence_threshold is not None else None
    iou_val = float(iou_threshold) if iou_threshold is not None else None
    cache_key = (api_url.rstrip("/"), resolved_key, conf_val, iou_val)

    with _CLIENT_LOCK:
        cached = _CLIENT_CACHE.get(cache_key)
        if cached is not None:
            return cached

        cfg = InferenceConfiguration(
            api_key_transport="header",  # Strictly Authorization: Bearer header, never query string!
            confidence_threshold=conf_val,
            iou_threshold=iou_val,
        )
        client = InferenceHTTPClient(
            api_url=api_url,
            api_key=resolved_key,
        ).configure(cfg)
        if len(_CLIENT_CACHE) >= 8:
            _CLIENT_CACHE.clear()
        _CLIENT_CACHE[cache_key] = client
        return client


def infer_duplo_image(
    image: str | Path | np.ndarray,
    model_id: str = DEFAULT_DUPLO_MODEL_ID,
    color_format: str = "RGB",
    confidence_threshold: float | None = 0.25,
    iou_threshold: float | None = 0.45,
    api_key: str | None = None,
    api_url: str = DEFAULT_ROBOFLOW_API_URL,
) -> dict[str, Any]:
    """Run Roboflow Serverless inference (`duplo-model/1`) on a file path, URL, or numpy image.

    Note: `inference_sdk` encodes `np.ndarray` inputs via `cv2.imencode('.jpg', image)` (which
    expects BGR channel order), so `RGB` arrays are converted to `BGR` before transmission.
    """
    client = get_roboflow_client(
        api_key=api_key,
        api_url=api_url,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
    )

    if isinstance(image, Path):
        infer_input: str | np.ndarray = str(image)
    elif isinstance(image, str):
        infer_input = image
    elif isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 image array for YOLO inference, got shape={image.shape}.")
        if color_format.upper().strip() == "RGB":
            infer_input = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            infer_input = image
    else:
        raise TypeError(f"Unsupported image input type: {type(image)!r}")

    result = client.infer(infer_input, model_id=model_id)
    if not isinstance(result, dict):
        raise RuntimeError(f"Unexpected Roboflow response type: {type(result)!r}")
    return result


def parse_yolo_predictions(
    result: Mapping[str, Any],
    image_hw: tuple[int, int] | None = None,
    min_confidence: float = 0.25,
    class_filter: Sequence[str] | None = None,
) -> list[YoloDetection]:
    """Convert Roboflow JSON `predictions` (`x, y, width, height`) into clamped `YoloDetection` boxes."""
    img_meta = result.get("image") if isinstance(result.get("image"), Mapping) else {}
    if image_hw is not None:
        h_img, w_img = int(image_hw[0]), int(image_hw[1])
    else:
        h_img = int(img_meta.get("height", 0) or 0)
        w_img = int(img_meta.get("width", 0) or 0)

    allowed_classes = (
        {c.strip().lower() for c in class_filter if c.strip()}
        if class_filter
        else None
    )

    detections: list[YoloDetection] = []
    raw_preds = result.get("predictions", [])
    if not isinstance(raw_preds, Sequence):
        return detections

    for item in raw_preds:
        if not isinstance(item, Mapping):
            continue
        conf = float(item.get("confidence", 0.0))
        if conf < float(min_confidence):
            continue
        cls_name = str(item.get("class", "duplo"))
        if allowed_classes and cls_name.strip().lower() not in allowed_classes:
            continue

        cx = float(item.get("x", 0.0))
        cy = float(item.get("y", 0.0))
        bw = float(item.get("width", 0.0))
        bh = float(item.get("height", 0.0))
        if bw <= 1.0 or bh <= 1.0:
            continue

        x1 = int(np.floor(cx - 0.5 * bw))
        y1 = int(np.floor(cy - 0.5 * bh))
        x2 = int(np.ceil(cx + 0.5 * bw))
        y2 = int(np.ceil(cy + 0.5 * bh))

        if w_img > 0 and h_img > 0:
            x1 = max(0, min(w_img - 1, x1))
            y1 = max(0, min(h_img - 1, y1))
            x2 = max(x1 + 1, min(w_img, x2))
            y2 = max(y1 + 1, min(h_img, y2))

        detections.append(
            YoloDetection(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                confidence=conf,
                class_name=cls_name,
                raw=dict(item),
            )
        )

    detections.sort(key=lambda d: (d.confidence, d.area), reverse=True)
    return detections


def detect_best_duplo_box(
    image: str | Path | np.ndarray,
    model_id: str = DEFAULT_DUPLO_MODEL_ID,
    color_format: str = "RGB",
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    class_filter: Sequence[str] | None = None,
    api_key: str | None = None,
    api_url: str = DEFAULT_ROBOFLOW_API_URL,
) -> tuple[YoloDetection | None, list[YoloDetection], dict[str, Any]]:
    """Run `duplo-model/1` and return `(best_detection, all_detections, raw_json)`."""
    img_hw: tuple[int, int] | None = None
    if isinstance(image, np.ndarray) and image.ndim >= 2:
        img_hw = (int(image.shape[0]), int(image.shape[1]))

    raw_json = infer_duplo_image(
        image=image,
        model_id=model_id,
        color_format=color_format,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        api_key=api_key,
        api_url=api_url,
    )
    dets = parse_yolo_predictions(
        raw_json,
        image_hw=img_hw,
        min_confidence=confidence_threshold,
        class_filter=class_filter,
    )
    best = dets[0] if dets else None
    return best, dets, raw_json


def draw_yolo_detections(
    image_rgb: np.ndarray,
    detections: Sequence[YoloDetection],
    best: YoloDetection | None = None,
) -> np.ndarray:
    """Render YOLO bounding boxes on an RGB image copy for inspection/debugging."""
    vis = image_rgb.copy()
    for det in detections:
        is_best = best is not None and det.xyxy == best.xyxy
        col = (255, 215, 0) if is_best else (160, 160, 255)
        thick = 2 if is_best else 1
        cv2.rectangle(vis, (det.x1, det.y1), (det.x2, det.y2), col, thick)
        txt = f"{det.class_name} {det.confidence:.2f}"
        cv2.putText(
            vis,
            txt,
            (det.x1, max(16, det.y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            col,
            1,
            cv2.LINE_AA,
        )
    return vis


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Roboflow Serverless 'duplo-model/1' Object Detection on an image."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to local image or image URL to analyze.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_DUPLO_MODEL_ID,
        help=f"Roboflow model identifier (default: {DEFAULT_DUPLO_MODEL_ID}).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum detection confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_ROBOFLOW_API_URL,
        help=f"Roboflow Inference API URL (default: {DEFAULT_ROBOFLOW_API_URL}).",
    )
    parser.add_argument(
        "--save-vis",
        default=None,
        help="Optional output PNG path with YOLO bounding boxes drawn.",
    )
    args = parser.parse_args(argv)

    best, dets, raw_json = detect_best_duplo_box(
        image=args.image,
        model_id=args.model_id,
        confidence_threshold=args.confidence,
        api_url=args.api_url,
    )
    print(json.dumps(raw_json, indent=2))

    if args.save_vis and Path(args.image).is_file():
        bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            vis = draw_yolo_detections(rgb, dets, best=best)
            out_p = Path(args.save_vis)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_p), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            logger.info("Saved YOLO visualization -> %s", out_p)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
