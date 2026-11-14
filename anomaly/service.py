#!/usr/bin/env python3
"""ZeroMQ Single-Camera Anomaly Service (`tcp://127.0.0.1:5555`) for VLA integration.

Contract:
- Accepts raw full-camera frames OR 256x256 crops over ZeroMQ `REP` (JSON `{"image": "<base64_png>"}`,
  plain base64 image string, or raw PNG/JPEG bytes), runs realtime background subtraction + cropping
  automatically when given a full-frame camera photo, and evaluates `PatchCore + PaDiM` in parallel.
- Replies with ONLY the plain string `"perfect"` or `"defective"`.
- Cleanly exits within <200ms on `Ctrl+C` (`SIGINT` / `SIGTERM`) on both Windows and Linux.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Mapping
import json
import logging
from pathlib import Path
import signal
import threading
from typing import Any, Sequence

import cv2
import numpy as np
import zmq

from .detector import BinaryVerdict, BrickAnomalyEvaluator

logger = logging.getLogger("anomaly.service")


def _decode_png_b64(b64_str: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    s = b64_str.strip()
    if not s:
        return None
    if "," in s and s.startswith("data:image"):
        s = s.split(",", 1)[1]
    raw = base64.b64decode(s)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, flags)


def _decode_raw_bytes(img_bytes: bytes, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    if not img_bytes:
        return None
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, flags)


class AnomalyZMQService:
    """ZeroMQ `REP` service returning strictly `'perfect'` or `'defective'`."""

    def __init__(
        self,
        model_export: str | Sequence[str] | None = "auto",
        exports_root: str | Path = "./anomaly/exports",
        rules_path: str | Path = "./configs/rules.yaml",
        defect_cfg_path: str | Path = "./configs/defect_config.json",
        calibration_dir: str | Path = "./data/calibration",
        cache_dir: str | Path = "./anomaly/ov_cache",
        enable_ensemble: bool = True,
    ) -> None:
        self.evaluator = BrickAnomalyEvaluator(
            export_spec=model_export,
            exports_root=exports_root,
            rules_path=rules_path,
            defect_cfg_path=defect_cfg_path,
            calibration_dir=calibration_dir,
            cache_dir=cache_dir,
            enable_ensemble=enable_ensemble,
        )
        self.detector = self.evaluator.detector
        self._stop_event = threading.Event()

    def stop(self) -> None:
        """Signal `serve_forever` to exit cleanly on the next poll tick (<200ms)."""
        self._stop_event.set()

    def _parse_payload(
        self, payload: bytes | str | Mapping[str, Any]
    ) -> tuple[np.ndarray | None, np.ndarray | None, bool]:
        """Parse JSON dict, base64 string, or raw PNG/JPEG bytes into `(color_rgb, mask_u8, force_crop)`."""
        if isinstance(payload, Mapping):
            b64_img = str(payload.get("image") or payload.get("frame") or "")
            bgr = _decode_png_b64(b64_img, cv2.IMREAD_COLOR)
            if bgr is None:
                return None, None, False
            mask_u8 = (
                _decode_png_b64(str(payload["mask"]), cv2.IMREAD_GRAYSCALE)
                if payload.get("mask")
                else None
            )
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), mask_u8, bool(payload.get("crop", False))

        if isinstance(payload, bytes):
            stripped = payload.lstrip()
            if stripped.startswith(b"{"):
                req = json.loads(stripped.decode("utf-8"))
                if isinstance(req, Mapping):
                    return self._parse_payload(req)
                return None, None, False
            bgr_raw = _decode_raw_bytes(payload, cv2.IMREAD_COLOR)
            if bgr_raw is not None:
                return cv2.cvtColor(bgr_raw, cv2.COLOR_BGR2RGB), None, False
            return self._parse_payload(payload.decode("utf-8", errors="ignore"))

        text = str(payload).strip()
        if not text:
            return None, None, False
        if text.startswith("{"):
            req = json.loads(text)
            if isinstance(req, Mapping):
                return self._parse_payload(req)
            return None, None, False
        bgr = _decode_png_b64(text, cv2.IMREAD_COLOR)
        if bgr is None:
            return None, None, False
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), None, False

    def handle_request(self, payload: bytes | str | Mapping[str, Any]) -> BinaryVerdict:
        """Evaluate incoming camera image (raw full frame or 256x256 crop) -> `'perfect'` | `'defective'`."""
        try:
            color_rgb, mask_u8, force_crop = self._parse_payload(payload)
        except Exception:
            logger.exception("Failed to decode request payload")
            return "defective"

        if color_rgb is None:
            return "defective"

        return self.evaluator.classify(color_rgb=color_rgb, mask_u8=mask_u8, force_crop=force_crop)

    def serve_forever(
        self,
        bind_addr: str = "tcp://127.0.0.1:5555",
        poll_timeout_ms: int = 200,
    ) -> None:
        """Run ZeroMQ REP loop with short recv timeout so `Ctrl+C` (`SIGINT`) exits immediately on Windows."""
        self._stop_event.clear()
        prev_sigint = None
        prev_sigterm = None

        if threading.current_thread() is threading.main_thread():
            def _on_signal(signum: int, _frame: Any) -> None:
                logger.info("Received signal %s — stopping Anomaly ZeroMQ service...", signum)
                self._stop_event.set()

            prev_sigint = signal.signal(signal.SIGINT, _on_signal)
            if hasattr(signal, "SIGTERM"):
                prev_sigterm = signal.signal(signal.SIGTERM, _on_signal)

        ctx = zmq.Context()
        sock = ctx.socket(zmq.REP)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, max(50, int(poll_timeout_ms)))
        sock.bind(bind_addr)
        logger.info(
            "Anomaly ZeroMQ service listening on %s (model=%s on %s) -> replies 'perfect' | 'defective' (Ctrl+C to stop)",
            bind_addr,
            self.detector.model_name,
            self.detector.active_device,
        )
        try:
            while not self._stop_event.is_set():
                try:
                    raw_msg = sock.recv()
                except zmq.Again:
                    # Periodic wakeup lets CPython on Windows process Ctrl+C (SIGINT) immediately
                    continue
                except zmq.ZMQError as exc:
                    if self._stop_event.is_set() or exc.errno == zmq.ETERM:
                        break
                    raise

                try:
                    verdict = self.handle_request(raw_msg)
                except Exception:
                    logger.exception("Unhandled error in request evaluation")
                    verdict = "defective"
                sock.send_string(verdict)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt — stopping Anomaly ZeroMQ service...")
        finally:
            sock.close(linger=0)
            ctx.destroy(linger=0)
            if prev_sigint is not None:
                signal.signal(signal.SIGINT, prev_sigint)
            if prev_sigterm is not None and hasattr(signal, "SIGTERM"):
                signal.signal(signal.SIGTERM, prev_sigterm)
            logger.info("Anomaly ZeroMQ service stopped cleanly.")


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Run single-camera ZeroMQ Anomaly Service returning 'perfect' or 'defective'."
    )
    parser.add_argument("--bind", default="tcp://127.0.0.1:5555")
    parser.add_argument(
        "--model",
        default="auto",
        help="Export dir name(s) under anomaly/exports/ (default: 'auto' pairs latest PatchCore + PaDiM)",
    )
    parser.add_argument(
        "--single-model",
        action="store_true",
        help="Disable automatic PatchCore + PaDiM parallel ensemble pairing",
    )
    parser.add_argument("--exports-root", default="./anomaly/exports")
    parser.add_argument("--rules", default="./configs/rules.yaml")
    args = parser.parse_args(argv)

    svc = AnomalyZMQService(
        model_export=args.model,
        exports_root=args.exports_root,
        rules_path=args.rules,
        enable_ensemble=not args.single_model,
    )
    try:
        svc.serve_forever(bind_addr=args.bind)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
