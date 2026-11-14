"""Unit and live integration tests for single-camera binary (`'perfect'` | `'defective'`) anomaly detection."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import tempfile
import threading
import unittest

import cv2
import numpy as np
import zmq

from anomaly.config import normalize_model_type
from anomaly.detector import (
    ENSEMBLE_FAMILIES,
    NORM_DEFECT_THRESHOLD,
    discover_exports,
    fuse_ensemble_predictions,
    normalize_anomaly_score,
    required_majority_votes,
    resolve_model_device,
)
from anomaly.evaluate import main as evaluate_cli_main
from anomaly.service import AnomalyZMQService
from anomaly.train import _ensure_efficientad_ood_dir


def _encode_rgb_b64(rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _encode_rgb_png_bytes(rgb: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    return buf.tobytes()


class TestSingleThresholdAndBinaryEnsemble(unittest.TestCase):
    """Verify threshold normalization, Intel NPU/GPU routing, and 2/3 majority voting."""

    def test_single_threshold_normalization(self) -> None:
        # Threshold = 0.025 -> raw 0.025 maps to 1.0 ('defective')
        self.assertAlmostEqual(normalize_anomaly_score(0.0, 0.025), 0.0, places=5)
        self.assertAlmostEqual(normalize_anomaly_score(0.0125, 0.025), 0.5, places=5)
        self.assertAlmostEqual(normalize_anomaly_score(0.025, 0.025), NORM_DEFECT_THRESHOLD, places=5)
        self.assertGreater(normalize_anomaly_score(0.040, 0.025), NORM_DEFECT_THRESHOLD)

        # Heterogeneous thresholds across PatchCore, PaDiM, and EfficientAd map onto identical 1.0 boundary
        s_pc = normalize_anomaly_score(0.1574, 0.1574)
        s_pd = normalize_anomaly_score(0.0508, 0.0508)
        s_ea = normalize_anomaly_score(0.4200, 0.4200)
        self.assertAlmostEqual(s_pc, 1.0, places=5)
        self.assertAlmostEqual(s_pd, 1.0, places=5)
        self.assertAlmostEqual(s_ea, 1.0, places=5)

    def test_intel_npu_and_gpu_device_routing_and_discovery(self) -> None:
        self.assertEqual(ENSEMBLE_FAMILIES, ("patchcore", "padim", "efficient_ad"))
        self.assertEqual(normalize_model_type("efficientaid"), "efficient_ad")
        self.assertEqual(normalize_model_type("efficientad"), "efficient_ad")

        # Default AUTO_INTEL routes EfficientAd -> NPU, PatchCore -> GPU, PaDiM -> GPU
        dev_ea, exp_ea = resolve_model_device("efficient_ad_mat_v1")
        dev_ea_alias, _ = resolve_model_device("efficientaid_mat_v2")
        dev_pc, exp_pc = resolve_model_device("patchcore_mat_v1")
        dev_pd, exp_pd = resolve_model_device("padim_mat_v1")
        self.assertEqual((dev_ea, exp_ea), ("NPU", False))
        self.assertEqual(dev_ea_alias, "NPU")
        self.assertEqual((dev_pc, exp_pc), ("GPU", False))
        self.assertEqual((dev_pd, exp_pd), ("GPU", False))

        # Explicit CLI override forces chosen device
        dev_override, is_exp = resolve_model_device("efficient_ad_mat_v1", requested_device="CPU")
        self.assertEqual((dev_override, is_exp), ("CPU", True))

        # Auto-discovery picks latest version of all 3 families (PatchCore + PaDiM + EfficientAd)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("patchcore_mat_v1", "patchcore_mat_v2", "padim_mat_v1", "efficient_ad_mat_v1"):
                xml_p = root / name / "weights" / "openvino" / "model.xml"
                xml_p.parent.mkdir(parents=True, exist_ok=True)
                xml_p.write_text("<net/>", encoding="utf-8")
            found = discover_exports(root, requested="auto", enable_ensemble=True)
            self.assertEqual(found, ["patchcore_mat_v2", "padim_mat_v1", "efficient_ad_mat_v1"])

            ood_dir = _ensure_efficientad_ood_dir(ood_root=root / "ood_pen", num_images=8)
            self.assertEqual(len(list((ood_dir / "ood").glob("*.png"))), 8)

    def test_two_thirds_majority_voting_across_three_models(self) -> None:
        self.assertEqual(required_majority_votes(3), 2)
        self.assertEqual(required_majority_votes(2), 2)
        self.assertEqual(required_majority_votes(1), 1)

        # 1. 0/3 votes -> 'perfect'
        s_0, v_0 = fuse_ensemble_predictions(
            [
                {"model": "patchcore_mat_v1", "score": 0.08, "threshold": 0.16},     # norm = 0.50 (perfect)
                {"model": "padim_mat_v1", "score": 0.03, "threshold": 0.05},         # norm = 0.60 (perfect)
                {"model": "efficient_ad_mat_v1", "score": 0.18, "threshold": 0.40},  # norm = 0.45 (perfect)
            ]
        )
        self.assertEqual(v_0, "perfect")
        self.assertAlmostEqual(s_0, 0.50, places=4)

        # 2. 1/3 votes (only 1 model spikes >= threshold) -> 'perfect' regardless of which single model spikes!
        for spike_idx in range(3):
            members = [
                {"model": "patchcore_mat_v1", "normalized_score": 0.55, "verdict": "perfect"},
                {"model": "padim_mat_v1", "normalized_score": 0.72, "verdict": "perfect"},
                {"model": "efficient_ad_mat_v1", "normalized_score": 0.61, "verdict": "perfect"},
            ]
            members[spike_idx]["normalized_score"] = 1.85
            members[spike_idx]["verdict"] = "defective"
            s_1, v_1 = fuse_ensemble_predictions(members)
            self.assertEqual(v_1, "perfect", msg=f"1/3 spike on idx={spike_idx} must stay 'perfect'")
            self.assertLess(s_1, 1.0)

        # 3. 2/3 votes (any 2 of the 3 models >= threshold) -> 'defective'
        for clean_idx in range(3):
            members = [
                {"model": "patchcore_mat_v1", "normalized_score": 1.35, "verdict": "defective"},
                {"model": "padim_mat_v1", "normalized_score": 1.15, "verdict": "defective"},
                {"model": "efficient_ad_mat_v1", "normalized_score": 1.60, "verdict": "defective"},
            ]
            members[clean_idx]["normalized_score"] = 0.40
            members[clean_idx]["verdict"] = "perfect"
            s_2, v_2 = fuse_ensemble_predictions(members)
            self.assertEqual(v_2, "defective", msg=f"2/3 votes with clean idx={clean_idx} must be 'defective'")
            self.assertGreaterEqual(s_2, 1.0)

        # 4. 3/3 votes -> 'defective'
        s_3, v_3 = fuse_ensemble_predictions(
            [
                {"model": "patchcore_mat_v1", "normalized_score": 2.10, "verdict": "defective"},
                {"model": "padim_mat_v1", "normalized_score": 1.40, "verdict": "defective"},
                {"model": "efficient_ad_mat_v1", "normalized_score": 1.75, "verdict": "defective"},
            ]
        )
        self.assertEqual(v_3, "defective")
        self.assertAlmostEqual(s_3, 1.75, places=4)


class TestLiveSingleCameraService(unittest.TestCase):
    """End-to-end OpenVINO + ZeroMQ tests confirming strictly `'perfect'` or `'defective'` output."""

    @classmethod
    def setUpClass(cls) -> None:
        exports_dir = Path("anomaly/exports")
        avail = discover_exports(exports_dir, requested="auto", enable_ensemble=True)
        if not avail:
            raise unittest.SkipTest("No OpenVINO model exported in anomaly/exports")
        cls.service = AnomalyZMQService(model_export="auto")

        good_crops = sorted(Path("data/processed/mat/test/good").glob("*.png"))
        if not good_crops:
            good_crops = sorted(Path("data/processed/mat/train/good").glob("*.png"))
        cls.clean_crop_path = good_crops[0]
        cls.clean_rgb = cv2.cvtColor(cv2.imread(str(cls.clean_crop_path)), cv2.COLOR_BGR2RGB)

        # Create an unmistakable high-contrast marker defect across the brick for deterministic service testing
        cls.defect_rgb = cls.clean_rgb.copy()
        cv2.line(cls.defect_rgb, (70, 75), (185, 180), (10, 10, 10), thickness=9)
        cv2.circle(cls.defect_rgb, (128, 128), 18, (220, 20, 20), thickness=-1)

    def test_parallel_ensemble_returns_binary_verdict(self) -> None:
        exports = discover_exports("anomaly/exports", requested="auto", enable_ensemble=True)
        res = self.service.evaluator.evaluate(self.clean_rgb)
        self.assertIn(res["verdict"], {"perfect", "defective"})
        self.assertIn("votes", res)
        self.assertEqual(res["votes"]["rule"], "2/3_majority")
        if len(exports) >= 2:
            self.assertEqual(res["ensemble_size"], len(exports))
            self.assertTrue(res["parallel_execution"])

        def_res = self.service.evaluator.evaluate(self.defect_rgb)
        self.assertEqual(def_res["verdict"], "defective")

    def test_zmq_socket_replies_strictly_plain_string_perfect_or_defective(self) -> None:
        # Always verify wire-payload handler across JSON, raw PNG bytes, base64 text, and invalid bytes
        ans_json = self.service.handle_request(json.dumps({"image": _encode_rgb_b64(self.clean_rgb)}).encode("utf-8"))
        self.assertIsInstance(ans_json, str)
        self.assertIn(ans_json, {"perfect", "defective"})
        self.assertFalse(ans_json.startswith("{"))

        ans_png = self.service.handle_request(_encode_rgb_png_bytes(self.defect_rgb))
        self.assertEqual(ans_png, "defective")

        ans_b64 = self.service.handle_request(_encode_rgb_b64(self.defect_rgb))
        self.assertEqual(ans_b64, "defective")

        ans_empty = self.service.handle_request(b"")
        self.assertEqual(ans_empty, "defective")

        # Also exercise live ZeroMQ REP/REQ socket roundtrip when loopback sockets are permitted by OS/sandbox
        try:
            ctx = zmq.Context()
            rep = ctx.socket(zmq.REP)
            port = rep.bind_to_random_port("tcp://127.0.0.1")
        except zmq.ZMQError:
            return

        def _worker() -> None:
            for _ in range(2):
                raw = rep.recv()
                verdict = self.service.handle_request(raw)
                rep.send_string(verdict)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        req = ctx.socket(zmq.REQ)
        req.connect(f"tcp://127.0.0.1:{port}")
        try:
            req.send_string(json.dumps({"image": _encode_rgb_b64(self.clean_rgb)}))
            s1 = req.recv_string()
            self.assertIn(s1, {"perfect", "defective"})

            req.send(_encode_rgb_png_bytes(self.defect_rgb))
            s2 = req.recv_string()
            self.assertEqual(s2, "defective")
        finally:
            req.close()
            t.join(timeout=5.0)
            rep.close()
            ctx.term()

    def test_cli_evaluate_single_image_and_verdict_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vis_path = Path(tmp) / "eval_vis.png"
            rc = evaluate_cli_main(
                [
                    "--image",
                    str(self.clean_crop_path),
                    "--save-vis",
                    str(vis_path),
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue(vis_path.is_file())

    def test_realtime_raw_camera_photo_through_service(self) -> None:
        raw_goods = sorted(Path("data/raw/mat/test/good").glob("*.png"))
        if raw_goods:
            raw_good_rgb = cv2.cvtColor(cv2.imread(str(raw_goods[0])), cv2.COLOR_BGR2RGB)
            res = self.service.evaluator.evaluate(raw_good_rgb)
            self.assertIn(res["verdict"], {"perfect", "defective"})
            self.assertGreater(res["preprocess_ms"], 0.0)

            # Draw unmistakable marker strokes on the brick center in the raw uncropped camera frame
            h, w, _ = raw_good_rgb.shape
            raw_marked = raw_good_rgb.copy()
            cx, cy = w // 2, h // 2
            cv2.circle(raw_marked, (cx, cy), max(14, min(h, w) // 18), (10, 10, 10), thickness=-1)
            verdict = self.service.handle_request(_encode_rgb_png_bytes(raw_marked))
            self.assertIn(verdict, {"perfect", "defective"})

    def test_service_stop_flag_sets_and_clears(self) -> None:
        self.service.stop()
        self.assertTrue(self.service._stop_event.is_set())
        self.service._stop_event.clear()


if __name__ == "__main__":
    unittest.main()
