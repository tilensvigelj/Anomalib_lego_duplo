#!/usr/bin/env python3
"""Unit & 20-frame cross-lighting tests for RGB background-subtraction Duplo preprocessing."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from perception.augment_hue import compute_color_deviation, hue_shift_crop
from perception.duplo_rgbd_preprocess import (
    PreprocessConfig,
    calibrate_background_from_folder,
    compute_background_model,
    get_rotations_rgb,
    preprocess,
    process_dataset_batch,
    remove_specular_glare,
    run_acceptance_test,
)


def _make_textured_background(
    h: int = 480,
    w: int = 640,
    base_rgb: tuple[int, int, int] = (210, 212, 215),
    illum_gain: tuple[float, float, float] = (1.0, 1.0, 1.0),
    noise_std: float = 2.5,
    rng_seed: int = 0,
) -> np.ndarray:
    """Create a realistic textured background mat image with vignetting and sensor noise."""
    rng = np.random.default_rng(rng_seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Subtle spatial lighting gradient across the table/mat
    vignette = 1.0 - 0.08 * np.hypot((xx - w * 0.5) / w, (yy - h * 0.5) / h)
    bg = np.empty((h, w, 3), dtype=np.float32)
    for c in range(3):
        bg[:, :, c] = base_rgb[c] * vignette * illum_gain[c]
    bg += rng.normal(0.0, noise_std, size=(h, w, 3)).astype(np.float32)
    return np.clip(np.rint(bg), 0, 255).astype(np.uint8)


def _make_synthetic_duplo_rgb_scene(
    h: int = 480,
    w: int = 640,
    center: tuple[int, int] = (320, 240),
    brick_size: tuple[int, int, int] | tuple[int, int] = (140, 70),
    angle_deg: float = 0.0,
    brick_rgb: tuple[int, int, int] = (215, 42, 38),
    base_bg_rgb: tuple[int, int, int] = (210, 212, 215),
    illum_gain: tuple[float, float, float] = (1.0, 1.0, 1.0),
    cast_shadow: bool = True,
    low_contrast_patch: bool = True,
    noise_std: float = 3.0,
    rng_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize an RGB photo of a Duplo block on the background under `illum_gain`."""
    rng = np.random.default_rng(rng_seed)
    img_f = _make_textured_background(
        h=h, w=w, base_rgb=base_bg_rgb, illum_gain=illum_gain, noise_std=0.0, rng_seed=rng_seed
    ).astype(np.float32)

    gt_mask = np.zeros((h, w), dtype=np.uint8)
    bw, bh = int(brick_size[0]), int(brick_size[1])
    rect = ((float(center[0]), float(center[1])), (float(bw), float(bh)), float(angle_deg))
    box = np.int32(np.rint(cv2.boxPoints(rect)))
    cv2.fillConvexPoly(gt_mask, box, 255)

    # Diffuse cast shadow on the mat next to the brick
    if cast_shadow:
        shadow_mask = np.zeros((h, w), dtype=np.uint8)
        shadow_box = box + np.array([[20, 16]], dtype=np.int32)
        cv2.fillConvexPoly(shadow_mask, shadow_box, 255)
        img_f[(shadow_mask > 0) & (gt_mask == 0)] *= 0.82

    # Paint Duplo brick body scaled by current scene illumination
    brick_lit = np.array(brick_rgb, dtype=np.float32) * np.array(illum_gain, dtype=np.float32)
    img_f[gt_mask > 0] = brick_lit

    # Simulate a glare/shaded interior patch to verify continuous-shape recovery
    if low_contrast_patch:
        glare = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(glare, (center[0] - 8, center[1] - 4), 10, 255, -1)
        # Blend partway toward background color inside the brick
        img_f[glare > 0] = 0.55 * img_f[glare > 0] + 0.45 * brick_lit

    # Draw 2x4 studs
    rad = np.radians(angle_deg)
    cos_a, sin_a = float(np.cos(rad)), float(np.sin(rad))
    for r_idx, ly in enumerate((-bh * 0.24, bh * 0.24)):
        for lx in (-bw * 0.36, -bw * 0.12, bw * 0.12, bw * 0.36):
            sx = int(np.rint(center[0] + lx * cos_a - ly * sin_a))
            sy = int(np.rint(center[1] + lx * sin_a + ly * cos_a))
            stud_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(stud_mask, (sx, sy), 9, 255, -1)
            stud_sel = (stud_mask > 0) & (gt_mask > 0)
            factor = 1.16 if r_idx == 0 else 1.06
            img_f[stud_sel] = np.minimum(255.0, brick_lit * factor)

    img_f += rng.normal(0.0, noise_std, size=(h, w, 3)).astype(np.float32)
    color_rgb = np.clip(np.rint(img_f), 0, 255).astype(np.uint8)
    return color_rgb, gt_mask


class TestDuploRGBPreprocess(unittest.TestCase):
    def setUp(self) -> None:
        # Build a background model from 10 empty-scene photos with sensor noise
        bg_photos = [
            _make_textured_background(noise_std=3.5, rng_seed=10 + i)
            for i in range(10)
        ]
        self.bg_model = compute_background_model(bg_photos)

    def test_determinism_and_single_continuous_shape(self) -> None:
        color, gt_mask = _make_synthetic_duplo_rgb_scene(
            illum_gain=(0.82, 0.85, 0.90),  # Cooler/dimmer than background photo set!
            cast_shadow=True,
            low_contrast_patch=True,
            noise_std=4.0,
        )
        c_orig = color.copy()
        bg_orig = self.bg_model.copy()
        cfg = PreprocessConfig()

        rgb1, mask1, info1 = preprocess(color, self.bg_model, cfg)
        rgb2, mask2, info2 = preprocess(color, self.bg_model, cfg)

        # Pure function does not mutate inputs
        np.testing.assert_array_equal(color, c_orig)
        np.testing.assert_array_equal(self.bg_model, bg_orig)

        # Bit-for-bit deterministic
        self.assertTrue(info1["valid"])
        np.testing.assert_array_equal(rgb1, rgb2)
        np.testing.assert_array_equal(mask1, mask2)

        # Must be ONE continuous shape with zero interior holes
        full_mask = info1["full_mask"]
        num_labels, _ = cv2.connectedComponents((full_mask > 0).astype(np.uint8))
        self.assertEqual(num_labels, 2)  # 1 background + 1 single continuous foreground shape!
        recall = float(np.count_nonzero((full_mask > 0) & (gt_mask > 0))) / float(
            np.count_nonzero(gt_mask > 0)
        )
        self.assertGreater(recall, 0.99)

    def test_rotation_and_failure_gates(self) -> None:
        cfg = PreprocessConfig(CANONICALIZE_ROTATION=True)
        c1, _ = _make_synthetic_duplo_rgb_scene(angle_deg=35.0, cast_shadow=False, rng_seed=1)
        c2, _ = _make_synthetic_duplo_rgb_scene(angle_deg=215.0, cast_shadow=False, rng_seed=2)
        _, m1, info1 = preprocess(c1, self.bg_model, cfg)
        _, m2, info2 = preprocess(c2, self.bg_model, cfg)
        self.assertTrue(info1["valid"] and info2["valid"])
        _, _, w1, h1 = cv2.boundingRect(m1)
        _, _, w2, h2 = cv2.boundingRect(m2)
        self.assertGreater(w1, h1 * 1.7)
        self.assertGreater(w2, h2 * 1.7)

        # Empty background photo (even under shifted lighting!) -> valid=False ("no_component")
        empty_shifted = _make_textured_background(illum_gain=(0.75, 0.78, 0.80), noise_std=3.0, rng_seed=99)
        _, _, info_empty = preprocess(empty_shifted, self.bg_model, cfg)
        self.assertFalse(info_empty["valid"])
        self.assertEqual(info_empty["reason"], "no_component")

        # Border-touching brick -> valid=False ("touches_border")
        c_edge, _ = _make_synthetic_duplo_rgb_scene(center=(40, 240), brick_size=(90, 60))
        _, _, info_edge = preprocess(c_edge, self.bg_model, cfg)
        self.assertFalse(info_edge["valid"])
        self.assertEqual(info_edge["reason"], "touches_border")

    def test_hue_augmentation_and_color_deviation(self) -> None:
        color, _ = _make_synthetic_duplo_rgb_scene(cast_shadow=False, low_contrast_patch=False)
        crop, mask, _ = preprocess(color, self.bg_model, PreprocessConfig())

        shifted = hue_shift_crop(crop, mask_u8=mask, hue_delta_deg=120.0)
        np.testing.assert_array_equal(shifted[mask == 0], crop[mask == 0])
        self.assertFalse(np.array_equal(shifted[mask > 0], crop[mask > 0]))

        clean_check = compute_color_deviation(crop, mask_u8=mask)
        self.assertFalse(clean_check["flagged"])

        marked = crop.copy()
        cv2.line(marked, (110, 125), (145, 130), (15, 15, 15), thickness=4)
        defect_check = compute_color_deviation(marked, mask_u8=mask)
        self.assertTrue(defect_check["flagged"])

    def test_cast_and_colored_bounce_shadow_rejection_across_brick_colors(self) -> None:
        """Verify heavy diffuse penumbra + colored interreflection shadows are excluded across brick colors."""
        cfg = PreprocessConfig()
        palette = [
            ("yellow", (220, 185, 55)),
            ("red", (215, 42, 38)),
            ("blue", (35, 95, 205)),
            ("green", (40, 165, 65)),
            ("white", (245, 246, 248)),
            ("charcoal", (32, 34, 36)),
        ]
        for idx, (name, brick_rgb) in enumerate(palette):
            color, gt_mask = _make_synthetic_duplo_rgb_scene(
                angle_deg=28.0,
                brick_rgb=brick_rgb,
                illum_gain=(1.12, 1.10, 1.08),
                cast_shadow=False,
                low_contrast_patch=False,
                noise_std=2.5,
                rng_seed=500 + idx,
            )
            # Inject a realistic diffuse + bounce-tinted shadow lobe next to the brick
            h, w = gt_mask.shape
            M_shift = np.float32([[1.0, 0.0, -24.0], [0.0, 1.0, 22.0]])
            shifted = cv2.warpAffine(gt_mask, M_shift, (w, h))
            penumbra = cv2.GaussianBlur(shifted.astype(np.float32) / 255.0, (0, 0), sigmaX=9.0)
            outside = gt_mask == 0

            img_f = color.astype(np.float32)
            # Darken background up to 42% and tint with bounce light from chromatic bricks
            brick_norm = np.array(brick_rgb, dtype=np.float32) / max(float(max(brick_rgb)), 1.0)
            bounce_tint = 0.72 + 0.28 * brick_norm[None, None, :]
            shadow_scale = 1.0 - 0.42 * penumbra[:, :, None] * (2.0 - bounce_tint)
            img_f[outside] = img_f[outside] * shadow_scale[outside]
            shadowed_rgb = np.clip(np.rint(img_f), 0, 255).astype(np.uint8)

            _, _, info = preprocess(shadowed_rgb, self.bg_model, cfg)
            self.assertTrue(info["valid"], msg=f"Failed validity for brick color={name}: {info['reason']}")
            pred = info["full_mask"] > 0
            gt = gt_mask > 0
            inter = float(np.count_nonzero(pred & gt))
            union = float(np.count_nonzero(pred | gt))
            iou = inter / max(union, 1.0)
            self.assertGreater(iou, 0.88, msg=f"Shadow leaked or brick clipped for {name} (IoU={iou:.3f})")

    def test_differentiate_ground_shadows_from_block_shadows(self) -> None:
        """Self-shaded brick walls/studs must be kept while ground cast & bounce shadows are rejected."""
        cfg = PreprocessConfig.from_json("configs/defect_config.json")
        bg_base = (150, 152, 154)
        bg_model = compute_background_model(
            [_make_textured_background(base_rgb=bg_base, noise_std=2.5, rng_seed=70 + i) for i in range(6)]
        )
        img_f = _make_textured_background(base_rgb=bg_base, noise_std=0.0, rng_seed=88).astype(np.float32)
        h, w, _ = img_f.shape

        # Lit top face + self-shaded left vertical wall + protruding shaded studs
        top_mask = np.zeros((h, w), dtype=np.uint8)
        wall_mask = np.zeros((h, w), dtype=np.uint8)
        stud_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(top_mask, (230, 150), (430, 310), 255, -1)
        cv2.rectangle(wall_mask, (195, 155), (230, 310), 255, -1)
        cv2.circle(stud_mask, (192, 195), 11, 255, -1)
        cv2.circle(stud_mask, (192, 265), 11, 255, -1)
        gt_mask = np.maximum(np.maximum(top_mask, wall_mask), stud_mask)

        # Ground shadows: neutral cast shadow on left + colored overhang bounce shadow on bottom
        cast_sh = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(cast_sh, (135, 165), (195, 330), 255, -1)
        cast_blur = cv2.GaussianBlur(cast_sh.astype(np.float32) / 255.0, (0, 0), sigmaX=7.0)
        img_f[gt_mask == 0] *= (1.0 - 0.25 * cast_blur[:, :, None])[gt_mask == 0]

        overhang_sh = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(overhang_sh, (240, 310), (395, 352), 255, -1)
        oh_blur = cv2.GaussianBlur(overhang_sh.astype(np.float32) / 255.0, (0, 0), sigmaX=6.0)
        oh_scale = 1.0 - oh_blur[:, :, None] * np.array([[[0.38, 0.43, 0.58]]], dtype=np.float32)
        img_f[gt_mask == 0] *= oh_scale[gt_mask == 0]

        # Paint lit yellow top face, self-shaded vertical wall + studs, and a dark marker defect on the wall
        img_f[top_mask > 0] = np.array([196.0, 162.0, 58.0], dtype=np.float32)
        img_f[(wall_mask > 0) & (top_mask == 0)] = np.array([128.0, 108.0, 74.0], dtype=np.float32)
        img_f[(stud_mask > 0) & (top_mask == 0)] = np.array([116.0, 96.0, 58.0], dtype=np.float32)
        cv2.circle(img_f, (212, 232), 7, (24.0, 24.0, 24.0), -1)

        rng = np.random.default_rng(123)
        scene_rgb = np.clip(np.rint(img_f + rng.normal(0.0, 2.0, size=img_f.shape)), 0, 255).astype(np.uint8)

        _, _, info = preprocess(scene_rgb, bg_model, cfg)
        self.assertTrue(info["valid"], msg=info["reason"])
        pred = info["full_mask"] > 0

        # Self-shaded wall & protruding studs must be preserved (>92% recall)
        shaded_block = ((wall_mask > 0) | (stud_mask > 0)) & (top_mask == 0)
        shaded_recall = float(np.count_nonzero(pred & shaded_block)) / float(np.count_nonzero(shaded_block))
        self.assertGreater(shaded_recall, 0.92, msg=f"Self-shaded block clipped: recall={shaded_recall:.3f}")

        # Ground shadows (outside 3px dilation margin) must be rejected
        safe_outside = cv2.dilate(gt_mask, np.ones((9, 9), dtype=np.uint8)) == 0
        ground_shadow_zone = safe_outside & ((cast_sh > 0) | (overhang_sh > 0))
        shadow_fp_rate = float(np.count_nonzero(pred & ground_shadow_zone)) / float(np.count_nonzero(ground_shadow_zone))
        self.assertLess(shadow_fp_rate, 0.03, msg=f"Ground shadow leaked: fp_rate={shadow_fp_rate:.3f}")

    def test_20_frame_lighting_acceptance_and_batch_background_set(self) -> None:
        # 20 frames across wide brightness, color-temperature, shadow, and noise variations
        gains = [
            (0.60 + 0.035 * i, 0.65 + 0.032 * i, 0.70 + 0.028 * i)
            for i in range(20)
        ]
        frames = []
        for i, gain in enumerate(gains):
            c_rgb, _ = _make_synthetic_duplo_rgb_scene(
                illum_gain=gain,
                cast_shadow=(i % 2 == 0),
                low_contrast_patch=True,
                noise_std=3.5,
                rng_seed=300 + i,
            )
            frames.append((c_rgb, f"light_{i:02d}"))

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rep = run_acceptance_test(
                frames=frames,
                bg_color=self.bg_model,
                output_dir=tmp_path / "acceptance",
                cfg=PreprocessConfig(),
                same_pose=True,
                max_area_variation_pct=5.0,
            )
            self.assertTrue(rep["passed"], msg=str(rep))
            self.assertEqual(rep["num_valid"], 20)
            self.assertLessEqual(rep["area_max_min_spread_pct"], 5.0)

            # Test batch processing with a background photo set folder + train/good & test/defect splits
            raw_root = tmp_path / "data/raw/mat"
            bg_dir = raw_root / "background"
            good_dir = raw_root / "train/good"
            defect_dir = raw_root / "test/defect"
            bg_dir.mkdir(parents=True)
            good_dir.mkdir(parents=True)
            defect_dir.mkdir(parents=True)

            for i in range(5):
                bg_img = _make_textured_background(noise_std=3.0, rng_seed=50 + i)
                cv2.imwrite(str(bg_dir / f"bg_{i:02d}.png"), cv2.cvtColor(bg_img, cv2.COLOR_RGB2BGR))

            for i in range(3):
                cv2.imwrite(
                    str(good_dir / f"good_{i:02d}.png"),
                    cv2.cvtColor(frames[i][0], cv2.COLOR_RGB2BGR),
                )
            cv2.imwrite(
                str(defect_dir / "def_00.png"),
                cv2.cvtColor(frames[4][0], cv2.COLOR_RGB2BGR),
            )

            cal_bg_path = tmp_path / "background_mat.png"
            calibrate_background_from_folder(bg_dir, output_path=cal_bg_path)
            self.assertTrue(cal_bg_path.is_file())

            out_root = tmp_path / "data/processed/mat"
            summary = process_dataset_batch(
                input_dir=raw_root,
                output_dir=out_root,
                view="mat",
            )
            self.assertTrue(summary["used_background_reference"])
            self.assertEqual(summary["valid_saved"], 4)
            self.assertTrue((out_root / "train/good/good_00.png").is_file())
            self.assertTrue((out_root / "test/defect/def_00.png").is_file())

            rots = get_rotations_rgb(frames[0][0][:256, :256], angles=(0, 90, 180, 270))
            self.assertEqual(len(rots), 4)

    def test_specular_glare_removal_preserves_defects_and_white_bricks(self) -> None:
        """Specular lamp reflections must be inpainted out while dark/colored defects and white bricks stay intact."""
        cfg = PreprocessConfig(REMOVE_GLARE=True)

        # 1. Red Duplo brick with a bright desaturated lamp glare hotspot AND a dark marker stroke right next to it
        color, gt_mask = _make_synthetic_duplo_rgb_scene(
            brick_rgb=(205, 38, 34),
            cast_shadow=False,
            low_contrast_patch=False,
            noise_std=2.0,
            rng_seed=777,
        )
        scene_glare = color.copy()
        # Paint a realistic specular lamp glare spot (bright desaturated core + soft halo) on the brick
        glare_layer = np.zeros_like(scene_glare, dtype=np.float32)
        cv2.ellipse(glare_layer, (305, 236), (11, 6), 15.0, 0.0, 360.0, (255.0, 255.0, 255.0), -1)
        glare_alpha = cv2.GaussianBlur(glare_layer[:, :, 0] / 255.0, (0, 0), sigmaX=2.2)[:, :, None]
        scene_glare = np.clip(
            np.rint((1.0 - glare_alpha) * scene_glare.astype(np.float32) + glare_alpha * 252.0),
            0,
            255,
        ).astype(np.uint8)
        # Paint a dark marker scratch right next to the glare spot
        cv2.line(scene_glare, (332, 226), (355, 248), (18, 18, 20), thickness=4)

        crop_no_glare_rm, mask_ref, _ = preprocess(scene_glare, self.bg_model, replace(cfg, REMOVE_GLARE=False))
        crop_cleaned, _, info_clean = preprocess(scene_glare, self.bg_model, cfg)

        self.assertTrue(info_clean["valid"])
        self.assertGreater(info_clean["glare_pixels_removed"], 20)

        # Bright white peak inside the brick mask must drop substantially back toward diffuse red plastic
        brick_sel = mask_ref > 0
        max_luma_before = float(np.max(np.mean(crop_no_glare_rm[brick_sel], axis=1)))
        max_luma_after = float(np.max(np.mean(crop_cleaned[brick_sel], axis=1)))
        self.assertGreater(max_luma_before, 240.0)
        self.assertLess(max_luma_after, 215.0)

        # Dark marker defect pixels must remain dark and untouched!
        dark_before = int(np.count_nonzero(np.mean(crop_no_glare_rm[brick_sel], axis=1) < 45.0))
        dark_after = int(np.count_nonzero(np.mean(crop_cleaned[brick_sel], axis=1) < 45.0))
        self.assertGreater(dark_before, 25)
        self.assertAlmostEqual(dark_after, dark_before, delta=3)

        # 2. Clean matte white Duplo brick without sharp glare spikes must NOT be darkened
        white_crop = np.full((256, 256, 3), 128, dtype=np.uint8)
        white_mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.rectangle(white_mask, (48, 72), (208, 184), 255, -1)
        white_crop[white_mask > 0] = np.array([236, 238, 240], dtype=np.uint8)
        white_out, white_glare = remove_specular_glare(white_crop, white_mask, cfg)
        self.assertEqual(int(np.count_nonzero(white_glare)), 0)
        np.testing.assert_array_equal(white_out, white_crop)


if __name__ == "__main__":
    unittest.main(verbosity=2)
