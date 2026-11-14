#!/usr/bin/env python3
"""Anomalib v2.7.0 single-camera training script (`data/processed/mat/`).

CPU-optimized for PatchCore without an NVIDIA GPU:
- Uses `resnet18` backbone by default on CPU (~5-6x faster than `wide_resnet50_2` and ~4x faster NPU/CPU inference).
- Caps redundant hue-augmented `train/good` frames via `--max-train-images` (keeps all original photos + diverse hue samples).
- Computes adaptive `coreset_sampling_ratio` from `--coreset-size` (default 1500 patches instead of 10% = 130,000+ patches),
  reducing `KCenterGreedy` coreset selection from ~18 hours down to ~2 minutes on CPU.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .config import load_model_config, normalize_model_type

SUPPORTED_MODELS = ("padim", "patchcore", "mh_patchcore", "found_ad", "efficient_ad")
CLI_MODEL_CHOICES = SUPPORTED_MODELS + ("efficientad", "efficientaid")
PATCHES_PER_256_IMG = 32 * 32  # layer2 stride-8 feature grid on 256x256 crops = 1024 patches/image


def _ensure_efficientad_ood_dir(
    ood_root: str | Path = "./anomaly/ov_cache/efficientad_ood_penalty",
    num_images: int = 96,
) -> Path:
    """Ensure an ImageFolder-compatible out-of-distribution penalty set exists for `EfficientAd`.

    If `./datasets/imagenette` is already present on disk, uses it directly; otherwise synthesizes
    a deterministic diverse OOD texture dataset (`ood/*.png`) so `EfficientAd` trains immediately
    without downloading a 1.55 GB archive.
    """
    imagenette = Path("./datasets/imagenette")
    if imagenette.is_dir() and any(imagenette.rglob("*.JPEG")):
        return imagenette

    root = Path(ood_root)
    cls_dir = root / "ood"
    existing = list(cls_dir.glob("*.png")) if cls_dir.is_dir() else []
    if len(existing) >= num_images:
        return root

    cls_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    sz = 320
    yy, xx = np.mgrid[0:sz, 0:sz].astype(np.float32)

    for idx in range(num_images):
        mode = idx % 4
        if mode == 0:
            # Multi-scale smoothed color noise
            coarse = rng.integers(0, 256, size=(20, 20, 3), dtype=np.uint8)
            img = cv2.resize(coarse, (sz, sz), interpolation=cv2.INTER_CUBIC)
        elif mode == 1:
            # High-frequency oriented sinusoidal gratings + random color Mix
            ang = float(rng.uniform(0.0, np.pi))
            freq = float(rng.uniform(0.05, 0.25))
            wave = 0.5 + 0.5 * np.sin((xx * np.cos(ang) + yy * np.sin(ang)) * freq)
            c1 = rng.uniform(0, 255, size=(1, 1, 3)).astype(np.float32)
            c2 = rng.uniform(0, 255, size=(1, 1, 3)).astype(np.float32)
            img = np.clip(wave[:, :, None] * c1 + (1.0 - wave[:, :, None]) * c2, 0, 255).astype(np.uint8)
        elif mode == 2:
            # Random geometric shapes & lines (simulates clutter / scratches / foreign objects)
            img = np.full((sz, sz, 3), int(rng.integers(20, 235)), dtype=np.uint8)
            for _ in range(14):
                col = tuple(int(c) for c in rng.integers(0, 256, size=3))
                p1 = (int(rng.integers(0, sz)), int(rng.integers(0, sz)))
                p2 = (int(rng.integers(0, sz)), int(rng.integers(0, sz)))
                cv2.line(img, p1, p2, col, thickness=int(rng.integers(2, 12)))
                cv2.circle(img, p1, int(rng.integers(6, 40)), col, -1)
        else:
            # Checkerboard + speckle noise
            tile = int(rng.integers(12, 48))
            chk = (((xx // tile).astype(int) + (yy // tile).astype(int)) % 2).astype(np.float32)
            c1 = rng.uniform(0, 255, size=(1, 1, 3)).astype(np.float32)
            c2 = rng.uniform(0, 255, size=(1, 1, 3)).astype(np.float32)
            noise = rng.normal(0.0, 18.0, size=(sz, sz, 3)).astype(np.float32)
            img = np.clip(chk[:, :, None] * c1 + (1.0 - chk[:, :, None]) * c2 + noise, 0, 255).astype(np.uint8)

        cv2.imwrite(str(cls_dir / f"ood_{idx:03d}.png"), img)
    return root


def create_model(
    model_type: str,
    image_sensitivity: float = 0.5,
    pixel_sensitivity: float = 0.5,
    backbone: str | None = None,
    coreset_sampling_ratio: float = 0.01,
    num_neighbors: int = 5,
    imagenet_dir: str | Path | None = None,
):
    from anomalib.models import EfficientAd, FoundAD, MHPatchcore, Padim, Patchcore
    from anomalib.post_processing import PostProcessor
    import torch

    model_type = normalize_model_type(model_type)
    post_processor = PostProcessor(
        image_sensitivity=image_sensitivity,
        pixel_sensitivity=pixel_sensitivity,
    )
    if model_type == "padim":
        return Padim(
            backbone=backbone or "resnet18",
            layers=("layer1", "layer2", "layer3"),
            post_processor=post_processor,
        )
    if model_type == "patchcore":
        chosen_backbone = backbone or ("wide_resnet50_2" if torch.cuda.is_available() else "resnet18")
        return Patchcore(
            backbone=chosen_backbone,
            layers=("layer2", "layer3"),
            coreset_sampling_ratio=coreset_sampling_ratio,
            num_neighbors=num_neighbors,
            post_processor=post_processor,
        )
    if model_type == "mh_patchcore":
        return MHPatchcore(
            backbone=backbone or "wide_resnet50_2.tv2_in1k",
            layers=("layer2", "layer3"),
            memory_bank_size=1000,
            post_processor=post_processor,
        )
    if model_type == "found_ad":
        # UNVERIFIED on Intel NPU; intended as iGPU challenger
        return FoundAD(post_processor=post_processor)
    if model_type == "efficient_ad":
        ood_path = Path(imagenet_dir) if imagenet_dir is not None else _ensure_efficientad_ood_dir()
        # `padding=True` uses symmetric Conv2d padding instead of asymmetric F.pad ops for clean Intel NPU compilation
        return EfficientAd(
            imagenet_dir=ood_path,
            teacher_out_channels=384,
            model_size="s",
            padding=True,
            post_processor=post_processor,
        )
    raise ValueError(f"Unsupported model '{model_type}'. Choose from {SUPPORTED_MODELS}")


def train_model(
    model_type: str | None = None,
    version: int | None = None,
    processed_dir: str | Path = "./data/processed/mat",
    results_root: str | Path = "./anomaly/results",
    config_path: str | Path = "./configs/defect_config.json",
    backbone: str | None = None,
    coreset_size: int | None = None,
    max_train_images: int | None = None,
    max_steps: int | None = None,
    download_imagenette: bool = False,
    synthetic_only: bool = False,
    use_masks: bool = False,
    use_xpu: bool = False,
    overwrite: bool = False,
) -> Path:
    from anomalib.engine import Engine
    import torch

    from .data import build_datamodule

    if not torch.cuda.is_available():
        torch.set_num_threads(max(4, os.cpu_count() or 8))

    mcfg = load_model_config(config_path)
    chosen_model = normalize_model_type(model_type or mcfg.model_type)
    ver = int(version if version is not None else mcfg.version)
    run_name = f"{chosen_model}_mat_v{ver}"

    run_dir = Path(results_root) / run_name
    if run_dir.exists() and any(run_dir.rglob("*.ckpt")) and not overwrite:
        raise FileExistsError(
            f"Checkpoint already exists in {run_dir}. Increment --version (never overwrite models!) "
            "or pass --overwrite explicitly."
        )

    has_fast_gpu = torch.cuda.is_available() or use_xpu
    if max_train_images is None:
        if chosen_model == "patchcore" and not has_fast_gpu:
            eff_max_train = 220
        elif chosen_model == "efficient_ad":
            eff_max_train = 300
        else:
            eff_max_train = None
    else:
        eff_max_train = max_train_images if max_train_images > 0 else None

    batch_size = 1 if chosen_model == "efficient_ad" else (32 if not has_fast_gpu else 16)
    datamodule = build_datamodule(
        processed_dir=processed_dir,
        batch_size=batch_size,
        synthetic_only=synthetic_only,
        use_masks=use_masks,
        max_train_images=eff_max_train,
    )

    n_train = len(datamodule.train_data) if hasattr(datamodule, "train_data") else 200
    target_coreset = coreset_size or (3000 if has_fast_gpu else 1500)
    total_patches = max(1, n_train * PATCHES_PER_256_IMG)
    coreset_ratio = float(min(0.10, max(0.0005, target_coreset / total_patches)))

    if chosen_model == "patchcore":
        eff_backbone = backbone or ("wide_resnet50_2" if torch.cuda.is_available() else "resnet18")
        print(
            f"[{run_name}] Fast PatchCore setup: backbone={eff_backbone}, train_images={n_train}, "
            f"total_patches~={total_patches}, target_coreset={int(total_patches * coreset_ratio)} "
            f"(ratio={coreset_ratio:.5f})"
        )

    imagenet_dir = "./datasets/imagenette" if download_imagenette else None
    model = create_model(
        model_type=chosen_model,
        image_sensitivity=mcfg.image_sensitivity,
        pixel_sensitivity=mcfg.pixel_sensitivity,
        backbone=backbone,
        coreset_sampling_ratio=coreset_ratio,
        imagenet_dir=imagenet_dir,
    )

    engine_kwargs: dict = {"default_root_dir": str(run_dir)}
    if chosen_model == "mh_patchcore":
        engine_kwargs["max_epochs"] = 3
        engine_kwargs["devices"] = 1
    elif chosen_model in ("padim", "patchcore"):
        engine_kwargs["max_epochs"] = 1
    elif chosen_model == "efficient_ad":
        eff_steps = max_steps if (max_steps is not None and max_steps > 0) else (3000 if has_fast_gpu else 1000)
        engine_kwargs["max_steps"] = eff_steps
        print(f"[{run_name}] EfficientAd NPU-ready setup (model_size='s', padding=True, max_steps={eff_steps})")

    if use_xpu:
        # UNVERIFIED on Windows without Arc XPU driver; verify torch.xpu.is_available() on Intel laptop
        from anomalib.engine import SingleXPUStrategy, XPUAccelerator

        engine_kwargs["accelerator"] = XPUAccelerator()
        engine_kwargs["strategy"] = SingleXPUStrategy()

    engine = Engine(**engine_kwargs)
    engine.fit(datamodule=datamodule, model=model)
    metrics = engine.test(datamodule=datamodule, model=model)
    print(f"[{run_name}] Validation/Test Metrics: {metrics}")
    return run_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train Anomalib v2.7.0 model on data/processed/mat.")
    parser.add_argument("--model", default=None, choices=CLI_MODEL_CHOICES)
    parser.add_argument("--version", type=int, default=None, help="Version number -> <model>_mat_v<N>")
    parser.add_argument("--backbone", default=None, help="Override CNN backbone (default: resnet18 on CPU, wide_resnet50_2 on CUDA)")
    parser.add_argument("--coreset-size", type=int, default=None, help="Target PatchCore memory bank patches (default: 1500 on CPU)")
    parser.add_argument("--max-train-images", type=int, default=None, help="Cap train/good images (default: 220 for CPU PatchCore, 300 for EfficientAd; 0 = all)")
    parser.add_argument("--max-steps", type=int, default=None, help="Training steps for EfficientAd (default: 1000 on CPU, 3000 on GPU/XPU)")
    parser.add_argument("--download-imagenette", action="store_true", help="Download full 1.55GB Imagenette dataset for EfficientAd instead of compact local OOD set")
    parser.add_argument("--processed-dir", default="./data/processed/mat")
    parser.add_argument("--results-root", default="./anomaly/results")
    parser.add_argument("--config", default="./configs/defect_config.json")
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--use-masks", action="store_true")
    parser.add_argument("--xpu", action="store_true", help="Train on Intel Arc XPU (UNVERIFIED locally).")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    train_model(
        model_type=args.model,
        version=args.version,
        processed_dir=args.processed_dir,
        results_root=args.results_root,
        config_path=args.config,
        backbone=args.backbone,
        coreset_size=args.coreset_size,
        max_train_images=args.max_train_images,
        max_steps=args.max_steps,
        download_imagenette=args.download_imagenette,
        synthetic_only=args.synthetic_only,
        use_masks=args.use_masks,
        use_xpu=args.xpu,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
