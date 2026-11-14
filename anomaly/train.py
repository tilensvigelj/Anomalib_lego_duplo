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

from anomalib.engine import Engine
from anomalib.models import EfficientAd, FoundAD, MHPatchcore, Padim, Patchcore
from anomalib.post_processing import PostProcessor
import torch

from .config import load_model_config
from .data import build_datamodule

SUPPORTED_MODELS = ("padim", "patchcore", "mh_patchcore", "found_ad", "efficient_ad")
PATCHES_PER_256_IMG = 32 * 32  # layer2 stride-8 feature grid on 256x256 crops = 1024 patches/image


def create_model(
    model_type: str,
    image_sensitivity: float = 0.5,
    pixel_sensitivity: float = 0.5,
    backbone: str | None = None,
    coreset_sampling_ratio: float = 0.01,
    num_neighbors: int = 5,
):
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
        return EfficientAd(teacher_out_channels=384, post_processor=post_processor)
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
    synthetic_only: bool = False,
    use_masks: bool = False,
    use_xpu: bool = False,
    overwrite: bool = False,
) -> Path:
    if not torch.cuda.is_available():
        torch.set_num_threads(max(4, os.cpu_count() or 8))

    mcfg = load_model_config(config_path)
    chosen_model = model_type or mcfg.model_type
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
        eff_max_train = None if (has_fast_gpu or chosen_model != "patchcore") else 220
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

    model = create_model(
        model_type=chosen_model,
        image_sensitivity=mcfg.image_sensitivity,
        pixel_sensitivity=mcfg.pixel_sensitivity,
        backbone=backbone,
        coreset_sampling_ratio=coreset_ratio,
    )

    engine_kwargs: dict = {"default_root_dir": str(run_dir)}
    if chosen_model == "mh_patchcore":
        engine_kwargs["max_epochs"] = 3
        engine_kwargs["devices"] = 1
    elif chosen_model in ("padim", "patchcore"):
        engine_kwargs["max_epochs"] = 1
    elif chosen_model == "efficient_ad":
        engine_kwargs["max_epochs"] = 10

    if use_xpu:
        # UNVERIFIED on Windows without Arc XPU driver; verify torch.xpu.is_available() on Ubuntu 388H
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
    parser.add_argument("--model", default=None, choices=SUPPORTED_MODELS)
    parser.add_argument("--version", type=int, default=None, help="Version number -> <model>_mat_v<N>")
    parser.add_argument("--backbone", default=None, help="Override CNN backbone (default: resnet18 on CPU, wide_resnet50_2 on CUDA)")
    parser.add_argument("--coreset-size", type=int, default=None, help="Target PatchCore memory bank patches (default: 1500 on CPU)")
    parser.add_argument("--max-train-images", type=int, default=None, help="Cap train/good images (default: 220 for CPU PatchCore; 0 = all)")
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
        synthetic_only=args.synthetic_only,
        use_masks=args.use_masks,
        use_xpu=args.xpu,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
