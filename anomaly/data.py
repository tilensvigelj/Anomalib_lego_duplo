"""Anomalib v2.7.0 Folder datamodule builder reading `data/processed/mat/`."""

from __future__ import annotations

from pathlib import Path

from anomalib.data import Folder
import numpy as np
import pandas as pd


def build_datamodule(
    processed_dir: str | Path = "./data/processed/mat",
    batch_size: int = 16,
    synthetic_only: bool = False,
    use_masks: bool = False,
    max_train_images: int | None = None,
    seed: int = 42,
) -> Folder:
    """Build an Anomalib v2.7.0 `Folder` datamodule on `data/processed/mat/`."""
    root_dir = Path(processed_dir).resolve()
    good_dir = root_dir / "train" / "good"
    if not good_dir.is_dir() or not any(
        p.suffix.lower() in {".png", ".jpg", ".jpeg"} for p in good_dir.iterdir() if p.is_file()
    ):
        raise FileNotFoundError(
            f"No preprocessed training images found in {good_dir}. "
            "Drop raw photos into data/raw/mat/train/good and run: "
            "python -m perception.preprocess"
        )

    test_defect_dir = root_dir / "test" / "defect"
    has_real_defects = test_defect_dir.is_dir() and any(
        p.suffix.lower() in {".png", ".jpg", ".jpeg"} for p in test_defect_dir.iterdir() if p.is_file()
    )

    kwargs: dict = {
        "name": "duplo_mat",
        "root": root_dir,
        "normal_dir": "train/good",
        "extensions": (".png", ".jpg", ".jpeg"),
        "train_batch_size": batch_size,
        "eval_batch_size": batch_size,
        "num_workers": 0,
        "seed": seed,
    }

    if synthetic_only or not has_real_defects:
        kwargs["test_split_mode"] = "synthetic"
        kwargs["val_split_mode"] = "same_as_test"
    else:
        kwargs["normal_test_dir"] = "test/good"
        kwargs["abnormal_dir"] = "test/defect"
        kwargs["test_split_mode"] = "from_dir"
        kwargs["val_split_mode"] = "from_test"
        kwargs["val_split_ratio"] = 0.5
        if use_masks and (root_dir / "ground_truth" / "defect").is_dir():
            kwargs["mask_dir"] = "ground_truth/defect"

    datamodule = Folder(**kwargs)
    datamodule.setup()

    if max_train_images and max_train_images > 0 and hasattr(datamodule, "train_data"):
        samples = datamodule.train_data.samples
        if len(samples) > max_train_images:
            is_hue = samples["image_path"].astype(str).str.contains("_hue", regex=False)
            orig_df = samples[~is_hue]
            hue_df = samples[is_hue]
            if len(orig_df) >= max_train_images:
                idx = np.linspace(0, len(orig_df) - 1, max_train_images, dtype=int)
                selected = orig_df.iloc[idx]
            else:
                rem = max_train_images - len(orig_df)
                hue_idx = np.unique(np.linspace(0, len(hue_df) - 1, rem, dtype=int))
                selected = pd.concat([orig_df, hue_df.iloc[hue_idx]], ignore_index=True)
            datamodule.train_data.samples = selected.reset_index(drop=True)

    return datamodule
