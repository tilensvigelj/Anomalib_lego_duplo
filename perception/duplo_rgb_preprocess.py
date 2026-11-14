"""RGB-only Duplo background-subtraction preprocessing module (alias for duplo_rgbd_preprocess)."""

from .duplo_rgbd_preprocess import (
    PreprocessConfig,
    calibrate_background_from_folder,
    compute_background_model,
    create_debug_overlay,
    discover_rgb_images,
    get_rotations_rgb,
    load_background_model,
    preprocess,
    process_dataset_batch,
    remove_specular_glare,
    run_acceptance_test,
)

__all__ = [
    "PreprocessConfig",
    "calibrate_background_from_folder",
    "compute_background_model",
    "create_debug_overlay",
    "discover_rgb_images",
    "get_rotations_rgb",
    "load_background_model",
    "preprocess",
    "process_dataset_batch",
    "remove_specular_glare",
    "run_acceptance_test",
]
