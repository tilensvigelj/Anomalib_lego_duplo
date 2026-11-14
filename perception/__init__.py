"""Person C perception package: RGB background subtraction, canonical 256x256 crops, hue augmentation, and Lab color-deviation checks."""

from typing import Any

_PREPROCESS_EXPORTS = {
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
}

_YOLO_EXPORTS = {
    "YoloDetection",
    "detect_best_duplo_box",
    "draw_yolo_detections",
    "get_roboflow_api_key",
    "get_roboflow_client",
    "infer_duplo_image",
    "parse_yolo_predictions",
}


def __getattr__(name: str) -> Any:
    if name in _PREPROCESS_EXPORTS:
        from . import duplo_rgbd_preprocess

        return getattr(duplo_rgbd_preprocess, name)
    if name in _YOLO_EXPORTS:
        from . import yolo_duplo

        return getattr(yolo_duplo, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "PreprocessConfig",
    "YoloDetection",
    "calibrate_background_from_folder",
    "compute_background_model",
    "create_debug_overlay",
    "detect_best_duplo_box",
    "discover_rgb_images",
    "draw_yolo_detections",
    "get_roboflow_api_key",
    "get_roboflow_client",
    "get_rotations_rgb",
    "infer_duplo_image",
    "load_background_model",
    "parse_yolo_predictions",
    "preprocess",
    "process_dataset_batch",
    "remove_specular_glare",
    "run_acceptance_test",
]
