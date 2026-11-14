#!/usr/bin/env python3
"""Export versioned single-camera Anomalib models to OpenVINO IR (`anomaly/exports/<model>_mat_v<N>`).

Reshapes dynamic batch dimensions to static `[1, 3, 256, 256]` for Intel NPU compilation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from anomalib.deploy import CompressionType, ExportType
from anomalib.engine import Engine
from anomalib.models import EfficientAd, FoundAD, MHPatchcore, Padim, Patchcore
import openvino as ov

from .config import load_model_config
from .data import build_datamodule

MODEL_CLASSES = {
    "padim": Padim,
    "patchcore": Patchcore,
    "mh_patchcore": MHPatchcore,
    "found_ad": FoundAD,
    "efficient_ad": EfficientAd,
}

COMPRESSION_MAP = {
    "fp16": CompressionType.FP16,
    "int8": CompressionType.INT8,
    "int8_ptq": CompressionType.INT8_PTQ,
    "int8_acq": CompressionType.INT8_ACQ,
}


def find_latest_ckpt(results_root: Path, run_name: str) -> Path:
    ckpts = sorted((results_root / run_name).rglob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt found under {results_root / run_name}")
    return ckpts[-1]


def enforce_static_batch1_shape(xml_path: Path, hw: int = 256) -> None:
    """Pin input shape to static `[1, 3, hw, hw]` as required by Intel NPU compiler."""
    core = ov.Core()
    model = core.read_model(str(xml_path))
    pshape = model.input(0).partial_shape
    if pshape.is_dynamic:
        target = [1, 3, hw, hw]
        if len(pshape) == 4 and pshape[-1].is_static and pshape[-1].get_length() == 3:
            target = [1, hw, hw, 3]
        model.reshape({model.input(0): target})
        tmp_xml = xml_path.with_name("model_static.xml")
        ov.save_model(model, str(tmp_xml))
        tmp_bin = tmp_xml.with_suffix(".bin")
        xml_path.unlink()
        bin_path = xml_path.with_suffix(".bin")
        if bin_path.exists():
            bin_path.unlink()
        tmp_xml.rename(xml_path)
        if tmp_bin.exists():
            tmp_bin.rename(bin_path)


def export_model(
    model_type: str | None = None,
    version: int | None = None,
    precision: str = "fp16",
    ckpt_path: str | Path | None = None,
    processed_dir: str | Path = "./data/processed/mat",
    results_root: str | Path = "./anomaly/results",
    exports_root: str | Path = "./anomaly/exports",
    config_path: str | Path = "./configs/defect_config.json",
    static_npu_shape: bool = True,
    overwrite: bool = False,
) -> Path:
    mcfg = load_model_config(config_path)
    chosen_model = model_type or mcfg.model_type
    ver = int(version if version is not None else mcfg.version)
    run_name = f"{chosen_model}_mat_v{ver}"
    suffix = "" if precision == "fp16" else f"_{precision}"
    export_name = f"{run_name}{suffix}"

    export_dir = Path(exports_root) / export_name
    xml_out = export_dir / "weights" / "openvino" / "model.xml"
    if xml_out.exists() and not overwrite:
        raise FileExistsError(
            f"Export already exists at {xml_out}. Increment --version (never overwrite exports!) "
            "or pass --overwrite."
        )

    resolved_ckpt = Path(ckpt_path) if ckpt_path else find_latest_ckpt(Path(results_root), run_name)
    model_cls = MODEL_CLASSES[chosen_model]
    model = model_cls.load_from_checkpoint(resolved_ckpt)
    engine = Engine()

    export_kwargs: dict = {
        "model": model,
        "export_type": ExportType.OPENVINO,
        "export_root": str(export_dir),
        "input_size": (256, 256),
        "compression_type": COMPRESSION_MAP[precision],
        "onnx_kwargs": (
            {"dynamic_shapes": None, "opset_version": 18}
            if static_npu_shape
            else {"opset_version": 18}
        ),
    }
    if precision in ("int8_ptq", "int8_acq"):
        export_kwargs["datamodule"] = build_datamodule(
            processed_dir=processed_dir,
            batch_size=1 if chosen_model == "efficient_ad" else 16,
        )

    engine.export(**export_kwargs)
    if static_npu_shape and xml_out.is_file():
        enforce_static_batch1_shape(xml_out, hw=256)
    print(f"Exported {export_name} -> {xml_out}")
    return xml_out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export trained Anomalib model to OpenVINO IR.")
    parser.add_argument("--model", default=None, choices=tuple(MODEL_CLASSES))
    parser.add_argument("--version", type=int, default=None)
    parser.add_argument("--precision", default="fp16", choices=tuple(COMPRESSION_MAP))
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--processed-dir", default="./data/processed/mat")
    parser.add_argument("--results-root", default="./anomaly/results")
    parser.add_argument("--exports-root", default="./anomaly/exports")
    parser.add_argument("--config", default="./configs/defect_config.json")
    parser.add_argument("--no-static-shape", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    export_model(
        model_type=args.model,
        version=args.version,
        precision=args.precision,
        ckpt_path=args.ckpt,
        processed_dir=args.processed_dir,
        results_root=args.results_root,
        exports_root=args.exports_root,
        config_path=args.config,
        static_npu_shape=not args.no_static_shape,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
