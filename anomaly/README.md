# Person C — Single-Camera Anomalib 2.7.0 & ZeroMQ VLA Service (`anomaly/`)

## Verified Commands (`anomalib==2.7.0`, `openvino==2026.4.1`)

```bash
# 1. Train single-camera 3-model ensemble on data/processed/mat/
python -m anomaly.train --model padim --version 1
python -m anomaly.train --model patchcore --version 1
python -m anomaly.train --model efficient_ad --version 1

# 2. Export to OpenVINO IR (pins static [1, 3, 256, 256] input shape for Intel NPU & GPU)
python -m anomaly.export --model padim --version 1 --precision fp16
python -m anomaly.export --model patchcore --version 1 --precision fp16
python -m anomaly.export --model efficient_ad --version 1 --precision fp16

# 3. Verify parallel PatchCore(GPU) + PaDiM(GPU) + EfficientAd(NPU) 2/3 voting ensemble & update configs/rules.yaml
python -m anomaly.verify --device AUTO_INTEL --update-rules

# 4. Test a single image from CLI
python -m anomaly.evaluate --image data/processed/mat/test/good/sample.png --verdict-only

# 5. Start ZeroMQ REP service (replies strictly "perfect" or "defective")
python -m anomaly.service
```

## Status & UNVERIFIED Items (Check on Intel Core Ultra GPU + NPU Laptop)
- **VERIFIED locally**: `anomalib==2.7.0` (`Padim`, `Patchcore`, `EfficientAd`), static `[1,3,256,256]` OpenVINO export, parallel C++ `ov.InferRequest.start_async` execution, `AUTO_INTEL` device routing (`EfficientAd -> NPU`, `PatchCore -> GPU`, `PaDiM -> GPU` with automatic fallback to `CPU`), threshold normalization + **2/3 majority voting**, specular glare removal (`remove_specular_glare`), and ZeroMQ plain-string reply (`"perfect"` | `"defective"`).
- **UNVERIFIED (hardware-dependent on Intel Core Ultra laptop)**:
  - `torch.xpu.is_available()` with `--xpu` (`XPUAccelerator` / `SingleXPUStrategy`).
  - Live hardware execution on Intel `NPU` (`EfficientAd`) and Intel Arc `GPU` (`PatchCore`, `PaDiM`) — local dev laptop is AMD Ryzen without Intel NPU/Arc GPU, so fallback to `CPU` was verified locally.
  - Live `pyrealsense2` camera streaming on Ubuntu 26.04.
