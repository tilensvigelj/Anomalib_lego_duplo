# Person C — Single-Camera Anomalib 2.7.0 & ZeroMQ VLA Service (`anomaly/`)

## Verified Commands (`anomalib==2.7.0`, `openvino==2026.4.1`)

```bash
# 1. Train single-camera models on data/processed/mat/
python -m anomaly.train --model padim --version 1
python -m anomaly.train --model patchcore --version 1

# 2. Export to OpenVINO IR (pins static [1, 3, 256, 256] input shape)
python -m anomaly.export --model padim --version 3 --precision fp16
python -m anomaly.export --model patchcore --version 4 --precision fp16

# 3. Verify parallel PatchCore + PaDiM ensemble & write single thresholds into configs/rules.yaml
python -m anomaly.verify --device CPU --update-rules

# 4. Test a single image from CLI
python -m anomaly.evaluate --image data/processed/mat/test/good/sample.png --verdict-only

# 5. Start ZeroMQ REP service (replies strictly "perfect" or "defective")
python -m anomaly.service
```

## Status & UNVERIFIED Items (Check on Ubuntu 26.04 Core Ultra X9 388H)
- **VERIFIED locally**: `anomalib==2.7.0` (`Padim`, `Patchcore`), static `[1,3,256,256]` OpenVINO export, parallel C++ `ov.InferRequest.start_async` execution of `PatchCore + PaDiM` + Lab color-deviation check, single-threshold calibration (`configs/rules.yaml`), and ZeroMQ plain-string reply (`"perfect"` | `"defective"`).
- **UNVERIFIED (hardware-dependent on Ubuntu 388H)**:
  - `torch.xpu.is_available()` with `--xpu` (`XPUAccelerator` / `SingleXPUStrategy`).
  - Native `NPU` compilation (`SingleModelDetector` automatically falls back `NPU -> HETERO:NPU,CPU -> GPU -> CPU`).
  - Live `pyrealsense2` camera streaming on Ubuntu 26.04.
