# `j26` — Junction 2026 Intel Physical AI (Person C: Perception & Anomalib)

Single-camera Duplo anomaly detection pipeline (`perception/`, `anomaly/`, `data/`, `configs/rules.yaml`) optimized for **Intel Core Ultra laptops (`GPU` + `NPU`)**.
ZeroMQ service (`tcp://127.0.0.1:5555`) removes shadows & specular glare, evaluates incoming camera frames using a **3-model parallel OpenVINO ensemble** (`PatchCore` on `GPU` + `PaDiM` on `GPU` + `EfficientAd` on `NPU`) with per-model threshold normalization and **2/3 majority voting**, and replies to the VLA with **only** `"perfect"` or `"defective"`.

```text
j26/
├── configs/
│   ├── rules.yaml               # Single binary threshold + per-model thresholds
│   └── defect_config.json       # RGB preprocessing & Anomalib model parameters
├── perception/                  # (C) RGB background subtraction, 256x256 crops, hue aug
├── anomaly/                     # (C) Anomalib 2.7.0 train, OpenVINO export, ZeroMQ service (:5555)
│   └── exports/                 # Versioned IRs: padim_mat_v3/, patchcore_mat_v4/, ...
├── data/
│   ├── calibration/             # Saved median background reference (background_mat.png)
│   ├── raw/                     # <-- DROP RAW PHOTOS FROM THE SINGLE CAMERA HERE
│   │   └── mat/
│   │       ├── background/      # <-- Empty-scene photos WITHOUT the Duplo block
│   │       ├── train/good/      # <-- Clean Duplo photos (printed numbers, curved, etc.)
│   │       ├── test/good/       # <-- Held-out clean Duplo photos
│   │       └── test/defect/     # <-- Defective Duplo photos (marker marks)
│   └── processed/               # <-- AUTO-GENERATED 256x256 CROPS FOR ANOMALIB
│       └── mat/{train/good, test/good, test/defect, ground_truth/defect}/
├── orchestrator/                # (D) state machine, safety, capture
├── policy/                      # (B) SmolVLA / ACT exports & scripted poses
├── bench/                       # (D) latency & power CSVs / plots
├── dashboard/                   # (D) live UI
└── docs/                        # (D) write-up, diagrams, datasets.md
```

---

## 1. End-to-End Workflow (Single Camera -> `"perfect"` | `"defective"`)

### Step 1: Put Raw Photos in `data/raw/mat/`
1. **Background photo set (no Duplo block):**
   Put 5–30 RGB photos of the empty mat/background into `data/raw/mat/background/`
   *(or capture directly: `python -m perception.capture_realsense background --count 15`)*.
2. **Duplo block photos:**
   - `data/raw/mat/train/good/` (clean bricks)
   - `data/raw/mat/test/good/` (held-out clean bricks)
   - `data/raw/mat/test/defect/` (bricks with marker marks)

### Step 2: YOLO Detection (`duplo-model/1`) + Background Subtraction, Glare Removal & Crop (`256x256`)
```powershell
# Set your Roboflow API key (sent strictly in 'Authorization: Bearer' header, never in query string)
$env:ROBOFLOW_API_KEY = "YOUR_ROBOFLOW_API_KEY"

# Test Roboflow Serverless 'duplo-model/1' YOLO detection on a single image
python -m perception.yolo_duplo --image data/raw/mat/test/defect/realsense_rgb_20261114_104222_559.png --save-vis anomaly/results/yolo_vis.png

# Segment strictly inside the YOLO bounding box, remove cast/bounce shadows + glare, and write 256x256 crops
python -m perception.preprocess

# Optional: generate hue-shifted copies in train/good to cover unseen brick colors
python -m perception.augment_hue
```

### Step 3: Train 3-Model Ensemble, Export OpenVINO IR, Calibrate Thresholds & Start Service
```powershell
# 1. Train Anomalib 2.7.0 3-model ensemble (Padim + PatchCore + EfficientAd)
python -m anomaly.train --model padim --version 1
python -m anomaly.train --model patchcore --version 1
python -m anomaly.train --model efficient_ad --version 1

# 2. Export to static [1, 3, 256, 256] OpenVINO IR -> anomaly/exports/<model>_mat_v<N>/weights/openvino/model.xml
python -m anomaly.export --model padim --version 1 --precision fp16
python -m anomaly.export --model patchcore --version 1 --precision fp16
python -m anomaly.export --model efficient_ad --version 1 --precision fp16

# 3. Calibrate per-model thresholds into configs/rules.yaml & verify 2/3 majority ensemble (EfficientAd->NPU, PatchCore/PaDiM->GPU)
python -m anomaly.verify --device AUTO_INTEL --update-rules

# 4. Inspect any single photo locally (prints JSON or --verdict-only 'perfect'/'defective' + optional heatmap PNG)
python -m anomaly.evaluate --image data/raw/mat/test/defect/realsense_rgb_20261114_104222_559.png --save-vis anomaly/results/eval_vis.png
python -m anomaly.evaluate --image data/raw/mat/test/defect/realsense_rgb_20261114_104222_559.png --verdict-only

# 5. Start ZeroMQ service on tcp://127.0.0.1:5555 (replies ONLY "perfect" or "defective")
python -m anomaly.service
```

---

## 2. ZeroMQ VLA Service Contract (`anomaly.service`)
- **Endpoint:** `tcp://127.0.0.1:5555` (`zmq.REP`)
- **Accepted Request Formats:**
  1. JSON string: `{"image": "<base64_png>"}` (optional `"crop": true` to force preprocessing)
  2. Plain base64-encoded PNG/JPEG string
  3. Raw binary PNG/JPEG bytes (`sock.send(png_bytes)`)
- **Reply Format (`sock.recv_string()`):**
  - Strictly `"perfect"` (`score < threshold` -> blue bin) or `"defective"` (`score >= threshold` or unreadable image -> pink bin).
