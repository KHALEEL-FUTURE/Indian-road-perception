# Indian Road Perception

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-orange)
![YOLOv11](https://img.shields.io/badge/YOLOv11m-8.4.38-green)
![ROS2](https://img.shields.io/badge/ROS2-Humble-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)
![GPU](https://img.shields.io/badge/GPU-RTX%203060-76b900)
![mAP50](https://img.shields.io/badge/mAP50-50.1%25-brightgreen)
![Dataset](https://img.shields.io/badge/Dataset-IDD%2BIDD--AW-orange)

**End-to-end real-time perception system for Indian unstructured driving scenarios**

*Object Detection + Semantic Segmentation + AEB trigger logic*

*Fine-tuned on IDD + IDD-AW for India-specific VRU detection*

*M.Tech Automotive Electronics Dissertation @ BITS Pilani (WILP)*

</div>

---

## 🎯 Project Overview

Standard COCO-trained perception models fail on Indian roads because they lack
critical classes like **auto-rickshaws, cattle, and combined motorcycle-rider units**.
This project builds a production-grade perception stack specifically designed for
**Indian unstructured ODD (Operational Design Domain)**, targeting L2+/L3 ADAS
and AEB (Autonomous Emergency Braking) applications.

### Why Indian Roads Are Different
- 🛺 Auto-rickshaws — unpredictable lateral cuts, not in COCO
- 🐄 Free-roaming cattle — low radar cross-section, sudden trajectory changes
- 🏍️ 2-wheelers with riders — highest fatality rate (MoRTH 2023)
- 🚶 Dense pedestrian-vehicle mixing — no lane discipline
- 🌧️ Extreme weather — monsoon, dust, fog degrading sensor performance

---

## Key Results

### Model Performance Comparison

| Model | Dataset | mAP50 | mAP50-95 | autorickshaw |
|-------|---------|-------|----------|--------------|
| YOLOv11m COCO (baseline) | COCO | 47.0% | 32.0% | ~15% |
| YOLOv11m Phase 1 | IDD Detection | 48.6% | 32.8% | 70.2% |
| **YOLOv11m Phase 2** | **IDD + IDD-AW** | **50.1%** | **34.5%** | **72.0%** |

### Per-Class mAP50 (Final Model)

| Class | mAP50 |
|-------|-------|
| bus | 74.9% |
| autorickshaw | 72.0% |
| truck | 70.5% |
| motorcycle | 69.8% |
| car | 69.7% |
| rider | 61.7% |
| person | 57.0% |
| bicycle | 51.8% |

### Weather Condition Performance

| Condition | Detection Accuracy |
|-----------|-------------------|
| Daylight Clear | 95% |
| Daylight Traffic | 90% |
| Twilight | 80% |
| Night Low Light | 62% |

---

## 🗃️ Training Datasets

| Dataset | Images | Source | Purpose |
|---------|--------|--------|---------|
| IDD Detection | 41,794 | IIIT Hyderabad | Primary VRU detection |
| IDD-AW Adverse Weather | 3,853 | IIIT Hyderabad | Weather robustness |
| **Combined Total** | **45,647** | **IDD + IDD-AW** | **Final model** |

### IDD-AW Weather Breakdown
| Condition | Train | Val |
|-----------|-------|-----|
| FOG | 2,066 | 308 |
| RAIN | 2,124 | 240 |
| LOWLIGHT | 1,369 | 200 |
| SNOW | 1,302 | 202 |

---

## System Architecture
Video Input (Camera / ROS2 Topic)
│
▼
┌─────────────────────┐     ┌──────────────────────┐
│  YOLOv11m Detection │     │  YOLOv11m Segmentation│
│  India VRU Model    │     │  Instance Masks       │
│  mAP50 = 50.1%      │     │                      │
└─────────┬───────────┘     └──────────┬───────────┘
│                            │
└──────────┬─────────────────┘
▼
┌──────────────────────┐
│   Combined Pipeline  │
│  Distance Estimation │
│  TTC Computation     │
│  AEB Trigger Logic   │
└──────────┬───────────┘
▼
Annotated Output + AEB Alert
---

## AEB Logic
IF object_class IN [person, rider, motorcycle, bicycle,
autorickshaw, animal]
AND distance < 30m  → AEB WARNING (orange banner)
AND distance < 15m  → AEB CRITICAL (red banner)
Distance = (real_height × focal_length) / bbox_height_px
Focal length = 233px (Qubo 4K dashcam, 140° FOV)
---

## Project Structure
Indian-road-perception/
├── src/
│   ├── detection/
│   │   └── yolo_inference.py       # YOLOv11m detection engine
│   ├── segmentation/
│   │   └── seg_inference.py        # YOLOv11m segmentation engine
│   └── pipeline/
│       └── combined_pipeline.py    # Combined AEB pipeline
├── scripts/
│   ├── run_detection_vru.py        # Detection runner
│   ├── run_segmentation_vru.py     # Segmentation runner
│   ├── run_combined_vru.py         # Combined runner
│   ├── convert_idd_detection_to_yolo.py
│   └── convert_iddaw_to_yolo.py
├── configs/
│   ├── classes_india.yaml          # India VRU class taxonomy
│   ├── india_vru.yaml              # IDD dataset config
│   └── india_vru_combined.yaml     # IDD+IDD-AW dataset config
├── results/videos/                 # Demo output videos
├── requirements.txt
└── environment.yml
---

## Installation

### Prerequisites
- Ubuntu 22.04 LTS
- NVIDIA GPU (RTX 3060 recommended)
- CUDA 12.0+
- Anaconda / Miniconda
- ROS2 Humble (optional)

### Step 1 — Clone Repository
```bash
git clone https://github.com/KHALEEL-FUTURE/Indian-road-perception.git
cd Indian-road-perception
```

### Step 2 — Create Environment
```bash
conda create -n perception_india python=3.10 -y
conda activate perception_india
```

### Step 3 — Install Dependencies
```bash
pip install ultralytics opencv-python-headless numpy
```

### Step 4 — Verify GPU
```bash
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
```

---

## Quick Start

### Detection Only
```bash
python scripts/run_detection_vru.py \
  --source your_video.mp4 \
  --output output_detection.mp4 \
  --model india_vru_best.pt
```

### Segmentation Only
```bash
python scripts/run_segmentation_vru.py \
  --source your_video.mp4 \
  --output output_segmentation.mp4
```

### Combined Detection + Segmentation + AEB
```bash
python scripts/run_combined_vru.py \
  --source your_video.mp4 \
  --output output_combined.mp4 \
  --det-model india_vru_best.pt
```

---

## Demo Videos Tested On

| Video | Location | Condition | Model |
|-------|----------|-----------|-------|
| Chennai Traffic | Chennai, TN | Daytime Urban | IDD fine-tuned |
| Delhi Traffic | Delhi, NCR | Daytime Highway | IDD fine-tuned |
| Rainy Night | Indian City | Night + Rain | IDD+AW fine-tuned |
| Dashcam 1 | Gurugram | Daytime Real | IDD fine-tuned |
| Dashcam 2 | Gurugram | Mixed | IDD fine-tuned |
| Dashcam 3 | Gurugram | Twilight | IDD+AW fine-tuned |

---

## Roadmap

- [x] YOLOv11m object detection on Indian road videos
- [x] YOLOv11m instance segmentation pipeline
- [x] Combined detection + segmentation + AEB pipeline
- [x] Monocular distance estimation (Qubo 4K calibrated)
- [x] AEB warning trigger logic
- [x] **Fine-tuning on IDD Detection (41,794 images)**
- [x] **Fine-tuning on IDD-AW Adverse Weather (3,853 images)**
- [x] **autorickshaw mAP50 improved from ~15% → 72%**
- [x] Testing on 6 Indian road videos
- [ ] ROS2 node integration
- [ ] Ouster OS1 LiDAR point cloud fusion
- [ ] Continental ARS408 Radar integration
- [ ] Multi-modal sensor fusion (Camera + LiDAR + Radar)
- [ ] TIHAN testbed validation (IIT Hyderabad)
- [ ] Autoware Universe integration

---

## References & Credits

- Detection: [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics) (AGPL-3.0)
- Dataset: [IDD Detection](https://idd.insaan.iiit.ac.in), IIIT Hyderabad
- Dataset: [IDD-AW Adverse Weather](https://idd.insaan.iiit.ac.in), IIIT Hyderabad
- AEB Standards: Euro NCAP AEB VRU Protocol 2023
- MoRTH Road Accident Statistics 2023
- Vehicle dimensions: IS 11231, MoRTH Vehicle Classification
- Lane width: IRC 86 (Indian Roads Congress)

> **Licensing note:** This project uses Ultralytics YOLOv11 (AGPL-3.0).
> For commercial deployment, replace with a commercially licensed detection backbone.

---

## 👤 Author

**Ibrahim Khaleel Shaik**

- 🎓 M.Tech Automotive Electronics, BITS Pilani (WILP)
- 🏅 ISO 26262 Functional Safety Engineer — TÜV SÜD Certified (Level 1)
- 🔬 Research: Multi-modal perception for Indian L3+ ADAS
- 🌐 GitHub: [KHALEEL-FUTURE](https://github.com/KHALEEL-FUTURE)

---

## 📄 License

This project is licensed under the MIT License.
See [LICENSE](LICENSE) for details.

---

<div align="center">
<i>Built for Indian roads. Designed for safety. Driven by engineering.</i>
</div>
