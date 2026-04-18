# Indian Road Perception

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-orange)
![YOLOv11](https://img.shields.io/badge/YOLOv11-8.4.38-green)
![ROS2](https://img.shields.io/badge/ROS2-Humble-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)
![GPU](https://img.shields.io/badge/GPU-RTX%203060-76b900)

**End-to-end real-time perception system for Indian unstructured driving scenarios**

*Object Detection + Semantic Segmentation + AEB trigger logic*

*M.Tech Automotive Electronics Dissertation @ BITS Pilani (WILP)*

</div>

---

## Project Overview

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

## System Architecture
Video Input (Camera / ROS2 Topic)
│
▼
┌─────────────────────┐     ┌──────────────────────┐
│  YOLOv11 Detection  │     │  YOLOv11 Segmentation │
│  (Bounding Boxes)   │     │  (Instance Masks)     │
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
---

## Demo Results

### Tested On
| Video | Location | Condition | AEB Alerts |
|-------|----------|-----------|------------|
| Chennai Traffic | Chennai, TN | Daytime, Urban | ✅ Active |
| Delhi Traffic | Delhi, NCR | Daytime, Highway | ✅ Active |
| Rainy Night | Indian City | Night + Rain | ⚠️ Reduced |

### Detection Performance (COCO Pretrained Baseline)
| Metric | Value |
|--------|-------|
| FPS on RTX 3060 | ~25-30 |
| Input Resolution | 640×640 |
| Model Size | 18.4 MB |
| GPU Memory | ~2GB |

> **Note:** Currently using COCO pretrained weights.
> Fine-tuning on IDD (India Driving Dataset) in progress
> for Indian-specific classes (auto-rickshaw, buffalo, cattle).

---

## 🚨 AEB Logic
Annotated Output + AEB Alert
---

## Demo Results

### Tested On
| Video | Location | Condition | AEB Alerts |
|-------|----------|-----------|------------|
| Chennai Traffic | Chennai, TN | Daytime, Urban | ✅ Active |
| Delhi Traffic | Delhi, NCR | Daytime, Highway | ✅ Active |
| Rainy Night | Indian City | Night + Rain | ⚠️ Reduced |

### Detection Performance (COCO Pretrained Baseline)
| Metric | Value |
|--------|-------|
| FPS on RTX 3060 | ~25-30 |
| Input Resolution | 640×640 |
| Model Size | 18.4 MB |
| GPU Memory | ~2GB |

> **Note:** Currently using COCO pretrained weights.
> Fine-tuning on IDD (India Driving Dataset) in progress
> for Indian-specific classes (auto-rickshaw, buffalo, cattle).

---

## AEB Logic
IF object_class IN [pedestrian, bicycle, motorcycle, cow, dog]
AND distance < 30m  → AEB WARNING (orange banner)
AND distance < 15m  → AEB CRITICAL (red banner)
Distance estimated using monocular pinhole camera model: distance = (real_object_height × focal_length_px) / bbox_height_px
---

## Project Structure
Indian-road-perception/
├── configs/
│   └── classes_india.yaml       # India-specific class taxonomy
├── src/
│   ├── detection/
│   │   └── yolo_inference.py    # YOLOv11 detection engine
│   ├── segmentation/
│   │   └── seg_inference.py     # YOLOv11 segmentation engine
│   ├── pipeline/
│   │   └── combined_pipeline.py # Combined AEB pipeline (MAIN DEMO)
│   └── utils/                   # Distance, TTC, visualisation
├── data/samples/                # Indian road test videos
├── results/videos/              # Annotated output videos
├── notebooks/                   # Training and evaluation
├── docs/                        # Architecture and results
├── requirements.txt
└── environment.yml
---

## Installation

### Prerequisites
- Ubuntu 22.04 LTS
- NVIDIA GPU (RTX 3060 recommended)
- CUDA 12.0+
- Anaconda / Miniconda
- ROS2 Humble (optional — for vehicle integration)

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
pip install -r requirements.txt
```

### Step 4 — Verify GPU
```bash
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
```

---

## 🚀 Quick Start

### Run Combined Pipeline (Recommended)
```bash
# On video file
python src/pipeline/combined_pipeline.py \
  --source data/samples/chennai_traffic.mp4 \
  --output results/videos/output.mp4

# On live webcam
python src/pipeline/combined_pipeline.py \
  --source 0 \
  --output results/videos/webcam_output.mp4
```

### Run Detection Only
```bash
python src/detection/yolo_inference.py \
  --source data/samples/chennai_traffic.mp4 \
  --output results/videos/detection.mp4
```

### Run Segmentation Only
```bash
python src/segmentation/seg_inference.py \
  --source data/samples/chennai_traffic.mp4 \
  --output results/videos/segmentation.mp4
```

---

## 🗺️ Roadmap

- [x] YOLOv11 object detection on Indian road videos
- [x] YOLOv11 instance segmentation
- [x] Combined detection + segmentation pipeline
- [x] Monocular distance estimation
- [x] AEB warning trigger logic
- [ ] Fine-tuning on IDD (India Driving Dataset)
- [ ] Auto-rickshaw and cattle class improvement
- [ ] ROS2 node integration
- [ ] Ouster LiDAR point cloud fusion
- [ ] ARS408 Radar integration
- [ ] Multi-modal sensor fusion (Camera + LiDAR + Radar)
- [ ] Autoware Universe integration
- [ ] Night/rain robustness improvement

---

## 📚 References & Credits

- Detection: [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics) (AGPL-3.0)
- Dataset: [IDD — India Driving Dataset](https://idd.insaan.iiit.ac.in), IIIT Hyderabad
- Segmentation: [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch) (MIT)
- AEB Standards: Euro NCAP AEB VRU Protocol 2023
- Vehicle dimensions: IS 11231, MoRTH Vehicle Classification
- Lane width: IRC 86 (Indian Roads Congress)

> **Licensing note:** This project uses Ultralytics YOLOv11 (AGPL-3.0).
> For commercial deployment, replace with a commercially licensed detection backbone.

---

## 👤 Author

**Ibrahim Khaleel Shaik**

- 💼 Mechanical Design Engineer — EPB Systems, Maruti Suzuki India Ltd.
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
