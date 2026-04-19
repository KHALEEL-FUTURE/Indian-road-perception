# =============================================================================
# Indian Road Perception - YOLOv11 Segmentation
# =============================================================================
# Author      : Ibrahim Khaleel Shaik
# Affiliation : M.Tech Automotive Electronics, BITS Pilani (WILP)
#               Mechanical Design Engineer, Maruti Suzuki India Ltd.
# Camera      : Qubo Dashcam Pro 4K | Sony IMX415 | F1.8 | 140deg FOV
# =============================================================================

import cv2
import time
import argparse
import numpy as np
from pathlib import Path
from ultralytics import YOLO


class YOLOSegmentor:
    COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (0, 255, 255), (255, 0, 255),
        (128, 255, 0), (0, 128, 255), (255, 0, 128),
        (128, 0, 255), (0, 255, 128), (255, 128, 0),
        (0, 128, 128), (128, 128, 0), (128, 0, 0),
    ]
    AEB_CRITICAL = [0, 1, 2, 15, 16]

    def __init__(self, model_path='yolo11m-seg.pt', conf=0.45):
        print(f"[INFO] Loading segmentation model: {model_path}")
        self.model = YOLO(model_path)
        self.conf = conf
        print("[INFO] Segmentation model loaded successfully")

    def segment(self, frame):
        results = self.model(
            frame,
            conf=self.conf,
            imgsz=1280,
            verbose=False
        )[0]
        overlay = frame.copy()
        if results.masks is not None:
            masks = results.masks.data.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()
            for i, (mask, cls_id, conf) in enumerate(zip(masks, classes, confs)):
                color = self.COLORS[cls_id % len(self.COLORS)]
                mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                bool_mask = mask_resized > 0.5
                overlay[bool_mask] = (
                    overlay[bool_mask] * 0.45 +
                    np.array(color) * 0.55
                ).astype(np.uint8)
                mask_uint8 = (mask_resized * 255).astype(np.uint8)
                contours, _ = cv2.findContours(
                    mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, color, 2)
                ys, xs = np.where(bool_mask)
                if len(xs) > 0:
                    cx = int(xs.mean())
                    cy = int(ys.mean())
                    label = f"{self.model.names[cls_id]} {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    cv2.rectangle(overlay,
                                  (cx - 2, cy - th - 8),
                                  (cx + tw + 2, cy), color, -1)
                    cv2.putText(overlay, label, (cx, cy - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        return overlay

    def get_detections(self, frame):
        results = self.model(
            frame, conf=self.conf, imgsz=1280, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            detections.append({
                'class_id': cls_id,
                'class_name': self.model.names[cls_id],
                'confidence': float(box.conf[0]),
                'aeb_critical': cls_id in self.AEB_CRITICAL
            })
        return detections


def run_segmentation(video_path, output_path=None, show=True):
    segmentor = YOLOSegmentor(model_path='yolo11m-seg.pt')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_path}")
        return
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    frame_count = 0
    fps_list = []
    print("[INFO] Segmentation running... Press Q to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t_start = time.time()
        annotated = segmentor.segment(frame)
        fps_val = 1.0 / (time.time() - t_start + 1e-6)
        fps_list.append(fps_val)
        frame_count += 1
        detections = segmentor.get_detections(frame)
        aeb_objects = [d for d in detections if d['aeb_critical']]
        if aeb_objects:
            cv2.rectangle(annotated, (0, 0),
                          (annotated.shape[1], 50), (0, 0, 180), -1)
            cv2.putText(annotated,
                        f"AEB WARNING: {aeb_objects[0]['class_name'].upper()}",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(annotated, f"FPS: {fps_val:.1f}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(annotated, f"Segments: {len(detections)}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(annotated,
                    "Ibrahim Khaleel Shaik | BITS Pilani | Maruti Suzuki",
                    (10, annotated.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        if output_path:
            writer.write(annotated)
        if show:
            cv2.imshow('Indian Road Perception - Segmentation', annotated)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
    cap.release()
    if output_path:
        writer.release()
    cv2.destroyAllWindows()
    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
    print(f"\nFrames: {frame_count} | Avg FPS: {avg_fps:.1f} | Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='data/samples/chennai_traffic.mp4')
    parser.add_argument('--output', default='results/videos/segmentation_output.mp4')
    args = parser.parse_args()
    source = int(args.source) if args.source == '0' else args.source
    Path('results/videos').mkdir(parents=True, exist_ok=True)
    run_segmentation(source, args.output)
