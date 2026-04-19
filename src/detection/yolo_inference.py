# =============================================================================
# Indian Road Perception - YOLOv11 Object Detection
# =============================================================================
# Author      : Ibrahim Khaleel Shaik
# Affiliation : M.Tech Automotive Electronics, BITS Pilani (WILP)
#               Mechanical Design Engineer, Maruti Suzuki India Ltd.
# Camera      : Qubo Dashcam Pro 4K | Sony IMX415 | F1.8 | 140deg FOV
# =============================================================================

import cv2
import time
import argparse
from pathlib import Path
from ultralytics import YOLO


class YOLODetector:
    AEB_CRITICAL = [0, 1, 2, 15, 16]

    def __init__(self, model_path='yolo11m.pt', conf=0.45, iou=0.50):
        print(f"[INFO] Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        print("[INFO] Model loaded successfully")

    def detect(self, frame):
        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=640,
            verbose=False
        )[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf_score = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                'class_id': cls_id,
                'class_name': self.model.names[cls_id],
                'confidence': conf_score,
                'bbox': (x1, y1, x2, y2),
                'aeb_critical': cls_id in self.AEB_CRITICAL
            })
        return detections

    def draw(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class_name']} {det['confidence']:.2f}"
            color = (0, 0, 255) if det['aeb_critical'] else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return frame


def run_on_video(video_path, output_path=None, show=True):
    detector = YOLODetector(model_path='yolo11m.pt')
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
    print("[INFO] Detection running... Press Q to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t_start = time.time()
        detections = detector.detect(frame)
        frame = detector.draw(frame, detections)
        fps_val = 1.0 / (time.time() - t_start + 1e-6)
        fps_list.append(fps_val)
        frame_count += 1
        aeb_objects = [d for d in detections if d['aeb_critical']]
        if aeb_objects:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 180), -1)
            cv2.putText(frame,
                        f"AEB WARNING: {aeb_objects[0]['class_name'].upper()}",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps_val:.1f}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Objects: {len(detections)}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame,
                    "Ibrahim Khaleel Shaik | BITS Pilani | Maruti Suzuki",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        if output_path:
            writer.write(frame)
        if show:
            cv2.imshow('Indian Road Perception - Detection', frame)
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
    parser.add_argument('--output', default='results/videos/detection_output.mp4')
    args = parser.parse_args()
    source = int(args.source) if args.source == '0' else args.source
    Path('results/videos').mkdir(parents=True, exist_ok=True)
    run_on_video(source, args.output)
