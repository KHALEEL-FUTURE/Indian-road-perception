# =============================================================================
# Indian Road Perception — YOLOv11 Object Detection
# =============================================================================
# Author      : Ibrahim Khaleel Shaik
# Affiliation : M.Tech Automotive Electronics, BITS Pilani (WILP)
#               Mechanical Design Engineer, Maruti Suzuki India Ltd.
# Purpose     : AEB-critical actor detection for Indian unstructured ODD
# Dataset     : Pretrained COCO — Fine-tuning on IDD in progress
# =============================================================================

import cv2
import time
import argparse
from pathlib import Path
from ultralytics import YOLO


class YOLODetector:
    """
    YOLOv11-based object detector configured for Indian road conditions.
    Detects AEB-critical actors: pedestrians, 2-wheelers, cattle, autos.
    """

    # COCO classes most relevant to Indian ODD
    INDIA_RELEVANT_CLASSES = {
        0:  'pedestrian',
        1:  'bicycle',
        2:  'motorcycle',
        5:  'bus',
        7:  'truck',
        3:  'car',
        15: 'cow',
        16: 'dog',
    }

    # Classes that trigger AEB warning
    AEB_CRITICAL = [0, 1, 2, 15, 16]

    def __init__(self, model_path='yolo11s.pt', conf=0.45, iou=0.50):
        """
        Args:
            model_path : YOLOv11 weights path
            conf       : Confidence threshold
            iou        : NMS IoU threshold
        """
        print(f"[INFO] Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.conf  = conf
        self.iou   = iou
        print("[INFO] Model loaded successfully")

    def detect(self, frame):
        """
        Run detection on a single frame.
        Returns list of detection dicts.
        """
        results    = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            verbose=False
        )[0]

        detections = []
        for box in results.boxes:
            cls_id     = int(box.cls[0])
            conf_score = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append({
                'class_id'    : cls_id,
                'class_name'  : self.model.names[cls_id],
                'confidence'  : conf_score,
                'bbox'        : (x1, y1, x2, y2),
                'aeb_critical': cls_id in self.AEB_CRITICAL
            })

        return detections

    def draw(self, frame, detections):
        """
        Draw bounding boxes and labels on frame.
        Red box = AEB critical | Green box = non-critical
        """
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label  = f"{det['class_name']} {det['confidence']:.2f}"
            color  = (0, 0, 255) if det['aeb_critical'] else (0, 255, 0)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                frame,
                (x1, y1 - th - 10),
                (x1 + tw, y1),
                color, -1
            )

            # Draw label text
            cv2.putText(
                frame, label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2
            )

        return frame


def run_on_video(video_path, output_path=None, show=True):
    """
    Run detection on video file.
    Args:
        video_path  : Input video path or 0 for webcam
        output_path : Save annotated video here
        show        : Display live window
    """
    detector = YOLODetector(model_path='yolo11s.pt')
    cap      = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    # Setup video writer
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_count = 0
    fps_list    = []

    print("[INFO] Detection running... Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        t_start    = time.time()
        detections = detector.detect(frame)
        frame      = detector.draw(frame, detections)
        fps_val    = 1.0 / (time.time() - t_start + 1e-6)
        fps_list.append(fps_val)
        frame_count += 1

        # AEB warning banner
        aeb_objects = [d for d in detections if d['aeb_critical']]
        if aeb_objects:
            cv2.rectangle(frame, (0, 0),
                         (frame.shape[1], 50), (0, 0, 180), -1)
            cv2.putText(
                frame,
                f"AEB WARNING: {aeb_objects[0]['class_name'].upper()} DETECTED",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 2
            )

        # Info overlay
        cv2.putText(frame, f"FPS: {fps_val:.1f}",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Objects: {len(detections)}",
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2)

        # Author watermark
        cv2.putText(
            frame,
            "Ibrahim Khaleel Shaik | BITS Pilani M.Tech | Maruti Suzuki",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (200, 200, 200), 1
        )

        if output_path:
            writer.write(frame)

        if show:
            cv2.imshow('Indian Road Perception - YOLOv11 Detection', frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

    # Cleanup
    cap.release()
    if output_path:
        writer.release()
    cv2.destroyAllWindows()

    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
    print(f"\n{'='*50}")
    print(f"Frames processed : {frame_count}")
    print(f"Average FPS      : {avg_fps:.1f}")
    print(f"Output saved     : {output_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='YOLOv11 Indian Road Object Detection'
    )
    parser.add_argument(
        '--source',
        type=str,
        default='data/samples/chennai_traffic.mp4',
        help='Video path or 0 for webcam'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/videos/detection_output.mp4',
        help='Output video save path'
    )
    args   = parser.parse_args()
    source = int(args.source) if args.source == '0' else args.source

    Path('results/videos').mkdir(parents=True, exist_ok=True)
    run_on_video(source, args.output)
