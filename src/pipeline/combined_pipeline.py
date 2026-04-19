# =============================================================================
# Indian Road Perception - Combined Detection + Segmentation Pipeline
# =============================================================================
# Author      : Ibrahim Khaleel Shaik
# Affiliation : M.Tech Automotive Electronics, BITS Pilani (WILP)
#               Mechanical Design Engineer, Maruti Suzuki India Ltd.
# Camera      : Qubo Dashcam Pro 4K | Sony IMX415 | F1.8 | 140deg FOV
# THIS IS THE MAIN DEMO FILE - Run this for professor demonstration
# =============================================================================

import cv2
import time
import argparse
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from detection.yolo_inference import YOLODetector
from segmentation.seg_inference import YOLOSegmentor

# =============================================================================
# CAMERA PARAMETERS — Qubo Dashcam Pro 4K
# Sensor  : Sony IMX415
# FOV     : 140 degrees (horizontal)
# Res     : Downscaled to 1280x720 for processing
# Formula : fx = (width/2) / tan(FOV_deg/2 * pi/180)
#         : fx = 640 / tan(70 * pi/180)
#         : fx = 640 / 2.747 = 232.97 px
# Note    : Approximate — update with OpenCV checkerboard calibration
#           for production accuracy
# =============================================================================
FOCAL_LENGTH_PX = 233.0      # Qubo 4K dashcam at 720p — 140deg FOV
IMAGE_WIDTH_PX  = 1280       # 720p width
IMAGE_HEIGHT_PX = 720        # 720p height
FOV_DEGREES     = 140        # Qubo spec

# Real world object heights (metres)
# Source: IS 11231, MoRTH, field measurements
OBJECT_HEIGHTS_M = {
    'person'    : 1.70,
    'bicycle'   : 1.10,
    'motorcycle': 1.50,
    'car'       : 1.50,
    'bus'       : 3.20,
    'truck'     : 3.50,
    'cow'       : 1.30,
    'dog'       : 0.55,
}


class CombinedPipeline:
    AEB_WARNING_M  = 30.0
    AEB_CRITICAL_M = 15.0

    def __init__(self):
        print("[INFO] Initialising Combined Pipeline...")
        print(f"[INFO] Camera: Qubo Dashcam Pro 4K")
        print(f"[INFO] Focal length: {FOCAL_LENGTH_PX}px | FOV: {FOV_DEGREES}deg")
        print("[INFO] Loading detection model - yolo11m...")
        self.detector = YOLODetector(model_path='yolo11m.pt')
        print("[INFO] Loading segmentation model - yolo11m-seg...")
        self.segmentor = YOLOSegmentor(model_path='yolo11m-seg.pt')
        print("[INFO] Combined pipeline ready")

    def estimate_distance(self, class_name, bbox_height_px):
        """
        Monocular distance estimation using pinhole camera model.
        Formula: distance = (real_height_m * focal_length_px) / bbox_height_px
        Calibrated for Qubo Dashcam Pro 4K at 720p resolution.
        Note: Wide angle (140deg FOV) causes barrel distortion —
              distances beyond 20m have higher error margin.
              LiDAR fusion planned for final semester accuracy improvement.
        """
        real_height = OBJECT_HEIGHTS_M.get(class_name, 1.70)
        if bbox_height_px < 5:
            return None
        distance = (real_height * FOCAL_LENGTH_PX) / bbox_height_px
        # Cap at reasonable range for 140deg wide angle camera
        if distance > 80.0:
            return None
        return round(distance, 1)

    def process_frame(self, frame):
        annotated = self.segmentor.segment(frame)
        detections = self.detector.detect(frame)
        aeb_triggered  = False
        closest_dist   = float('inf')
        closest_object = None
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            bbox_height_px  = y2 - y1
            distance        = self.estimate_distance(
                det['class_name'], bbox_height_px)
            color = (0, 0, 255) if det['aeb_critical'] else (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            if distance:
                label = f"{det['class_name']} {det['confidence']:.2f} | {distance}m"
            else:
                label = f"{det['class_name']} {det['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated,
                          (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            if det['aeb_critical'] and distance:
                if distance < closest_dist:
                    closest_dist   = distance
                    closest_object = det['class_name']
        if closest_object and closest_dist < self.AEB_CRITICAL_M:
            cv2.rectangle(annotated, (0, 0),
                          (annotated.shape[1], 55), (0, 0, 200), -1)
            cv2.putText(annotated,
                        f"AEB CRITICAL: {closest_object.upper()} AT {closest_dist}m",
                        (10, 38), cv2.FONT_HERSHEY_SIMPLEX,
                        0.95, (255, 255, 255), 2)
            aeb_triggered = True
        elif closest_object and closest_dist < self.AEB_WARNING_M:
            cv2.rectangle(annotated, (0, 0),
                          (annotated.shape[1], 55), (0, 140, 255), -1)
            cv2.putText(annotated,
                        f"AEB WARNING: {closest_object.upper()} AT {closest_dist}m",
                        (10, 38), cv2.FONT_HERSHEY_SIMPLEX,
                        0.95, (255, 255, 255), 2)
        return annotated, detections, aeb_triggered

    def run(self, source, output_path=None, show=True):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open: {source}")
            return
        src_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INFO] Source: {w}x{h} @ {src_fps}fps")
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(output_path, fourcc, src_fps, (w, h))
        frame_count = 0
        fps_list    = []
        aeb_count   = 0
        print("[INFO] Running... Press Q to stop")
        print("=" * 55)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            t_start = time.time()
            final, detections, aeb_trigger = self.process_frame(frame)
            fps_val = 1.0 / (time.time() - t_start + 1e-6)
            fps_list.append(fps_val)
            frame_count += 1
            if aeb_trigger:
                aeb_count += 1
            panel_y = final.shape[0] - 80
            cv2.rectangle(final, (0, panel_y),
                          (320, final.shape[0]), (0, 0, 0), -1)
            cv2.putText(final, f"FPS      : {fps_val:.1f}",
                        (10, panel_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(final, f"Objects  : {len(detections)}",
                        (10, panel_y + 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(final, f"AEB Alerts: {aeb_count}",
                        (10, panel_y + 64),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            lx = final.shape[1] - 220
            cv2.rectangle(final, (lx, 60), (lx + 210, 130), (0, 0, 0), -1)
            cv2.rectangle(final, (lx+5, 70), (lx+20, 85), (0, 0, 255), -1)
            cv2.putText(final, "AEB Critical", (lx+25, 83),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.rectangle(final, (lx+5, 95), (lx+20, 110), (0, 255, 0), -1)
            cv2.putText(final, "Non Critical", (lx+25, 108),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(final,
                        "Ibrahim Khaleel Shaik | BITS Pilani | Maruti Suzuki",
                        (10, final.shape[0] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
            if output_path:
                writer.write(final)
            if show:
                cv2.imshow(
                    'Indian Road Perception | Detection + Segmentation + AEB',
                    final)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
        cap.release()
        if output_path:
            writer.release()
        cv2.destroyAllWindows()
        avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
        print(f"\n{'='*55}")
        print(f"Frames processed : {frame_count}")
        print(f"Average FPS      : {avg_fps:.1f}")
        print(f"AEB alerts fired : {aeb_count}")
        print(f"Output saved     : {output_path}")
        print(f"{'='*55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Combined Indian Road Perception Pipeline')
    parser.add_argument('--source',
                        default='data/samples/demo_dashcam_1_720p.mp4')
    parser.add_argument('--output',
                        default='results/videos/combined_output.mp4')
    args   = parser.parse_args()
    source = int(args.source) if args.source == '0' else args.source
    Path('results/videos').mkdir(parents=True, exist_ok=True)
    pipeline = CombinedPipeline()
    pipeline.run(source, args.output)
