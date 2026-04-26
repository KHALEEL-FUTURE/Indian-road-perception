# Combined Detection + Segmentation — no display
import cv2
import argparse
import numpy as np
from pathlib import Path
from ultralytics import YOLO

COLORS = [
    (255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),
    (255,0,255),(128,255,0),(0,128,255),(255,0,128),(128,0,255),
    (0,255,128),(255,128,0),(0,128,128),(128,128,0),(128,0,0)
]

FOCAL_LENGTH_PX = 233.0
OBJECT_HEIGHTS  = {
    'person':1.70,'rider':1.50,'motorcycle':1.50,
    'bicycle':1.10,'autorickshaw':1.80,'car':1.50,
    'truck':3.50,'bus':3.20,'animal':1.30
}
AEB_CLASSES = [0,1,2,3,4,9]

def estimate_distance(cls_name, bbox_h):
    real_h = OBJECT_HEIGHTS.get(cls_name, 1.70)
    if bbox_h < 5:
        return None
    d = (real_h * FOCAL_LENGTH_PX) / bbox_h
    return round(d, 1) if d < 80 else None

def run_combined(source, output, det_model, seg_model, skip=3):
    print(f"[INFO] Loading detection model: {det_model}")
    detector  = YOLO(det_model)
    print(f"[INFO] Loading segmentation model: {seg_model}")
    segmentor = YOLO(seg_model)

    cap   = cv2.VideoCapture(source)
    fps   = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] {w}x{h} @ {fps}fps | {total} frames")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    writer      = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
    frame_count = 0
    last_frame  = None
    aeb_count   = 0
    print("[INFO] Processing combined pipeline...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % skip == 0:
            # Segmentation layer
            seg_results = segmentor(frame, conf=0.45, verbose=False)[0]
            annotated   = frame.copy()
            if seg_results.masks is not None:
                masks   = seg_results.masks.data.cpu().numpy()
                classes = seg_results.boxes.cls.cpu().numpy().astype(int)
                for mask, cls_id in zip(masks, classes):
                    color        = COLORS[cls_id % len(COLORS)]
                    mask_resized = cv2.resize(mask,(w,h))
                    bool_mask    = mask_resized > 0.5
                    annotated[bool_mask] = (
                        annotated[bool_mask]*0.45 +
                        np.array(color)*0.55
                    ).astype(np.uint8)

            # Detection layer
            det_results     = detector(frame, conf=0.45, verbose=False)[0]
            closest_dist    = float('inf')
            closest_obj     = None

            for box in det_results.boxes:
                cls_id       = int(box.cls[0])
                conf         = float(box.conf[0])
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                cls_name     = detector.model.names[cls_id]
                distance     = estimate_distance(cls_name, y2-y1)
                color        = (0,0,255) if cls_id in AEB_CLASSES else (0,255,0)
                cv2.rectangle(annotated,(x1,y1),(x2,y2),color,2)
                label = f"{cls_name} {conf:.2f}"
                if distance:
                    label += f" | {distance}m"
                    if cls_id in AEB_CLASSES and distance < closest_dist:
                        closest_dist = distance
                        closest_obj  = cls_name
                cv2.putText(annotated, label,(x1,y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2)

            # AEB logic
            if closest_obj and closest_dist < 15:
                cv2.rectangle(annotated,(0,0),(w,55),(0,0,200),-1)
                cv2.putText(annotated,
                           f"AEB CRITICAL: {closest_obj.upper()} AT {closest_dist}m",
                           (10,38),cv2.FONT_HERSHEY_SIMPLEX,0.95,(255,255,255),2)
                aeb_count += 1
            elif closest_obj and closest_dist < 30:
                cv2.rectangle(annotated,(0,0),(w,55),(0,140,255),-1)
                cv2.putText(annotated,
                           f"AEB WARNING: {closest_obj.upper()} AT {closest_dist}m",
                           (10,38),cv2.FONT_HERSHEY_SIMPLEX,0.95,(255,255,255),2)

            # Info panel
            cv2.putText(annotated,f"Frame:{frame_count}/{total}",
                       (10,h-50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
            cv2.putText(annotated,f"AEB Alerts:{aeb_count}",
                       (10,h-25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
            cv2.putText(annotated,"Ibrahim Khaleel Shaik | BITS Pilani",
                       (10,h-5),cv2.FONT_HERSHEY_SIMPLEX,0.45,(200,200,200),1)
            last_frame = annotated

        out_frame = last_frame if last_frame is not None else frame
        writer.write(out_frame)
        frame_count += 1

        if frame_count % 500 == 0:
            print(f"  {frame_count}/{total} ({frame_count/total*100:.1f}%)")

    cap.release()
    writer.release()
    print(f"\n[DONE] Frames: {frame_count} | AEB alerts: {aeb_count}")
    print(f"[DONE] Saved : {output}")

parser = argparse.ArgumentParser()
parser.add_argument('--source',    required=True)
parser.add_argument('--output',    required=True)
parser.add_argument('--det-model', default='india_vru_best.pt')
parser.add_argument('--seg-model', default='yolo11m-seg.pt')
parser.add_argument('--skip',      type=int, default=3)
args = parser.parse_args()
run_combined(args.source, args.output,
             args.det_model, args.seg_model, args.skip)
