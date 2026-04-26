# Segmentation only — no display
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

def run_segmentation(source, output, model_path, skip=2):
    print(f"[INFO] Loading segmentation model: {model_path}")
    model = YOLO(model_path)
    cap   = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source}")
        return

    fps   = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] {w}x{h} @ {fps}fps | {total} frames")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output, fourcc, fps, (w, h))

    frame_count  = 0
    last_overlay = None
    print("[INFO] Processing segmentation...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % skip == 0:
            results  = model(frame, conf=0.45, verbose=False)[0]
            overlay  = frame.copy()
            if results.masks is not None:
                masks   = results.masks.data.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy().astype(int)
                for mask, cls_id in zip(masks, classes):
                    color        = COLORS[cls_id % len(COLORS)]
                    mask_resized = cv2.resize(mask,(w,h))
                    bool_mask    = mask_resized > 0.5
                    overlay[bool_mask] = (
                        overlay[bool_mask] * 0.45 +
                        np.array(color) * 0.55
                    ).astype(np.uint8)
            last_overlay = overlay

        out_frame = last_overlay if last_overlay is not None else frame
        cv2.putText(out_frame, f"Frame:{frame_count}/{total}",
                   (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv2.putText(out_frame, "Ibrahim Khaleel Shaik | BITS Pilani",
                   (10,h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        writer.write(out_frame)
        frame_count += 1

        if frame_count % 500 == 0:
            print(f"  {frame_count}/{total} ({frame_count/total*100:.1f}%)")

    cap.release()
    writer.release()
    print(f"\n[DONE] Frames: {frame_count}")
    print(f"[DONE] Saved : {output}")

parser = argparse.ArgumentParser()
parser.add_argument('--source', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--model', default='yolo11m-seg.pt')
parser.add_argument('--skip', type=int, default=3)
args = parser.parse_args()
run_segmentation(args.source, args.output, args.model, args.skip)
