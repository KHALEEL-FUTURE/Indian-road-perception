# Detection script without display
import cv2
import time
import argparse
from pathlib import Path
from ultralytics import YOLO

def run_detection(source, output, model_path):
    print(f"[INFO] Loading model: {model_path}")
    model = YOLO(model_path)
    print("[INFO] Model loaded")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Video: {w}x{h} @ {fps}fps")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output, fourcc, fps, (w, h))

    frame_count = 0
    print("[INFO] Processing... please wait")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.45, verbose=False)[0]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            label  = f"{model.names[cls_id]} {conf:.2f}"
            color  = (0,0,255) if cls_id in [0,1,2,3,4] else (0,255,0)
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,label,(x1,y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

        cv2.putText(frame,f"Frame:{frame_count}",(10,30),
                   cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
        cv2.putText(frame,"Ibrahim Khaleel Shaik | BITS Pilani",
                   (10,h-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),1)

        writer.write(frame)
        frame_count += 1

        if frame_count % 100 == 0:
            print(f"  Processed {frame_count} frames...")

    cap.release()
    writer.release()
    print(f"\n[DONE] Frames: {frame_count}")
    print(f"[DONE] Saved: {output}")

parser = argparse.ArgumentParser()
parser.add_argument('--source', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--model', default='india_vru_best.pt')
args = parser.parse_args()

run_detection(args.source, args.output, args.model)
