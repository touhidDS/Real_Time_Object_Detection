import argparse
import time

import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time object detection & tracking (local)")
    parser.add_argument("--source", default="0", help="Webcam index or path to a video file")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 weights (n/s/m)")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--max-age", type=int, default=30, help="Frames to keep a lost track before dropping it")
    parser.add_argument("--n-init", type=int, default=3, help="Detections needed in a row before a track gets an ID")
    parser.add_argument("--max-cosine-distance", type=float, default=0.3,
                         help="Appearance match leniency for DeepSORT (0-1, higher = fewer ID switches)")
    parser.add_argument("--save", default=None, help="Path to save the annotated output video")
    return parser.parse_args()


def main():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    print(f"[INFO] Loading model: {args.model}")
    model = YOLO(args.model)
    tracker = DeepSort(
        max_age=args.max_age,
        n_init=args.n_init,
        max_cosine_distance=args.max_cosine_distance,
        embedder="mobilenet",  # appearance-based matching, holds up better through brief occlusion than IoU alone
    )

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    writer = None
    if args.save:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))
        print(f"[INFO] Saving annotated output to: {args.save}")

    print("[INFO] Press 'q' to quit.")
    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream / cannot read frame.")
            break

        results = model(frame, verbose=False, conf=args.conf)[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, model.names[cls]))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            cls_name = track.get_det_class() or "object"
            l, t, r, b = map(int, track.to_ltrb())

            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
            label = f"{cls_name} | ID {track_id}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (l, t - th - 8), (l + tw + 4, t), (0, 255, 0), -1)
            cv2.putText(frame, label, (l + 2, t - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        now = time.time()
        fps_display = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Object Detection & Tracking (press 'q' to quit)", frame)

        if writer is not None:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
