# inference_edge.py

import sys
import cv2
import joblib
import numpy as np

from detect import sliding_window_detect, non_max_suppression

def main():
    if len(sys.argv) < 2:
        print("Usage: python inference_edge.py xgb_detector.pkl")
        sys.exit(1)

    model_path = sys.argv[1]
    print("[INFO] Loading model:", model_path)
    model = joblib.load(model_path)
    print("[INFO] Model loaded. Starting camera...")

    cap = cv2.VideoCapture(0)  # default camera
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        sys.exit(1)

    print("[INFO] Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera error.")
            break

        # run naive detection
        detections = sliding_window_detect(
            img=frame,
            model=model,
            win_size=(64,64),
            step=24,               # bigger step = faster, fewer boxes
            scale_factors=[1.0,1.3],
            score_thresh=0.6
        )

        # NMS
        final_boxes = non_max_suppression(detections, iou_thresh=0.3)

        # draw bboxes
        for (xmin,ymin,xmax,ymax,score) in final_boxes:
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (0,255,0), 2)
            cv2.putText(frame, f"{score:.2f}", (xmin,ymin-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
