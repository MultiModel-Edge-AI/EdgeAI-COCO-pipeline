"""
infer_edge.py

Loads the trained Faster R-CNN model, runs real-time bounding-box detection
on a webcam feed. GPU-accelerated if available.
"""

import sys
import cv2
import torch
import torchvision
import numpy as np

#class list from training
CLASS_NAMES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"
]

def main():
    if len(sys.argv) < 2:
        print("Usage: python infer_edge.py fasterrcnn_voc.pth")
        sys.exit(1)

    model_path = sys.argv[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #building model architecture model architecture as training
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT"
    )
    num_classes = 21
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = \
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    #load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"[INFO] Loaded model from {model_path}")

    #open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        sys.exit(1)

    print("[INFO] Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera read error.")
            break

        #convert BGR (OpenCV) to RGB (PyTorch)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #convert to tensor
        img_tensor = torch.from_numpy(rgb_frame).permute(2,0,1).float().to(device)
        img_tensor /= 255.0  # normalize 0..1

        #run inference
        with torch.no_grad():
            outputs = model([img_tensor])  # list of dict
        # each element in `outputs` has 'boxes', 'labels', 'scores'
        out = outputs[0]

        boxes = out["boxes"].cpu().numpy()
        labels = out["labels"].cpu().numpy()
        scores = out["scores"].cpu().numpy()

        #visualizaton
        for box, label, score in zip(boxes, labels, scores):
            if score < 0.5:
                continue
            x1, y1, x2, y2 = box.astype(int)
            cls_name = CLASS_NAMES[label] if label < len(CLASS_NAMES) else "Unknown"
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"{cls_name} {score:.2f}", (x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        cv2.imshow("Faster R-CNN Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Inference ended.")

if __name__ == "__main__":
    main()
