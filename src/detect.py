# detect.py

import numpy as np
import cv2
import xgboost as xgb
from skimage.transform import resize
from skimage.feature import hog

def sliding_window_detect(
    img,
    model,
    win_size=(64,64),
    step=16,
    scale_factors=[1.0, 1.2, 1.5],
    score_thresh=0.5
):
    """
    Naive sliding-window detection:
      - For each scale in scale_factors:
        - Resize the image
        - Slide a window of size win_size across
        - Extract HOG, run model.predict_proba
        - If prob > score_thresh => store bounding box

    Returns a list of (xmin, ymin, xmax, ymax, score).
    """

    detections = []

    orig_h, orig_w = img.shape[:2]
    for scale in scale_factors:
        new_w = int(orig_w / scale)
        new_h = int(orig_h / scale)

        if new_w < win_size[0] or new_h < win_size[1]:
            # scaling too large, skip
            continue

        # resized image for scanning
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        for y in range(0, new_h - win_size[1], step):
            for x in range(0, new_w - win_size[0], step):
                patch = resized[y:y+win_size[1], x:x+win_size[0]]
                # HOG
                feat = hog(
                    cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY),
                    orientations=9,
                    pixels_per_cell=(8,8),
                    cells_per_block=(2,2),
                    block_norm='L2-Hys',
                    transform_sqrt=True
                )

                X_input = np.array([feat], dtype=np.float32)
                # Probability of class=1
                prob = model.predict_proba(X_input)[0,1]

                if prob >= score_thresh:
                    # scale back to original image coords
                    xmin = int(x * scale)
                    ymin = int(y * scale)
                    xmax = int((x+win_size[0]) * scale)
                    ymax = int((y+win_size[1]) * scale)

                    detections.append((xmin, ymin, xmax, ymax, prob))

    return detections

def non_max_suppression(detections, iou_thresh=0.3):
    """
    Basic NMS on bounding boxes with scores.
    detections: list of (xmin, ymin, xmax, ymax, score)
    Returns a filtered list of bboxes.
    """
    if not detections:
        return []

    # sort by score desc
    dets = sorted(detections, key=lambda x: x[4], reverse=True)
    kept = []

    while dets:
        best = dets.pop(0)
        kept.append(best)
        to_remove = []
        for i, d in enumerate(dets):
            iou = box_iou(best, d)
            if iou > iou_thresh:
                to_remove.append(i)
        for idx in reversed(to_remove):
            del dets[idx]
    return kept

def box_iou(a, b):
    """
    Intersection over Union for two boxes a,b = (xmin, ymin, xmax, ymax, score).
    """
    ax1, ay1, ax2, ay2, _ = a
    bx1, by1, bx2, by2, _ = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)

    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union