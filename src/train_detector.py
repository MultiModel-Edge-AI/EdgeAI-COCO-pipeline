import os
import numpy as np
import cupy as cp
import xgboost as xgb
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.transform import resize

from voc_utils import gather_positive_patches, gather_negative_patches

def extract_hog_patch(img, size=(64,64)):
    """
    Convert patch to grayscale, resize, extract HOG.
    Return feature vector.
    """
    if img.ndim == 3:
        img = rgb2gray(img)
    img_resized = resize(img, size)
    feature_vec = hog(
        img_resized,
        orientations=9,
        pixels_per_cell=(8,8),
        cells_per_block=(2,2),
        block_norm='L2-Hys',
        transform_sqrt=True
    )
    return feature_vec

def main():
    # 1. Paths for the PASCAL VOC
    # Suppose you have VOC2007 or VOC2012 organized like:
    # VOCROOT/
    #    JPEGImages/*.jpg
    #    Annotations/*.xml
    VOC_ROOT = "C:/repos/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007"
    IMG_DIR = os.path.join(VOC_ROOT, "JPEGImages")
    ANNO_DIR = os.path.join(VOC_ROOT, "Annotations")

    CLASS_NAME = "person"  # or "dog", "cat", etc.
    # For multiple classes, you'd repeat the pipeline or do a multi-class strategy.

    # 2. Gather positive patches
    print("[INFO] Gathering positive patches...")
    pos_patches = gather_positive_patches(
        image_dir=IMG_DIR,
        anno_dir=ANNO_DIR,
        desired_class=CLASS_NAME,
        max_samples=1000  # adjust as needed
    )

    # 3. Gather negative patches
    print("[INFO] Gathering negative patches...")
    neg_patches = gather_negative_patches(
        image_dir=IMG_DIR,
        anno_dir=ANNO_DIR,
        desired_class=CLASS_NAME,
        max_samples=1000
    )

    # 4. Combine
    data = pos_patches + neg_patches
    print(f"[INFO] Total patches: {len(data)}")

    # 5. Extract HOG features
    X_list, y_list = [], []
    for (patch, label) in data:
        feats = extract_hog_patch(patch, size=(64,64))
        X_list.append(feats)
        y_list.append(label)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    print("[INFO] Feature matrix shape:", X.shape)
    # 6. Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 7. Define XGBoost
    model = xgb.XGBClassifier(
        tree_method='hist',
        device='cuda',  # or 'cpu'
        n_estimators=50,
        max_depth=8,
        random_state=42
    )

    print("[INFO] Training XGBoost classifier...")
    model.fit(X_train, y_train)
    print("[INFO] Training done.")

    # 8. Evaluate
    import cupy as cp
    X_test_gpu = cp.asarray(X_test)
    y_pred = model.predict(X_test_gpu)
    acc = accuracy_score(y_test, y_pred)
    print("[INFO] Test Accuracy:", acc)

    # 9. Save the model
    joblib.dump(model, "xgb_detector.pkl")
    print("[INFO] Model saved to xgb_detector.pkl")

if __name__ == "__main__":
    main()
