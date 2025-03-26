import os
import random
import numpy as np
import xgboost as xgb
from pycocotools.coco import COCO
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

from coco_helpers import gather_all_coco_images
from hog_extraction import build_dataset

def main():
    # Directory Configuration
    # Update these paths to your actual COCO dataset location
    COCO_ROOT = '/Users/shreyas/Repos/coco2017'
    TRAIN_IMG_DIR = os.path.join(COCO_ROOT, 'train2017')
    VAL_IMG_DIR   = os.path.join(COCO_ROOT, 'val2017')
    
    TRAIN_ANN_PATH = os.path.join(COCO_ROOT, 'annotations', 'instances_train2017.json')
    VAL_ANN_PATH   = os.path.join(COCO_ROOT, 'annotations', 'instances_val2017.json')

    # Limit how many images you load so that trainign and testing and quicker
    # Increase ONLY if you have time/compute resources.
    #MAX_TRAIN_IMAGES = 500
    #MAX_VAL_IMAGES   = 100

    # Load annotations
    print("[INFO] Loading train annotations...")
    coco_train = COCO(TRAIN_ANN_PATH)
    print("[INFO] Loading val annotations...")
    coco_val   = COCO(VAL_ANN_PATH)

    # Data gathering
    # We collect all images and ignore multiple objects
    # We pick the FIRST annotation's category as the label.
    print("[INFO] Gathering train data (first annotation as label)...")
    train_data = gather_all_coco_images(
        coco_obj=coco_train,
        img_dir=TRAIN_IMG_DIR,
        #max_images=MAX_TRAIN_IMAGES
    )
    
    print("[INFO] Gathering val data (first annotation as label)...")
    val_data   = gather_all_coco_images(
        coco_obj=coco_val,
        img_dir=VAL_IMG_DIR,
        #max_images=MAX_VAL_IMAGES
    )

    print(f"[INFO] Found {len(train_data)} training images.")
    print(f"[INFO] Found {len(val_data)} validation images.")

    if len(train_data) == 0:
        print("[ERROR] No training images found. Exiting.")
        return

    # Build Dataset for HOG
    print("[INFO] Extracting HOG features for train set...")
    X_train, y_train_raw = build_dataset(train_data, output_size=(128,128))

    print("[INFO] Extracting HOG features for val set...")
    X_val,   y_val_raw   = build_dataset(val_data,   output_size=(128,128))

    # Label Encoding
    # Combining all labels
    all_labels_raw = y_train_raw + y_val_raw

    # Fit once on the union of train+val labels
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels_raw)

    # Then transform each subset
    y_train = label_encoder.transform(y_train_raw)
    y_val   = label_encoder.transform(y_val_raw)

    # Defining XGBoost Classifier
    # Use CPU or GPU if you have one:
    #  - device='cpu'
    #  - device='cuda'   (for NVIDIA)
    #  - device='mps'    (for Apple Silicon GPUs)
    model = xgb.XGBClassifier(
        tree_method='hist',
        device='cpu',  
        n_estimators=10,
        max_depth=8,
        random_state=42
    )

    # Model Training
    print("[INFO] Training XGBoost model...")
    model.fit(X_train, y_train)
    print("[INFO] Training complete.")

    # Model evaluation
    if len(X_val) > 0:
        print("[INFO] Evaluating on validation set...")
        y_val_pred = model.predict(X_val)

        cm_val = confusion_matrix(y_val, y_val_pred, labels=np.arange(len(label_encoder.classes_)))
        print("\nConfusion Matrix (Validation Set):\n", cm_val)

        row_sums = cm_val.sum(axis=1)
        num_classes = len(label_encoder.classes_)
        print("\nError Rates by Class (Val Set):")
        for i in range(num_classes):
            class_name = label_encoder.classes_[i]
            correct = cm_val[i, i]
            total   = row_sums[i]
            err_rate = 0.0 if total == 0 else 1.0 - (correct / total)
            print(f"  {class_name}: {err_rate:.3f}")

        total_correct_val = np.trace(cm_val)
        total_images_val = cm_val.sum()
        val_error = 1.0 - (total_correct_val / total_images_val)
        print(f"\nOverall Val Error: {val_error:.3f}")
        print(f"Overall Val Accuracy: {1 - val_error:.3f}")
    else:
        print("[WARNING] No validation data to evaluate.")

    # Saving model to a JSON to run inference on Edge hardware
    model.save_model('my_xgb_model.json')
    print("[INFO] Model saved to my_xgb_model.json")

if __name__ == "__main__":
    main()