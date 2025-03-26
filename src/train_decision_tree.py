import os
import numpy as np
from pycocotools.coco import COCO
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

# Local imports (adjust if your files are in a "utils" subfolder)
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

    # Limit how many images are used (optional, for faster tests)
    MAX_TRAIN_IMAGES = 500
    MAX_VAL_IMAGES   = 100

    # Load Annotations
    print("[INFO] Loading train annotations...")
    coco_train = COCO(TRAIN_ANN_PATH)
    print("[INFO] Loading val annotations...")
    coco_val   = COCO(VAL_ANN_PATH)

    # Data Gathering
    print("[INFO] Gathering train data...")
    train_data = gather_all_coco_images(
        coco_obj=coco_train,
        img_dir=TRAIN_IMG_DIR,
        max_images=MAX_TRAIN_IMAGES
    )

    print("[INFO] Gathering val data...")
    val_data   = gather_all_coco_images(
        coco_obj=coco_val,
        img_dir=VAL_IMG_DIR,
        max_images=MAX_VAL_IMAGES
    )

    print(f"[INFO] Found {len(train_data)} training images.")
    print(f"[INFO] Found {len(val_data)} validation images.")

    if len(train_data) == 0:
        print("[ERROR] No training images found. Exiting.")
        return

    # Build Dataset for HOG
    print("[INFO] Extracting HOG features for training set...")
    X_train, y_train_raw = build_dataset(train_data, output_size=(128,128))

    print("[INFO] Extracting HOG features for validation set...")
    X_val,   y_val_raw   = build_dataset(val_data,   output_size=(128,128))

    # Label Encoding
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    y_val   = label_encoder.transform(y_val_raw) if len(y_val_raw) > 0 else []

    # Defining Decision Tree Classifier
    dt_model = DecisionTreeClassifier(min_samples_leaf=15, random_state=42)

    # Model Training
    print("[INFO] Training Decision Tree...")
    dt_model.fit(X_train, y_train)
    print("[INFO] Training complete.")

    # Model Evaluation
    if len(X_val) > 0:
        print("[INFO] Evaluating on validation set...")
        y_val_pred = dt_model.predict(X_val)
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
        total_images_val  = cm_val.sum()
        val_error = 1.0 - (total_correct_val / total_images_val)
        print(f"\nOverall Val Error: {val_error:.3f}")
        print(f"Overall Val Accuracy: {1 - val_error:.3f}")
    else:
        print("[WARNING] No validation data to evaluate.")

    # Save model to a JSON to run inference on edg device
    import joblib
    joblib.dump(dt_model, 'my_decision_tree.pkl')
    print("[INFO] Decision Tree saved to my_decision_tree.pkl")

if __name__ == "__main__":
    main()