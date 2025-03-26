import sys
import numpy as np
import cv2
import xgboost as xgb
# from sklearn.tree import DecisionTreeClassifier  # if you want to load a scikit-learn model
# import joblib

from hog_extraction import extract_hog_feature

# If you used a label encoder, you need the same class order:
TARGET_CLASSES = ['pizza', 'hot dog', 'donut', 'cake']

def main():
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python inference_edge.py <model_file.json>")
        print("Example: python inference_edge.py my_xgb_model.json")
        sys.exit(1)
    
    model_path = sys.argv[1]

    # Load model
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    print(f"[INFO] Loaded XGBoost model from {model_path}")

    # If using a scikit-learn Decision Tree:
    # dt_model = joblib.load('my_decision_tree.pkl')

    # Initialize camera
    cap = cv2.VideoCapture(0)  # 0 for default webcam
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        sys.exit(1)

    print("[INFO] Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera frame capture failed.")
            break

        # Extract HOG features for the frame
        hog_vec = extract_hog_feature(frame, output_size=(128,128))

        # Predict
        X_input = np.array([hog_vec], dtype=np.float32)
        y_pred = model.predict(X_input)  # numeric label
        pred_idx = int(y_pred[0])
        if 0 <= pred_idx < len(TARGET_CLASSES):
            pred_label = TARGET_CLASSES[pred_idx]
        else:
            pred_label = "Unknown"

        # Display result
        cv2.putText(frame, pred_label, (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Edge AI Inference", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()