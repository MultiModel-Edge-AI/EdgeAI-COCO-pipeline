# COCO-EdgeVision: HOG-Based Image Classification & Edge Deployment

A practical pipeline demonstrating:
- **COCO dataset** usage for classification
- **HOG feature extraction** (using `scikit-image`)
- **Multiple ML models** (e.g Decision Tree, etc.)
- **GPU acceleration** (XGBoost with CUDA)  
- **Edge device inference** on resource-constrained hardware (e.g., Kria KV260, Nvidia Jetson Orin NX)

## Table of Contents
1. [Overview](#overview)  
2. [Features](#features)  
3. [Project Structure](#project-structure)  
4. [Installation](#installation)  
5. [Downloading the COCO Dataset](#downloading-the-coco-dataset)  
6. [Usage](#usage)  
   1. [Data Preparation](#data-preparation)  
   2. [Model Training](#model-training)  
   3. [Evaluation](#evaluation)  
   4. [Inference on Edge Devices](#inference-on-edge-devices)  
7. [Results](#results)  
8. [Troubleshooting](#troubleshooting)  
9. [Contributing](#contributing)  
10. [License](#license)  
11. [References](#references)  

---

## Overview

This repository contains code to **classify food-related images** by leveraging the **COCO 2017 dataset** and extracting **HOG (Histogram of Oriented Gradients) features**. The classification models (such as a **Decision Tree** or **XGBoost**) can then be **deployed on edge devices** for real-time inference, even on low-power hardware.

Key highlights:

- Demonstrates how to **download and parse the COCO dataset** for a custom classification task.  
- Walks through **HOG feature extraction** on each image.  
- Uses **XGBoost with GPU support** (optional) for faster training, or a simpler **Decision Tree** classifier if GPU is unavailable.  
- Explains **exporting the trained model** and **loading** it on devices like the Kria KV260 and Nvidia Jetson Orin NX.  

---

## Features

- **COCO-based classification**: Select categories (like _pizza_, _hot dog_, _donut_, _cake_) for single-label classification.  
- **HOG feature extraction**: CPU-based approach using `scikit-image`.  
- **Multiple ML options**: XGBoost (GPU-accelerated) or other tree-based models (e.g., Decision Tree).  
- **Edge inference**: Minimal script to run **live camera classification** on devices with limited resources.  
- **Easy extension**: Adapt the pipeline to different categories, additional classes, or more advanced feature extraction methods.  

---

## Project Structure

A typical directory layout might look like this:

```
.
├── coco2017/
│   ├── train2017/
│   ├── val2017/
│   └── annotations/
│       ├── instances_train2017.json
│       └── instances_val2017.json
├── src/
│   ├── train_xgboost.py       # Main training script for XGBoost
│   ├── train_decision_tree.py # Example script for Decision Tree
│   ├── inference_edge.py      # Script to run inference (live camera or images)
│   ├── hog_extraction.py  # HOG feature extraction functions
│   └── coco_helpers.py    # COCO annotation parsing helpers
├── README.md
└── requirements.txt
```

Feel free to reorganize to match your preferences.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/MultiModel-Edge-AI/EdgeAI-COCO-pipeline.git
cd EdgeAI-COCO-pipeline
```

### 2. Install Dependencies

- Using `dependencies.txt`:
```bash
pip install -r dependencies.txt
```

---

## Downloading the COCO Dataset

1. **Create a directory** for COCO (e.g., `coco2017`) if you haven’t already.

2. **Download** train/val images and annotations from <https://cocodataset.org/#download>:

   ```bash
   mkdir coco2017
   cd coco2017
   wget http://images.cocodataset.org/zips/train2017.zip
   wget http://images.cocodataset.org/zips/val2017.zip
   wget http://images.cocodataset.org/zips/test2017.zip
   wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
   ```

3. **Unzip** the files:

   ```bash
   unzip train2017.zip
   unzip val2017.zip
   unzip test2017.zip
   unzip annotations_trainval2017.zip
   ```

   Your folder should contain `train2017/`, `val2017/`, `test2017/`, and `annotations/`.

4. **Check** that you have:
   ```
   coco2017/
   ├── train2017/
   ├── val2017/
   ├── test2017/
   ├── annotations/
      ├── instances_train2017.json
      ├── instances_val2017.json
       ...
   ```

---

## Usage

### Data Preparation

This project **filters** COCO images containing specific food classes (e.g., pizza, hot dog, donut, cake) and discards images with multiple categories. You can modify the code to handle multi-label or additional classes.

1. **Edit the list of target classes** in your training script (e.g., `TARGET_CLASSES = ['pizza','hot dog','donut','cake']`).
2. **Set** any optional parameters (like `MAX_IMAGES_PER_CLASS`) if you want to limit dataset size for a quick test.

### Model Training

In `src/train_xgboost.py` (for example):

```bash
python src/train_xgboost.py
```

What it does:
- Reads the COCO annotations (`instances_train2017.json`, `instances_val2017.json`) via `pycocotools`.
- Filters out images that don’t match your target classes or that contain multiple classes.
- Extracts **HOG features** for each selected image.
- Trains an **XGBoost** classifier (optionally on GPU if available).
- Outputs performance metrics (confusion matrix, accuracy).
- Saves the trained model as `my_xgb_model.json`.

*(For a **Decision Tree** or other algorithms, see `src/train_decision_tree.py` or a similar script.)*

### Evaluation

The script typically:
- Uses **COCO’s `val2017`** subset to evaluate.  
- Prints a **confusion matrix** and class-specific error rates.  
- Reports the **overall accuracy**.

Example output snippet:
```
Confusion Matrix (Validation Set):
[[123   1   0   2]
 [  3 134   5   0]
 [  0   2 145   1]
 [  4   0   3 118]]

Error Rates by Class:
 pizza: 0.024
 hot dog: 0.056
 donut: 0.020
 cake: 0.055

Overall Val Accuracy: 0.922
```

*(The actual numbers depend on your dataset filters and hyperparameters.)*

### Inference on Edge Devices

After training, we have a **serialized model** (e.g., `my_xgb_model.json`). Copy it to your edge device, then run an **inference script**. For example, `inference_edge.py`:

```bash
python src/inference_edge.py my_xgb_model.json
```

- Captures frames from a camera (using OpenCV).
- Extracts HOG features on each frame.
- Runs `model.predict()` to classify the image (pizza, hot dog, donut, or cake).
- Prints or overlays the label on the video feed.

On a **Kria KV260** or **Nvidia Jetson Orin NX**, you’d need:

```bash
pip install xgboost scikit-image opencv-python
```

> **Note**: If your edge device has limited CPU/GPU resources, consider smaller models, fewer classes, or **faster** feature extraction approaches.

---

## Results

**Potential results** from a typical run might be:

- **Accuracy** on COCO val set: ~85–95% (depending on classes, sample size, HOG parameters, and your model’s hyperparameters).
- **Inference speed**:  
  - On a **Desktop GPU**, tens of frames per second.  
  - On a **Nvidia Jetson Orin NX**, 1–2 FPS might be more realistic if using HOG + XGBoost on CPU.
  - On a **Kria KV260**, 1–2 FPS might be more realistic if using HOG + XGBoost on CPU.  

- **100% accuracy** might indicate data leakage (e.g., very small test set or mis-labeled data). Always verify your approach is correct.

---

## Troubleshooting

1. **Skipping Images**:  
   - If you see “Could not find class for image …” or skipping many images, ensure your script is filtering classes properly.  
2. **XGBoost Device Mismatch**:  
   - If you see warnings about “mismatched devices,” your data might be on CPU while the model is on GPU. Usually harmless but can slow inference.  
3. **No GPU**:  
   - If your edge device doesn’t support CUDA, remove `device='cuda'` from the XGBoost parameters (or use a CPU-based classifier).  
4. **Performance**:  
   - Large-scale HOG extraction can be slow. Consider limiting data or optimizing with GPU-based libraries (like CuPy) for preprocessing.

---

## Contributing

1. **Fork** this repository.  
2. **Create** a new branch for your feature/fix:
   ```bash
   git checkout -b feature-my-improvement
   ```
3. **Commit** your changes and push to your fork:
   ```bash
   git commit -m "Add my new feature"
   git push origin feature-my-improvement
   ```
4. **Open a Pull Request** to merge into the main branch.

We welcome suggestions, bug reports, and community contributions!

---

## License

This project is licensed under the [MIT License](LICENSE). You’re free to use, modify, and distribute the code as allowed by that license.

---

## References

1. **COCO Dataset** – [Official Website](https://cocodataset.org/)  
2. **XGBoost** – [Official GitHub](https://github.com/dmlc/xgboost), [Documentation](https://xgboost.readthedocs.io/)  
3. **scikit-image** – [Docs](https://scikit-image.org/docs/stable/)  
4. **pycocotools** – [GitHub](https://github.com/cocodataset/cocoapi)  
6. **NVIDIA Jetson** – [Developer Site](https://developer.nvidia.com/embedded-computing)

---

_Thank you for checking out **COCO-EdgeVision**! If you have any questions, suggestions, or issues, feel free to [open an issue](../../issues) or reach out._
