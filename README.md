# CNN-based Object Detection & Edge Deployment

A practical pipeline demonstrating:
- **Pascal VOC** usage for object detection
- **Modern CNN-based models** (e.g., Faster R-CNN, YOLO)
- **GPU acceleration** (PyTorch, Torchvision, etc.)
- **Edge device inference** on resource-constrained hardware (Kria KV260, Jetson Orin NX, Raspberry Pi 5)

## Table of Contents
1. [Overview](#overview)  
2. [Features](#features)  
3. [Project Structure](#project-structure)  
4. [Installation](#installation)  
5. [Downloading the Pascal VOC Dataset](#downloading-the-pascal-voc-dataset)  
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

This repository, **EdgeAI-Vision-pipeline**, showcases **object detection** on the **Pascal VOC** dataset using **CNN-based** models such as **Faster R-CNN** (with a ResNet-50 backbone) or **YOLO**. We demonstrate **GPU-accelerated training** and **edge inference** on devices like the **NVIDIA Jetson Orin NX**, **Xilinx Kria KV260**, or **Raspberry Pi 5**. By integrating the **dataset parsing** directly in the training script, we eliminate the need for a separate `voc_utils.py` file, simplifying the workflow.

Key highlights:

- Demonstrates how to **download and parse the Pascal VOC dataset** for object detection.  
- Walks through **end-to-end training** of CNN-based detectors on VOC.  
- Explains **exporting** the trained model to standard formats (e.g., `.pth` or ONNX) for edge deployment.  
- Illustrates real-time **inference** on resource-constrained hardware with a camera feed.

---

## Features

- **Pascal VOC** – 20 object classes (person, dog, car, etc.).  
- **Modern CNNs** – E.g., **Faster R-CNN** with **ResNet-50** backbone, YOLO, or others.  
- **GPU Training** – Leverage PyTorch + CUDA for faster training.  
- **Edge Inference** – Minimal script to run bounding-box detection on low-power devices.  
- **Easily Adaptable** – Swap in your favorite detection model or hardware target.

---

## Project Structure

A typical layout might look like this:

```
.
├── VOCdevkit/
│   └── VOC2007/
│       ├── Annotations/
│       ├── JPEGImages/
│       ├── ImageSets/
│       └── ...
├── src/
│   ├── train_detector.py   # Main training script for a CNN-based model
│   ├── infer_edge.py       # Script to run real-time inference (live camera)
│   └── models/             # (Optional) custom model code or configs
├── README.md
└── requirements.txt
```

Feel free to reorganize as needed.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/MultiModel-Edge-AI/EdgeAI-Vision-pipeline.git
cd EdgeAI-Vision-pipeline
```

### 2. Install Dependencies

```bash
pip install -r dependencies.txt
```

*(If you’re using Conda, you can instead create a conda environment and install PyTorch + Torchvision per your GPU drivers.)*

---

## Downloading the Pascal VOC Dataset

1. **Download** the [VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) (or VOC2012) train/val data, such as `VOCtrainval_06-Nov-2007.tar`.  
2. **Extract** it, resulting in a structure like:
   ```
   VOCdevkit/
     VOC2007/
       Annotations/
       JPEGImages/
       ImageSets/
       ...
   ```
3. **Check** your `train_detector.py` for a path variable (e.g., `VOC_ROOT = "VOCdevkit/VOC2007"`) and update it to match your local directory.

---

## Usage

### Data Preparation

For **Faster R-CNN** with PyTorch, you can directly use `torchvision.datasets.VOCDetection` or define a small wrapper inside `train_detector.py` to parse bounding-box data. No separate `voc_utils.py` is required; simply ensure your code references the correct `Annotations/` and `JPEGImages/` paths.

### Model Training

An example command:

```bash
python src/train_detector.py
```

- Loads Pascal VOC from `VOCdevkit/VOC2007`.  
- Builds a **ResNet-based Faster R-CNN** (or YOLO, depending on your script).  
- **Trains** on GPU if available, logs losses, etc.  
- Saves the final model to something like `fasterrcnn_voc.pth`.

### Evaluation

Depending on your script, you may:

- Check bounding-box metrics like **mAP** (mean average precision) on a validation split.  
- Inspect detection outputs on a handful of images.  

### Inference on Edge Devices

After training:

1. **Copy** your saved model file (e.g. `fasterrcnn_voc.pth`) to the edge device.
2. **Install** the appropriate environment (PyTorch or ONNX runtime, plus OpenCV).
3. **Run**:

```bash
python src/infer_edge.py fasterrcnn_voc.pth
```

The script should:

- Open a camera feed,  
- Convert frames to PyTorch tensors,  
- Run the CNN to predict bounding boxes + labels,  
- Draw them on the live video feed.

---

## Results

- **mAP**: Ranges from 50–80% depending on the model (e.g., Faster R-CNN, YOLOv5, etc.).  
- **Real-Time Inference**: 
  - Desktop GPU can easily reach 20–30+ FPS.  
  - On **Jetson Orin NX** or **Kria KV260**, you may get a few FPS out-of-the-box. Hardware-accelerated frameworks (e.g., TensorRT, Vitis AI) can improve this.  
- **Optimizations**: If speed is insufficient, consider smaller backbones (MobileNet), pruning/quantization, or specialized hardware acceleration.

---

## Troubleshooting

1. **Dataset Paths**  
   - Make sure your script points to the correct `VOCdevkit/VOC2007`.  
2. **CUDA Device**  
   - If you don’t have an NVIDIA GPU or correct drivers, training will fall back to CPU.  
3. **Low mAP**  
   - Ensure you aren’t mixing train/test sets. Check data augmentation or hyperparameters.  
4. **Inference Speed**  
   - Smaller models or hardware-accelerated deployments can help.

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
4. **Open a Pull Request** into the main branch.

We welcome community contributions, bug reports, and suggestions!

---

## License

This project is licensed under the [MIT License](LICENSE). You’re free to use, modify, and distribute the code as allowed by that license.

---

## References

1. **Pascal VOC** – [Official Site](http://host.robots.ox.ac.uk/pascal/VOC/)  
2. **PyTorch Torchvision** – [Docs](https://pytorch.org/vision/stable/index.html)  
3. **NVIDIA Jetson** – [Developer Site](https://developer.nvidia.com/embedded-computing)  
4. **Kria KV260** – [Xilinx Documentation](https://www.xilinx.com/products/som/kria/kv260-vision-starter-kit.html)  
5. **Raspberry Pi** – [Official Site](https://www.raspberrypi.com/)  

---

_Thank you for visiting **EdgeAI-Vision-pipeline**! If you have any questions, suggestions, or issues, feel free to [open an issue](../../issues) or reach out._
