"""
train_detector.py

Trains a Faster R-CNN (ResNet-50 backbone) on Pascal VOC (2007 or 2012).
GPU-accelerated if available.
"""

import os
import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import VOCDetection
import numpy as np

#Dataset Wrapper to produce target dictionary inline with what PyTorch expects
class VOCDetectionWrapper(VOCDetection):
    def __init__(self, root, year="2007", image_set="train", transforms=None):
        super().__init__(root=root, year=year, image_set=image_set, transform=None, target_transform=None, transforms=None)
        self.transforms = transforms

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        #target is a dict with "annotation" containing bounding boxes, labels, etc.
        annotation = target["annotation"]

        #parse annotation into PyTorch "instances" format
        boxes = []
        labels = []
        if not isinstance(annotation["object"], list):
            #if there's only one object, wrap it in a list
            objects = [annotation["object"]]
        else:
            objects = annotation["object"]

        for obj in objects:
            name = obj["name"]
            #Typically, you'd map class name -> an integer label
            #We'll do something naive, just to show the process.
            #For Pascal VOC, we have 20 classes. Let's do a simple mapping:
            label_id = CLASS_NAME_TO_ID.get(name, None)
            if label_id is None:
                #skip unknown classes or handle them as background
                continue

            bndbox = obj["bndbox"]
            xmin = float(bndbox["xmin"])
            ymin = float(bndbox["ymin"])
            xmax = float(bndbox["xmax"])
            ymax = float(bndbox["ymax"])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label_id)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        new_target = {}
        new_target["boxes"] = boxes
        new_target["labels"] = labels
        #optional: some models expect "image_id", "area", "iscrowd", etc.
        image_id = torch.tensor([idx])
        new_target["image_id"] = image_id

        if self.transforms:
            img = self.transforms(img)

        return img, new_target


#20 classes + background for a total of 21
CLASS_NAMES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"
]
CLASS_NAME_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES) if i != "background"}

def get_transform(train=True):
    #transformations
    transforms_list = []
    transforms_list.append(T.ToTensor())  # convert PIL -> torch
    if train:
        #adding random flips, etc., for better data augmentation
        transforms_list.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms_list)

def main():
    #paths
    VOC_ROOT = "C:/repos/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #building dataset and dataLoaders
    train_dataset = VOCDetectionWrapper(
        root=VOC_ROOT,
        year="2007",
        image_set="train", 
        transforms=get_transform(train=True)
    )
    val_dataset = VOCDetectionWrapper(
        root=VOC_ROOT,
        year="2007",
        image_set="val",
        transforms=get_transform(train=False)
    )

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=2, shuffle=False, num_workers=2, collate_fn=collate_fn)

    #creating a Faster R-CNN model and using ResNet-50 backbone - prebuilt from torchvision
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT"  # or None if you want random init
    )
    #number of classes: 21 for VOC (20 + background)
    num_classes = 21
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )

    model.to(device)

    #running an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=1e-4)

    #training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for images, targets in train_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)  # forward pass -> get losses
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()

            total_loss += float(losses)

        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] Training Loss: {avg_loss:.3f}")

        # Evaluate quickly (not fully implemented here)
        # Typically you'd do a bounding-box mAP check. This snippet just prints partial info.
        model.eval()
        with torch.no_grad():
            # do 1 batch check
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                outputs = model(images)
                # outputs is a list of dict with "boxes", "labels", "scores"
                # ...
                break  # just one batch

    #save model to run inference
    torch.save(model.state_dict(), "fasterrcnn_voc.pth")
    print("[INFO] Model saved to fasterrcnn_voc.pth")


def collate_fn(batch):
    return tuple(zip(*batch))

if __name__ == "__main__":
    main()
