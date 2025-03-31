import os
import random
import numpy as np
import xml.etree.ElementTree as ET
from skimage.io import imread

def parse_voc_annotation(xml_file, desired_class="person"):
    """
    Parses a single PASCAL VOC XML annotation file.
    Returns a list of bounding boxes for the given desired_class.
    Each box is (xmin, ymin, xmax, ymax).

    If desired_class is None, return all bounding boxes + class names.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    boxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        if desired_class is None:
            boxes.append((xmin, ymin, xmax, ymax, name))
        else:
            if name == desired_class:
                boxes.append((xmin, ymin, xmax, ymax))

    return boxes

def gather_positive_patches(image_dir, anno_dir, desired_class="person", max_samples=None):
    """
    Gathers bounding box patches for 'desired_class' from PASCAL VOC structure.
    Returns a list of (image_array, label=1) for each positive sample.
    """
    images = []
    for filename in os.listdir(anno_dir):
        if not filename.endswith('.xml'):
            continue
        xml_path = os.path.join(anno_dir, filename)
        img_id = filename.replace('.xml', '')
        jpg_path = os.path.join(image_dir, img_id + '.jpg')
        if not os.path.isfile(jpg_path):
            continue

        boxes = parse_voc_annotation(xml_path, desired_class=desired_class)
        if len(boxes) == 0:
            continue

        # load the image
        img = imread(jpg_path)
        h, w = img.shape[:2]

        # for each bounding box, extract patch
        for (xmin, ymin, xmax, ymax) in boxes:
            xmin_c = max(0, xmin)
            ymin_c = max(0, ymin)
            xmax_c = min(w, xmax)
            ymax_c = min(h, ymax)

            patch = img[ymin_c:ymax_c, xmin_c:xmax_c]
            images.append((patch, 1))  # label=1 for positive

    if max_samples and len(images) > max_samples:
        random.shuffle(images)
        images = images[:max_samples]

    return images

def gather_negative_patches(image_dir, anno_dir, desired_class="person", max_samples=500):
    """
    Gathers negative patches (not containing 'desired_class') by sampling random
    regions in images that do NOT have that class.
    Returns a list of (image_array, label=0).
    """
    # First, find images that do NOT contain the desired_class
    neg_imgs = []
    for filename in os.listdir(anno_dir):
        if not filename.endswith('.xml'):
            continue
        xml_path = os.path.join(anno_dir, filename)
        boxes = parse_voc_annotation(xml_path, desired_class=desired_class)
        if len(boxes) == 0:
            # this image has no desired_class
            img_id = filename.replace('.xml', '')
            jpg_path = os.path.join(image_dir, img_id + '.jpg')
            if os.path.isfile(jpg_path):
                neg_imgs.append(jpg_path)

    random.shuffle(neg_imgs)
    neg_patches = []
    needed = max_samples

    for jpg_path in neg_imgs:
        if needed <= 0:
            break
        img = imread(jpg_path)
        h, w = img.shape[:2]

        # sample some random patches per image
        # example: take 5 random patches from this image
        for _ in range(5):
            if needed <= 0:
                break
            patch_width = random.randint(30, min(100, w))
            patch_height = random.randint(30, min(100, h))
            x0 = random.randint(0, w - patch_width)
            y0 = random.randint(0, h - patch_height)
            patch = img[y0:y0+patch_height, x0:x0+patch_width]
            neg_patches.append((patch, 0))
            needed -= 1

    return neg_patches
