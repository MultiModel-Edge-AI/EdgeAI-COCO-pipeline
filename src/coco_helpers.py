import os
import random
from pycocotools.coco import COCO

def gather_all_coco_images(coco_obj, img_dir, max_images=None):

    # Returns a list of (file_path, label_string) for all images in coco_obj. We'll pick the *first annotation* of each image as its label, ignoring other objects.

    data = []
    img_ids = coco_obj.getImgIds()
    random.shuffle(img_ids)  # so we pick a random subset if we limit

    if max_images is not None:
        img_ids = img_ids[:max_images]

    for img_id in img_ids:
        ann_ids = coco_obj.getAnnIds(imgIds=[img_id], iscrowd=None)
        if len(ann_ids) == 0:
            # skip images with no annotations
            continue

        # Use just the first annotation as the label
        ann = coco_obj.loadAnns([ann_ids[0]])[0] 
        cat_id = ann['category_id']
        cat_info = coco_obj.loadCats([cat_id])[0]
        cat_name = cat_info['name']  # e.g. 'person', 'pizza', 'dog'

        # Load the actual image info
        img_info = coco_obj.loadImgs([img_id])[0]
        file_name = img_info['file_name']
        full_path = os.path.join(img_dir, file_name)

        if os.path.isfile(full_path):
            data.append((full_path, cat_name))

    return data