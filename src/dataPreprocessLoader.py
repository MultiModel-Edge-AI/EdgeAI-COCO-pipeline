import numpy as np
import os
from pycocotools.coco import COCO
from skimage import transform, io
from sklearn import preprocessing
import torch

trainImgDir = "COCO/train2017"
valImgDir = "COCO/val2017"
trainAnnotations = "COCO/annotations/instances_train2017.json"
valAnnotations = "COCO/annotations/instances_val2017.json"

trainObj = COCO(trainAnnotations)
trainImgIds = trainObj.getImgIds()
valObj = COCO(valAnnotations)
valImgIds = valObj.getImgIds()


##### GETTING DATA LOADED PROPERLY #####
def extractLabels(obj, img_ids):
    # right now this does single-label
    data = []

    for img_id in img_ids:
        ann_ids = obj.getAnnIds(imgIds=[img_id], iscrowd=None)

        if len(ann_ids) == 0:
            continue
        
        #change for multilabel
        ann = obj.loadAnns([ann_ids[0]])[0]
        cat_id = ann['category_id']
        cat_info = obj.loadCats([cat_id])[0] 
        cat_name = cat_info['name']

        img_info = obj.loadImgs([img_id])[0]
        file_name = img_info['file_name']
        full_path = os.path.join(valImgDir, file_name)

        if os.path.isfile(full_path):
            data.append((full_path, cat_name))
    return data

trainData = extractLabels(trainObj,trainImgIds)
valData = extractLabels(valObj,valImgIds)


##### PREPROCESS AND FORMAT DATA #####
def transformImage(image_path, output_size=(224, 224)):
    img = io.imread(image_path)
    img = transform.resize(img, output_size)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    img = img.transpose(2,0,1)
    return img

def createDataloader(image_label_pairs, output_size=(224, 224)):
    X_list, y_list = [], []
    for (fpath, lbl) in image_label_pairs:
        img = transformImage(fpath, output_size=output_size)
        X_list.append(img)
        y_list.append(lbl)
    
    y = preprocessing.LabelEncoder().fit_transform(np.array(y_list))
    X = torch.tensor(np.array(X_list, dtype=np.float32))
    y = torch.tensor(y)
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=32)
    return dataloader

valDataloader = createDataloader(valData)
trainDataloader = createDataloader(trainData)

