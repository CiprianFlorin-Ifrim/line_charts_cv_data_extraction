import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import VOCDetection
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_fscore_support
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from typing import Dict
import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image
import xml.etree.ElementTree as ET
import collections
from torchvision import transforms
import pandas as pd
import torch.multiprocessing as mp
import multiprocessing

CUSTOM_CLASSES = {"name": 1, "value": 2, "x-axis": 3, "y-axis": 4, "plot":5}
PASSTHROUGH_FIELDS = ['folder', 'filename', 'source', 'size', 'segmented', 'object']

class CustomVOCDetection(Dataset):
    def __init__(self, root, dataset_name, image_set='train', transforms=None, classes=None):
        self.root = root
        self.classes = classes
        
        voc_root = os.path.join(self.root, 'VOCdevkit', dataset_name)
        image_dir = os.path.join(voc_root, 'JPEGImages')
        annotation_dir = os.path.join(voc_root, 'Annotations')

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.')

        splits_dir = os.path.join(voc_root, 'ImageSets', 'Main')
        
        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.annotations = [os.path.join(annotation_dir, x + ".xml") for x in file_names]
        self.transforms = transforms

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if len(img.getbands()) != 3:
            print(f"Image at {self.images[index]} does not have 3 channels after conversion to RGB")
            
        # Get the original image size
        width, height = img.size
        target = self.parse_voc_xml(
            ET.parse(self.annotations[index]).getroot())

        if self.transforms is not None:
            img = self.transforms(img)
            #print(f'Image shape after transform: {img.shape}')  # Debugging print
        target = transform_voc_target(target, width, height)         # Pass the width and height to the function

        return img, target

    def __len__(self):
        return len(self.images)

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag in PASSTHROUGH_FIELDS:
                voc_dict[node.tag] = [def_dic[ind][0] if len(def_dic[ind]) == 1 else def_dic[ind] for ind in def_dic]
            else:
                voc_dict[node.tag] = {ind: def_dic[ind][0] if len(def_dic[ind]) == 1 else def_dic[ind] for ind in def_dic}
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict

def transform_voc_target(target, width, height):
    boxes = []
    labels = []
    for obj in target["annotation"]["object"]:
        class_name = obj[0]
        bbox = obj[-1]
        # Normalize the bounding box coordinates
        boxes.append([float(bbox["xmin"]) / width, float(bbox["ymin"]) / height, float(bbox["xmax"]) / width, float(bbox["ymax"]) / height])
        if class_name in CUSTOM_CLASSES:
            labels.append(CUSTOM_CLASSES[class_name])
        else:
            print(f"Warning: {class_name} is not in CUSTOM_CLASSES")
            # you might want to handle this situation better
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.int64)
    
    # Hash the filename to a unique numeric value
    image_id = torch.tensor([hash(target["annotation"]["filename"])])
    target = {}
    target["boxes"] = boxes
    target["labels"] = labels
    target["image_id"] = image_id

    return target