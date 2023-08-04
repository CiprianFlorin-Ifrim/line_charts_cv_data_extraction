#!/usr/bin/env python
# coding: utf-8

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------LIBRARIES---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import VOCDetection
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
from torch.cuda.amp import autocast, GradScaler
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
import torch.distributed as dist
from datetime import timedelta
import gc
import time
import warnings

warnings.filterwarnings("ignore")

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

def collate_fn(batch):
	return tuple(zip(*batch))

def train(rank, world_size):
	torch.manual_seed(0)
	torch.cuda.set_device(rank)
	device = torch.device(f'cuda:{rank}')

	# initialize the process group
	dist.init_process_group("gloo", rank=rank, world_size=world_size, timeout=timedelta(minutes=int(1e6)))     # there seem to be some timeout issues with my network, this high value stops the system from crashing, could be a windows NIC issue

	# Load the pretrained model
	model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
	num_classes = 6
	in_features = model.roi_heads.box_predictor.cls_score.in_features
	model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

	model.to(device)

	# Data transform
	mean = torch.tensor([0.485, 0.456, 0.406])  # mean values for ImageNet
	std = torch.tensor([0.229, 0.224, 0.225])   # standard deviation values for ImageNet

	data_transforms = transforms.Compose([
		transforms.Resize(512), 
		transforms.ToTensor(),
		transforms.Normalize(mean=mean, std=std)
	])

	# Apply this function to your dataset using the transforms parameter
	train_data = CustomVOCDetection(
		root="pascal_voc_datasets/",
		dataset_name="Plots_Experimental",
		image_set="testing",
		transforms=data_transforms,
		classes=CUSTOM_CLASSES 
	)

	val_data = CustomVOCDetection(
		root="pascal_voc_datasets/",
		dataset_name="Plots_Experimental",
		image_set="testing",  # assuming the set name is 'validation'
		transforms=data_transforms,
		classes=CUSTOM_CLASSES 
	)

	# Define optimizer
	params = [p for p in model.parameters() if p.requires_grad]
	#optimizer = torch.optim.SGD(params, lr=0.0001, momentum=0.9, weight_decay=0.0005)
	optimizer = torch.optim.Adam(params, lr=0.0001, weight_decay=0.0005)

	# Initialize the gradient scaler
	scaler = GradScaler()

	# Initialize loss histories for plotting
	loss_hist = []
	valid_loss_hist = []

	# Add a path for the checkpoint
	MODEL_NAME = "EXPERIMENTAL_8_rcnn_batch-16_epoch-30_crypto.com-experimental_non-augmented"
	MODEL_EXTENSION = ".pt"
	MODEL_SAVE_DIR = "pytorch_rcnn_models/"
	MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_NAME + MODEL_EXTENSION)
	CHECKPOINT_DIR = "pytorch_rcnn_checkpoints/"
	CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, MODEL_NAME + "/")

	# Check if model and checkpoint directories exist, if not, create them
	os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
	os.makedirs(CHECKPOINT_DIR, exist_ok=True)
	os.makedirs(CHECKPOINT_PATH, exist_ok=True)

	# Load the latest checkpoint if exists
	start_epoch = 0
	if os.path.exists(CHECKPOINT_PATH):
		try:
			checkpoint_files = [f for f in os.listdir(CHECKPOINT_PATH) if f.endswith('.pth')]
			checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))  # sort by epoch number
			latest_checkpoint = checkpoint_files[-1]
			checkpoint = torch.load(os.path.join(CHECKPOINT_PATH, latest_checkpoint))

			model.load_state_dict(checkpoint['model_state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
			start_epoch = checkpoint['epoch'] + 1
			loss_hist = checkpoint['loss_hist']
			if rank == 0: print(f"Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")

		except Exception as e:
			if rank == 0: print(f"No checkpoint found at {CHECKPOINT_PATH} or loading failed. Starting from scratch. Error: {str(e)}")
	else:
		if rank == 0: print(f"No checkpoint directory found at {CHECKPOINT_PATH}. Starting from scratch.")
		
	# Wrap the model for usage with multiple GPUs if available
	if torch.cuda.device_count() > 1:
		if rank == 0: print("Using", torch.cuda.device_count(), "GPUs!")
		model = torch.nn.parallel.DistributedDataParallel(model.to(device), device_ids=[device])
		
	# create dataloaders
	train_sampler = DistributedSampler(train_data)
	val_sampler = DistributedSampler(val_data)

	train_data_loader = DataLoader(train_data, batch_size=16, sampler=train_sampler, num_workers = 0, pin_memory=True, collate_fn=collate_fn)
	val_data_loader = DataLoader(val_data, batch_size=16, sampler=val_sampler, num_workers = 0, pin_memory=True, collate_fn=collate_fn)

	# Training loop
	num_epochs = 30
	for epoch in range(start_epoch, num_epochs):
		gc.collect()
		if torch.cuda.is_available():
			if rank == 0: 
				for i in range(torch.cuda.device_count()):
					print(f'GPU {i+1}/{torch.cuda.device_count()}: {torch.cuda.get_device_name(i)}')
					print('Memory Usage:')
					print('Allocated:', round(torch.cuda.memory_allocated(i)/1024**3,1), 'GB')
					print('Cached:   ', round(torch.cuda.memory_reserved(i)/1024**3,1), 'GB')
					print('-------------------------------------')
				
		# Model Training
		torch.cuda.empty_cache()         # clear cuda cache
		model.train()

		loss_epoch = []
		if rank == 0:
			progress_bar = tqdm(train_data_loader, desc=f"Training epoch {epoch+1}/{num_epochs}", unit="batch")
		else:
			progress_bar = train_data_loader
		
		for images, targets in progress_bar:
			images = list(image.to(device) for image in images)
			targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

			optimizer.zero_grad()

			with autocast():
				loss_dict = model(images, targets)
				losses = sum(loss for loss in loss_dict.values())

			scaler.scale(losses).backward()
			scaler.step(optimizer)
			scaler.update()

			# Wrap the loss in a tensor.
			# We'll use dist.all_reduce to sum it across all processes.
			loss_tensor = torch.tensor(losses.item()).to(device)
			dist.all_reduce(loss_tensor)
			loss_tensor /= world_size  # Average loss across all processes

			loss_value = loss_tensor.item()
			loss_epoch.append(loss_value)
			if rank == 0: progress_bar.set_postfix({"batch_loss": loss_value})

		epoch_loss = sum(loss_epoch)/len(loss_epoch)
		loss_hist.append(epoch_loss)
		if rank == 0: print(f"Epoch loss: {epoch_loss}")

		# Validation Loop
		torch.cuda.empty_cache()         # clear cuda cache
		valid_loss = 0

		with torch.no_grad():
			if rank == 0:
				progress_bar = tqdm(val_data_loader, desc=f"Training epoch {epoch+1}/{num_epochs}", unit="batch")
			else:
				progress_bar = val_data_loader

			for images, targets in progress_bar:
				images = list(image.to(device) for image in images)
				targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

				# Forward
				loss_dict = model(images, targets)
				losses = sum(loss for loss in loss_dict.values())

				# Wrap the loss in a tensor.
				# We'll use dist.all_reduce to sum it across all processes.
				loss_tensor = torch.tensor(losses.item()).to(device)
				dist.all_reduce(loss_tensor)
				loss_tensor /= world_size  # Average loss across all processes

				valid_loss += loss_tensor.item()

			# Average validation loss
			valid_loss /= len(val_data_loader)
			valid_loss_hist.append(valid_loss)
			if rank == 0: print(f"Validation loss: {valid_loss}")

		# Save the model checkpoint at the end of each epoch
		if rank == 0:
			torch.save({
				'epoch': epoch,
				'model_state_dict': model.module.state_dict(),  # Save state_dict of the model, not the DDP wrapper
				'optimizer_state_dict': optimizer.state_dict(),
				'loss': epoch_loss,
				'loss_hist': loss_hist,  
				'valid_loss_hist': valid_loss_hist,
			}, os.path.join(CHECKPOINT_PATH, f"checkpoint_epoch_{epoch+1}.pth"))

		torch.distributed.barrier()  # Synchronize at the end of each epoch

	# Save the model after training
	if rank == 0:
		torch.save(model.module.state_dict(), MODEL_SAVE_PATH)  # distributed dataparallel model save
		print("The model has been saved!")

def main():
	os.environ["OMP_NUM_THREADS"] = "16"             # 16 is too high, same with 12, 4 is too low, 8 is perfect
	world_size = torch.cuda.device_count()
	torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
	main()
