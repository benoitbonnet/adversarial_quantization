import torch
import numpy as np
import pandas
from torchvision import models, datasets
import cv2
import torch.nn as nn
import time
import glob
import os
import sys
from attacks.attack_gen import attack_generator
from quantizers.quantizer import Quantizer
from models.efficientnet_pytorch import EfficientNet
from data_utils.data_utils import Preprocessing_Layer, AdversarialDataset, Preprocessing_Layer_robust

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--inputs', type=str, default='inputs', help='path to folder containing images')
parser.add_argument('--model', type=str, default='efficientnet-b0', choices=['resnet18', 'resnet50', 'efficientnet-b0'], required=False, help="network to attack")
parser.add_argument('--advprop', type=bool, default=False, choices=[True, False], required=False, help="For efficientnets only, advprop == robust training")
parser.add_argument('--batch_size', type=int, default='32', required=False, help="batch size")
parser.add_argument('--labels_path', type=str, default='/nfs/nas4/bbonnet/bbonnet/datasets/labels/imagenet_2012_val/processed.csv', required=False, help="labels of the images, default will result in attacking the image regardless of the initial prediction")
parser.add_argument('--gpu', type=bool, default=True, required=False, help="determines usage of gpu device")

args = parser.parse_args()

input_path = "/nfs/nas4/bbonnet/bbonnet/datasets/images/test"#args.inputs
model_name = args.model
robust     = args.advprop
gpu        = args.gpu
labels_path= args.labels_path
batch_size = args.batch_size

print("testing accuracy on {} with advprop=".format(model_name,robust) if 'efficient' in model_name else "model used {}".format(model_name))
print("batch size {}".format(batch_size))

efficient_net = True if 'efficientnet' in model_name else False
device = ('cuda:0') if gpu else ('cpu:0')


if efficient_net:
    model_init = EfficientNet.from_pretrained(model_name, advprop=robust)
else:
    model_init = getattr(models,model_name)(pretrained=True)

if robust:
    preprocess_layer = Preprocessing_Layer_robust(torch_device=device)
else:
    preprocess_layer = Preprocessing_Layer(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225], torch_device=device)

softmax = nn.Softmax(dim=1)
model = nn.Sequential(preprocess_layer, model_init, softmax)
model.eval();
model.to(device)

orig_paths = glob.glob(input_path+'/*')

orig_paths.sort()
orig_paths = orig_paths

orig_set = AdversarialDataset(orig_paths,labels=labels_path, device=device)
orig_loader = torch.utils.data.DataLoader(orig_set, batch_size=batch_size, shuffle=False)

print(len(orig_loader), " batches")
images = 0
correctly_predicted = 0
for i, data_batch in enumerate(orig_loader, 0):
    orig_batch, initial_label, image_nbs = data_batch

    model.zero_grad()
    prediction = model(orig_batch)
    pred_label = torch.argmax(prediction,axis=-1)

    correctly_predicted+=(pred_label==initial_label).float().sum()
    images += orig_batch.shape[0]

print("{}% accuracy on {} images".format(correctly_predicted/images*100, images))
