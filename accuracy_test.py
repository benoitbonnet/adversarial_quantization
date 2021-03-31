import torch
import numpy as np
import cv2
import sys
import os
from attacks.attack_gen import attack_generator
from quantizers.quantizer import Quantizer
from data_utils.data_utils import  AdversarialDataset,AttackParser, load_model
import argparse
import foolbox as fb

parser = AttackParser(argparse.ArgumentParser())
parser.labels_path = ('/nfs/nas4/bbonnet/bbonnet/datasets/labels/imagenet_2012_val/processed.csv')
parser.input_path = '/nfs/nas4/bbonnet/bbonnet/datasets/images/test/'

orig_set = AdversarialDataset(parser)
orig_loader = torch.utils.data.DataLoader(orig_set, batch_size=parser.batch_size, shuffle=False)

print(len(orig_loader), " batches")

for model_cpt,model_name in enumerate(parser.models):
    images = 0
    correctly_predicted = 0
    model, device = load_model(parser, model_cpt)
    for i, data_batch in enumerate(orig_loader, 0):
        orig_batch, initial_label, image_nbs = data_batch

        model.zero_grad()
        prediction = model(orig_batch)
        pred_label = torch.argmax(prediction,axis=-1)

        correctly_predicted+=(pred_label==initial_label).float().sum()
        images += orig_batch.shape[0]

    print("{}% accuracy on {} images. Model: {}".format(correctly_predicted/images*100, images, model_name))
