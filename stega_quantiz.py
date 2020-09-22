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
parser.add_argument('--outputs', type=str, default='outputs/', help='path to store adversarial images')
parser.add_argument('--model', type=str, default='efficientnet-b0', choices=['resnet18', 'resnet50', 'efficientnet-b0'], required=False, help="network to attack")
parser.add_argument('--advprop', type=bool, default=False, choices=[True, False], required=False, help="For efficientnets only, advprop == robust training")
parser.add_argument('--attack', type=str, default='FGSM', choices=['FGSM', 'PGD', 'DDN', 'CW2'], required=False, help="attack used")
parser.add_argument('--quantizer', type=str, default='L2', choices=['L2', 'GINA', 'HILL', 'QUAD'], required=False, help="quantizer used")
parser.add_argument('--max_degree', type=int, default='1', required=False, help="maximum distortion on a given pixel from the adversarial perturbation")
parser.add_argument('--batch_size', type=int, default='6', required=False, help="batch size")
parser.add_argument('--labels_path', type=str, default='/nfs/nas4/bbonnet/bbonnet/datasets/labels/imagenet_2012_val/processed.csv', required=False, help="labels of the images, default will result in attacking the image regardless of the initial prediction")
parser.add_argument('--gpu', type=bool, default=True, required=False, help="determines usage of gpu device")

labels_path = ('/nfs/nas4/bbonnet/bbonnet/datasets/labels/imagenet_2012_val/processed.csv')
args = parser.parse_args()
print(args)

input_path = args.inputs
save_path  = args.outputs
model_name = args.model
robust     = args.advprop
attack_name= args.attack
quantizer  = args.quantizer
max_deg    = args.max_degree
gpu        = args.gpu
labels_path= args.labels_path
batch_size = args.batch_size

print("Starting {} with {} quantization (max distortion from adversarial perturbation = {})".format(attack_name, quantizer, max_deg))
print("attacking images in {}".format(input_path))
print("storing results in {}".format(save_path))
print("model used {} with advprop=".format(model_name,robust) if 'efficient' in model_name else "model used {}".format(model_name))
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

orig_paths = glob.glob(input_path+'/*png')
orig_paths.sort()
orig_paths = orig_paths
if not os.path.isdir(save_path):
    os.makedirs(save_path)

orig_set = AdversarialDataset(orig_paths,labels=labels_path, device=device)
orig_loader = torch.utils.data.DataLoader(orig_set, batch_size=batch_size, shuffle=False)#, num_workers=0)

epsilon = 1

attack = attack_generator(attack_name, device)
quantizer = Quantizer(model, 1000, device, max_deg, quantizer)

#l1_norms = np.zeros(len(orig_loader))
#l1_norms_2 = np.zeros(len(orig_loader))
cpt1=0
print(len(orig_loader), " batches")
cpt = 0
for i, data_batch in enumerate(orig_loader, 0):
    orig_batch, initial_label, image_nbs = data_batch

    model.zero_grad()
    prediction_before = model(orig_batch)
    init_pred_label = torch.argmax(prediction_before,axis=-1)
    adversarial_image = attack.attack(model, orig_batch, initial_label)
    pred_adv = model(adversarial_image)
    adv_label = torch.argmax(pred_adv,axis=-1)
    quantized_adv = quantizer.quantize_samples(adversarial_image, orig_batch, initial_label)

    for batch_image in range(orig_batch.shape[0]):
            if init_pred_label[batch_image]!=initial_label[batch_image]:
                pass
            adv_element, orig_element, quant_element, label_element, adv_label_element = adversarial_image[batch_image], orig_batch[batch_image], quantized_adv[batch_image], initial_label[batch_image], adv_label[batch_image]
            prediction_after_quant = model(quant_element.unsqueeze(0))
            quant_label = torch.argmax(prediction_after_quant,axis=-1)
            print((adv_element-orig_element).norm(), (quant_element-orig_element).norm(), quant_label, label_element)

            if label_element.item()==quant_label.item():
                cpt1+=1
                print('{} failed'.format(cpt1))
            else:
                if initial_label[batch_image].item()==init_pred_label[batch_image].item():
                    img_rgb = cv2.cvtColor(np.float32(quantized_adv[0].cpu()), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(save_path+"/{}.png".format(image_nbs[0][batch_image]),img_rgb)
