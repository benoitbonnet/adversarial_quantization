import torch
import numpy as np
import torch.nn as nn
from scipy import misc
from glob import glob
import cv2
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision
import os
import sys
import random

class AdversarialStegoDataset(Dataset):
    """Dataset for SRNet training"""

    def __init__(self, natural_dirs, adversarial_dirs, device='cpu'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.nat_images = natural_dirs
        self.adv_images = adversarial_dirs
        self.device=device

    def adjust_paths(self):
        new_nats = []
        for nat_path in self.nat_images:
            for adv_path in self.adv_images:
                if nat_path[-9:]==adv_path[-9:]:
                    new_nats.append(nat_path)
                    break
        self.nat_images = new_nats

    def __len__(self):
        return len(self.nat_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        nat_path = self.nat_images[idx]
        adv_path = self.adv_images[idx]
        nat_image = cv2.imread(nat_path)
        adv_image = cv2.imread(adv_path)

        nat_tensor = torch.autograd.Variable(torch.Tensor(nat_image).long(),requires_grad=False)
        adv_tensor = torch.autograd.Variable(torch.Tensor(adv_image).long(),requires_grad=False)

        nat_tensor = nat_tensor.permute(2,0,1).to(self.device)
        adv_tensor = adv_tensor.permute(2,0,1).to(self.device)
        sample = {0: nat_tensor, 1: adv_tensor}
        return sample

BATCH_SIZE = 16
COVER_PATH = '/nfs/nas4/bbonnet/bbonnet/datasets/images/test/'
STEGO_PATH = '/nfs/nas4/bbonnet/bbonnet/datasets/images/test_advs/PGD1_QUAD_nat/'

"""
FGSM1_GINA_nat	FGSM1_L2_rob	PGD1_GINA_nat  PGD1_HILL_nat  PGD1_L2_nat  PGD1_QUAD_nat  PGD2_GINA_nat  PGD2_HILL_nat	PGD2_QUAD_nat
FGSM1_L2_nat	FGSM2_GINA_nat	PGD1_GINA_rob  PGD1_HILL_rob  PGD1_L2_rob  PGD1_QUAD_rob  PGD2_GINA_rob  PGD2_HILL_rob	PGD2_QUAD_rob
"""


device=('cpu:0')
device=('cuda:0')

cover_image_names = glob(COVER_PATH+"*png")
stego_image_names = glob(STEGO_PATH+"*png")
print(len(stego_image_names))
cover_image_names.sort()
stego_image_names.sort()

stego_image_names = stego_image_names[:len(stego_image_names)]
cover_image_names = cover_image_names[:len(cover_image_names)]

print(len(cover_image_names), len(stego_image_names))
cover_labels = np.zeros((len(cover_image_names)))
stego_labels = np.ones((len(stego_image_names)))

train_set = AdversarialStegoDataset(cover_image_names,stego_image_names, device=device)

print(len(train_set), 'before adjust')
train_set.adjust_paths()
print(len(train_set), 'after adjust')

train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

last_ep = 0

distos = 0
total = 0
for i, data in enumerate(train_loader, 0):
    cover_data = data[0]
    stego_data = data[1]
    batch_s,c,h,w = cover_data.shape
    for batch_nb in range(batch_s):
        cov = cover_data[batch_nb].float()
        steg = stego_data[batch_nb].float()
        distos+= (cov-steg).norm()
        total+=1
    #print(distos)
print(distos/total, total/843.0)
