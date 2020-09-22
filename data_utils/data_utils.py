import torch
import numpy as np
import pandas
from torchvision import models, datasets
import cv2
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

class AdversarialDataset(Dataset):
    """Dataset for SRNet training"""

    def __init__(self, natural_dirs,  labels= None ,device='cpu', img_size = 224):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.nat_images = natural_dirs
        self.device=device
        self.img_size = img_size
        self.labels = self.get_labels(labels)
    def __len__(self):
        return len(self.nat_images)

    def image_name(self, idx):
        file_name_index = self.nat_images[idx][::-1].index('/')
        return self.nat_images[idx][-file_name_index:-4]

    def get_labels(self, labels_path):
        labels = pandas.read_csv(labels_path)
        labels_val = labels.values
        labels_dict = {}
        for value_tuple in labels_val:
            labels_dict[value_tuple[0]]=value_tuple[1]
        return(labels_dict)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        nat_path = self.nat_images[idx]
        nat_image = cv2.imread(nat_path)[..., ::-1]
        nat_image = cv2.resize(nat_image, (self.img_size, self.img_size))
        nat_tensor = torch.autograd.Variable(torch.Tensor(nat_image),requires_grad=False)

        image_name = self.image_name(idx)
        label = int(self.labels['{}.png'.format(image_name)][1:-1])

        label = torch.tensor(label).to(self.device)
        image_name = [image_name]

        return (nat_tensor.to(self.device), label, image_name)



class Preprocessing_Layer(torch.nn.Module):
    """
    This is an added layer to the pytorch model, it allows to back-propagate gradients through the preprocessing of the images
    and then work on [0,255] images instead of the preprocessed domain. -> model = nn.Sequential(Preprocessing_Layer(), model)
    This preprocessing works on ResNet models as well as EfficientNet (when not trained with adv prop) with the following values:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    For EfficientNet trained with adv_prop, see Preprocessing_Layer_robust
    """
    def __init__(self, mean, std, torch_device='cpu'):
        super(Preprocessing_Layer, self).__init__()
        self.mean = mean
        self.std = std
        self.device = torch_device

    def preprocess(self, img, mean, std):
        image = img.clone().to(self.device)
        image /= 255.0

        image = image.transpose(1, 3).transpose(2, 3)
        image[:,0,:,:] = (image[:,0,:,:] - mean[0]) / std[0]
        image[:,1,:,:] = (image[:,1,:,:] - mean[1]) / std[1]
        image[:,2,:,:] = (image[:,2,:,:] - mean[2]) / std[2]

        return(image)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        res = self.preprocess(x, self.mean, self.std)
        return res

class Preprocessing_Layer_robust(torch.nn.Module):
    """
    This is an added layer to the pytorch model, it allows to back-propagate gradients through the preprocessing of the images
    and then work on [0,255] images instead of the preprocessed domain. -> model = nn.Sequential(Preprocessing_Layer(), model)
    This preprocessing works on EfficientNet trained with adv prop
    """
    def __init__(self,torch_device='cpu'):
        super(Preprocessing_Layer_robust, self).__init__()
        self.device = torch_device

    def forward(self, x):
        image = x/255.0

        image = image.transpose(1, 3).transpose(2, 3)
        res = image * 2.0 - 1.0
        res = res.to(self.device)
        return res

class AdversarialSample(torch.nn.Module):
    def __init__(self, initial_label, adversarial_label, adv_image, orig_image, torch_device='cpu'):
        super(AdversarialSample, self).__init__()
        self.initial_label = initial_label
        self.adversarial_label = adversarial_label
        self.adv_image = adv_image
        self.orig_image = orig_image
        self.device = torch_device

    def get_gradients(self, py_model):
        return(find_grads(initial_labels, adversarial_image, py_model))
