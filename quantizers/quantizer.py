from .l2_quantiz import  best_quantization_l2
from .dct_quantiz import best_quantization_dct
import torch
import numpy as np

class Quantizer:
    def __init__(self, pytorch_model, arg_parser , number_classes=1000, binary_search_steps=10):
        self.quantizer = arg_parser.quantization_method
        self.max_dist = arg_parser.max_deg
        self.pytorch_device = arg_parser.device
        self.model = pytorch_model
        self.num_classes = 1000
        self.jpeg_quant =  arg_parser.jpeg_quality
        self.binary_search_steps = binary_search_steps

    def classif_loss(self, init_label, adv_label, py_image, need_grads=False):
        """
        The loss used to study adverariality. If <0, then the image is adversarial
        """
        if self.jpeg_quant!=0 and need_grads==False:
            for i in range(py_image.shape[0]):
                py_image[i,:,:,:] = jpeg.jpeg_to_spatial(py_image[i,:,:,:].double(), self.jpeg_quant,quantize=True).float()

        init_labels_onehot = torch.zeros(init_label.size(0), 1000, device=self.pytorch_device)
        init_labels_onehot.scatter_(1, init_label.unsqueeze(1).long(),1)

        adv_labels_onehot = torch.zeros(adv_label.size(0), 1000, device=self.pytorch_device)
        adv_labels_onehot.scatter_(1, adv_label.unsqueeze(1).long(),1)

        prediction = self.model(py_image)
        classification_loss = prediction*init_labels_onehot-prediction*adv_labels_onehot
        classification_loss = classification_loss.sum(axis=1)
        return(classification_loss)

    def find_grads(self, initial_labels, adversarial_images):
        self.model.zero_grad()
        adversarial_variables = torch.autograd.Variable(adversarial_images, requires_grad=True)
        prediction_adv = self.model(adversarial_variables)
        adversarial_labels  = prediction_adv.argsort(1)[:,-2]

        #Manually setting values to 0 will result in an error during the backward obviously hence the copy
        #This allows to find the second most predicted label, used to perform the quantization
        only_adv_pred = prediction_adv.detach().clone()
        for ind in range(adversarial_labels.shape[0]):
            if adversarial_labels[ind]==initial_labels[ind]:
                only_adv_pred[ind,adversarial_labels[ind]]=0
        adversarial_labels  = only_adv_pred.argmax(1)

        loss = self.classif_loss(initial_labels, adversarial_labels, adversarial_variables, need_grads=True)
        loss.sum().backward()
        gradients = adversarial_variables.grad.data
        if self.jpeg_quant!=0:
            for i in range(gradients.shape[0]):
                gradients[i,:,:,:] = jpeg.spatial_to_jpeg(gradients[i,:,:,:].double(), self.jpeg_quant,quantize=False).float()
        return(gradients, adversarial_labels)

    def quantize_samples(self, adv_element, orig_element, label_element):
        if self.quantizer == 'L2':
            quantized_adv = best_quantization_l2(self, adv_element, orig_element, label_element)
        elif self.quantizer == 'JPEG':
            quantized_adv = best_quantization_dct(self, adv_element, orig_element, label_element)
        return(quantized_adv)
