
from .variance_quantiz import best_quantization_variance
from .gina_quantiz import best_quantifization_gina
from .l2_quantiz import  best_quantization_l2
from .hill_quantiz import best_quantifization_hill
import torch
import numpy as np

class Quantizer:
    def __init__(self, pytorch_model, number_classes, torch_device, max_distortion, quantization_method):
        self.quantizer = quantization_method
        self.max_dist = max_distortion
        self.pytorch_device = torch_device
        self.model = pytorch_model
        self.num_classes = number_classes

    def classif_loss(self, init_label, adv_label, py_image):
        """
        The loss used to study adverariality. If <0, then the image is adversarial
        """
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
        adversarial_labels  = prediction_adv.argmax(1)

        #Manually setting values to 0 will result in an error during the backward obviously hence the copy
        #This allows to find the second most predicted label, used to perform the quantization
        only_adv_pred = prediction_adv.detach().clone()
        for ind in range(adversarial_labels.shape[0]):
            if adversarial_labels[ind]==initial_labels[ind]:
                only_adv_pred[ind,adversarial_labels[ind]]=0
        adversarial_labels  = only_adv_pred.argmax(1)

        loss = self.classif_loss(initial_labels, adversarial_labels, adversarial_variables)
        loss.sum().backward()
        gradients = adversarial_variables.grad.data
        return(gradients, adversarial_labels)

    def quantize_samples(self, adv_element, orig_element, label_element):
        if self.quantizer == 'L2':
            quantized_adv = best_quantization_l2(self, adv_element, orig_element, label_element)
        elif self.quantizer == 'HILL':
            quantized_adv = best_quantifization_hill(self, adv_element, orig_element, label_element)
        elif self.quantizer == 'QUAD':
            quantized_adv = best_quantization_variance(self, adv_element, orig_element, label_element)
        elif self.quantizer == 'GINA':
            quantized_adv = best_quantifization_gina(self, adv_element, orig_element, label_element)
        return(quantized_adv)
