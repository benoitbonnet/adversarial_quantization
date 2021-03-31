from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class PGD:
    """
    PGD attack: Projected Gradient Descent. Optimized to look for the best L2-radius to find an adversarial sample.

    binary_search_steps: number of radius updates
    iterations: number of steps within a radius to find an adversarial sample
    upper_radius: maximum L2-radius to find an adversarial sample

    """

class PGD:
        def __init__(self,
                     num_classes: int,
                     binary_search: bool = True,
                     binary_search_steps: int = 7,
                     iterations: int = 7,
                     upper_radius: float = 12000., #6000
                     device: torch.device = torch.device('cpu')):

            self.device = device
            self.iterations = iterations
            self.b_search = binary_search
            self.b_search_steps = binary_search_steps
            self.num_classes = num_classes
            self.upper_radius = upper_radius

        def __call__(self, model: nn.Module, inputs: torch.Tensor, criterion,
                   targeted: bool = False, epsilons=None) -> torch.Tensor:
            """
            Performs the attack of the model for the inputs and labels.

            Parameters
            ----------
            model : nn.Module
                Model to attack.
            inputs : torch.Tensor
                Batch of samples to attack. Values should be in the [0, 1] range.
            labels : torch.Tensor
                Labels of the samples to attack if untargeted, else labels of targets.
            targeted : bool, optional
                Whether to perform a targeted attack or not.

            Returns
            -------
            torch.Tensor
                Batch of samples modified to be adversarial to the model.

            """
            labels = criterion.labels.raw
            batch_size = inputs.shape[0]
            multiplier = 1 if targeted else -1
            orig = inputs.to(self.device)
            lowers = torch.zeros(batch_size,1,1,1).to(self.device)
            uppers = torch.ones(batch_size,1,1,1).to(self.device)*self.upper_radius

            prediction = model(inputs)
            pred_labels = torch.argmax(prediction,axis=-1)
            misclassified = (pred_labels!=labels).float().view(batch_size,1,1,1)

            best_l2 = uppers*(1-misclassified)+misclassified*lowers
            best_adversarial_samples = orig

            for binary_step in range(self.b_search_steps):
                radiuses = (uppers+lowers)/2
                epsilons = (radiuses/(self.iterations))
                adv_samples_temp = orig.detach().clone().to(self.device)
                adversarial_found = torch.zeros(batch_size,1,1,1, device=self.device)
                for i in range(self.iterations):
                    inputs_variable = torch.autograd.Variable(adv_samples_temp,requires_grad=True).to(self.device)
                    prediction = model(inputs_variable)
                    pred_labels = prediction.argmax(1)
                    adversarial = (labels!=pred_labels).view(batch_size,1,1,1).float().to(self.device)
                    adversarial_found = adversarial_found + adversarial

                    labels_onehot = torch.zeros(labels.size(0), self.num_classes, device=self.device)
                    labels_onehot.scatter_(1, labels.unsqueeze(1).long(),1)

                    adversarial_loss = prediction*labels_onehot
                    adversarial_loss = adversarial_loss.sum(axis=1)
                    adversarial_loss.backward(torch.ones(batch_size, device=self.device))
                    gradients = inputs_variable.grad.data

                    normalized_gradients = gradients/(gradients.view(batch_size,-1).norm(dim=1).view(batch_size,1,1,1))
                    adv_samples_temp += multiplier*epsilons.view(batch_size,1,1,1)*normalized_gradients
                    adversarial_samples = torch.clamp(adv_samples_temp,0,255)
                    perturbations = (adversarial_samples - orig)
                    distortions = perturbations.view(batch_size,-1).norm(dim=1).view(batch_size,1,1,1)
                    adv_samples_temp = adversarial_samples.detach()

                    normalized_perturbations = perturbations/distortions.view(batch_size,1,1,1)
                    within_radiuses = (distortions<=radiuses).float()
                    adversarial_samples = (orig+radiuses*normalized_perturbations)*(1-within_radiuses)+within_radiuses*adv_samples_temp

                    #Check if current sample is both adversarial and has a lower distortion
                    adv_lower_dist = adversarial*(distortions<best_l2).float()
                    best_l2 = adv_lower_dist*distortions+best_l2*(1-adv_lower_dist)
                    best_adversarial_samples = adv_lower_dist*adversarial_samples+best_adversarial_samples*(1-adv_lower_dist)

                found_in_iteration = (adversarial_found!=0).float()
                lowers = (1-found_in_iteration)*radiuses+found_in_iteration*lowers.to(self.device)
                uppers = found_in_iteration*radiuses+(1-found_in_iteration)*uppers.to(self.device)
                radiuses = torch.ones(batch_size,1,1,1).to(self.device)*((lowers+uppers)/2)

            adversarial_samples = best_adversarial_samples.detach()
            return adversarial_samples, None, None
