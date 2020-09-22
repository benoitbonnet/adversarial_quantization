from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class FGSM:
    """
    FGSM attack: basic Fast Gradient Sign Method to perform a fast attack based on gradient sign.

    epsilon : if binary search is not performed: factor by which the gradient sign will be added to the inputs
    binary_search : if True, attack will be optimized on binary_search_steps steps between O and max_epsilon

    """

    def __init__(self,
                 num_classes: int,
                 epsilon: float = 0.05,
                 binary_search: bool = False,
                 binary_search_steps: int = 20,
                 max_epsilon: float = 15.,
                 device: torch.device = torch.device('cpu')):


        self.epsilon = epsilon
        self.device = device
        self.b_search = binary_search
        self.b_search_steps = binary_search_steps
        self.num_classes = num_classes
        self.max_epsilon = max_epsilon

    def attack(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor,
               targeted: bool = False) -> torch.Tensor:
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
        batch_size = inputs.shape[0]
        multiplier = 1 if targeted else -1

        if self.b_search:
            lowers = torch.zeros(batch_size)
            uppers = torch.ones(batch_size)*self.max_epsilon
            epsilons = torch.ones(batch_size)*((lowers+uppers)/2)
        else:
            epsilons = torch.ones(batch_size)*self.epsilon

        epsilons = epsilons.to(self.device)

        inputs_variable = torch.autograd.Variable(inputs,requires_grad=True)
        prediction = model(inputs_variable)

        labels_onehot = torch.zeros(labels.size(0), self.num_classes, device=self.device)
        labels_onehot.scatter_(1, labels.unsqueeze(1).long(),1)

        adversarial_loss = prediction*labels_onehot
        adversarial_loss = adversarial_loss.sum(axis=1)
        adversarial_loss.backward(torch.ones(batch_size, device=self.device))
        gradients = inputs_variable.grad.data

        adversary = inputs + multiplier*epsilons.view(batch_size,1,1,1)*torch.sign(gradients)
        adversary = torch.clamp(adversary,0,255)
        adversary = torch.autograd.Variable(adversary, requires_grad=True).to(self.device)

        best_adversarial = adversary.to(self.device)

        if self.b_search:
            for i in range(self.b_search_steps):
                new_pred = model(adversary)

                #determining if current iteration is adversarial
                pred_labels = new_pred.argmax(1)
                adversarial = (labels!=pred_labels).float().to(self.device)
                best_adversarial = adversarial.view(batch_size,1,1,1)*adversary + (1-adversarial).view(batch_size,1,1,1)*best_adversarial

                #updating epsilon value
                lowers = (1-adversarial)*epsilons+adversarial*lowers.to(self.device)
                uppers = adversarial*epsilons+(1-adversarial)*uppers.to(self.device)
                epsilons = torch.ones(batch_size).to(self.device)*((lowers+uppers)/2)

                adversary = inputs +  multiplier*epsilons.view(batch_size,1,1,1)*torch.sign(gradients)
                adversary = torch.clamp(adversary,0,255)
                adversary = torch.autograd.Variable(adversary, requires_grad=True).to(self.device)

        new_pred = model(adversary)
        pred_labels = new_pred.argmax(1)
        adversarial = (labels!=pred_labels).float().to(self.device)
        best_adversarial = adversarial.view(batch_size,1,1,1)*adversary + (1-adversarial).view(batch_size,1,1,1)*best_adversarial

        return best_adversarial.detach()
