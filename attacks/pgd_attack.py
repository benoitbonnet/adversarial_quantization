from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class PGD:
    """
    DDN attack: decoupling the direction and norm of the perturbation to achieve a small L2 norm in few steps.

    Parameters
    ----------
    steps : int
        Number of steps for the optimization.
    gamma : float, optional
        Factor by which the norm will be modified. new_norm = norm * (1 + or - gamma).
    init_norm : float, optional
        Initial value for the norm.
    quantize : bool, optional
        If True, the returned adversarials will have quantized values to the specified number of levels.
    levels : int, optional
        Number of levels to use for quantization (e.g. 256 for 8 bit images).
    max_norm : float or None, optional
        If specified, the norms of the perturbations will not be greater than this value which might lower success rate.
    device : torch.device, optional
        Device on which to perform the attack.
    callback : object, optional
        Visdom callback to display various metrics.

    """

class PGD:
        def __init__(self,
                     binary_search: bool = True,
                     binary_search_steps: int = 50,
                     num_classes: int = 1000,
                     iterations: int = 10,
                     upper_radius: float = 6000.,
                     costs_map = 0,
                     device: torch.device = torch.device('cpu')):

            self.device = device
            self.iterations = iterations
            self.b_search = binary_search
            self.b_search_steps = binary_search_steps
            self.upper_radius = upper_radius
            self.costs = costs_map
            print('init PGD')

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
            orig = inputs.to(self.device)
            lowers = torch.zeros(batch_size)
            uppers = torch.ones(batch_size)*self.upper_radius
            one_norm = torch.norm(torch.ones(inputs.shape))
            best_adversarial = inputs.clone().to(self.device)

            steps = 0
            print('b')
            while steps<self.b_search_steps:
                print('a')
                found = False
                radiuses = (uppers+lowers)/2
                epsilons = (radiuses/(2*self.iterations)).to(self.device)
                #print(epsilons)
                image_temp = orig.detach().clone().to(self.device)
                found = False
                for i in range(self.iterations):
                    steps+=1
                    if found:
                        break
                    image_v = torch.autograd.Variable(image_temp,requires_grad=True).to(self.device)
                    prediction = model(image_v)
                    pred_labels = prediction.argmax(1)
                    #print(pred_labels,labels)
                    if pred_labels!=labels:
                        uppers = radiuses
                        disto = (image_temp-orig).norm()
                        best_adversarial = image_v
                        found = True
                        break
                    leausse = prediction[:,labels]
                    leausse.sum().backward()
                    normalized_gradient = image_v.grad.data/torch.norm(image_v.grad.data)

                    image_temp += multiplier*epsilons*normalized_gradient
                    leausse = 0
                    adversary = torch.clamp(image_temp,0,255)

                    distorsion = adversary - orig
                    distorsion = distorsion.cpu()
                    #print(distorsion.norm())
                    if distorsion.norm()>radiuses:
                        normalized_distorsion = distorsion/torch.norm(distorsion)
                        adversary = orig.detach().cpu() + radiuses*normalized_distorsion
                    image_temp = torch.autograd.Variable(adversary, requires_grad=False).to(self.device)
                if found==False:
                    lowers = radiuses
            adversary = best_adversarial.detach()
            return adversary#, first_gradients
