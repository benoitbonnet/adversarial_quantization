import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


def teddy_decay(current_step, total_steps, gamma_min):
    rate = current_step/(total_steps+1.0)
    rage = 1-gamma_min
    epsi = gamma_min + rate*rage
    return epsi

def classif_loss(model, inputs_variable ,labels, num_classes, attack_device):
    batch_size = labels.shape[0]
    prediction = model(inputs_variable)
    labels_onehot = torch.zeros(batch_size, num_classes, device=attack_device)
    labels_onehot.scatter_(1, labels.unsqueeze(1).long(),1)

    adversarial_loss = prediction*labels_onehot
    adversarial_loss = adversarial_loss.sum(axis=1)
    adversarial_loss.backward(torch.ones(batch_size, device=attack_device))
    gradients = inputs_variable.grad.data
    return(prediction.argmax(1),gradients)

def find_betas(current_adv, current_z):
    batch_size = current_adv.shape[0]
    current_z_norm = current_z.view(batch_size,-1).norm(dim=1)
    upper_limit = 2
    lower_limit = 0
    for i in range(10):
        new_beta = (upper_limit+lower_limit)/2
        new_adv = current_adv + new_beta*(current_z-current_adv)
        if new_adv.view(batch_size,-1).norm(dim=1)>current_z_norm:
            upper_limit = new_beta
        else:
            lower_limit = new_beta
        #print( new_adv.view(batch_size,-1).norm(dim=1),current_z_norm)
    return(new_adv)

class BP:

    def __init__(self,
                steps=20,
                gamma = 0.3,
                upper_radius=1000,
                alpha = 500,
                num_classes=1000,
                device = torch.device('cpu')):
        self.steps = steps
        self.gamma = gamma
        self.alpha = alpha
        self.device = device
        self.upper_radius = upper_radius
        self.num_classes = num_classes

    def attack(self,model,inputs,labels,targeted=False):
        batch_size = inputs.shape[0]
        multiplier = 1 if targeted else -1
        best_adv = inputs.clone()
        best_norm = torch.zeros(batch_size).to(best_adv.device)
        adv = inputs.clone()
        adv = torch.autograd.Variable(adv, requires_grad=True)
        pred_labels, grad = classif_loss(model, adv, labels, self.num_classes, self.device)
        grad_norm = grad.view(batch_size,-1).norm(dim=1).view(batch_size,1,1,1)
        normalized_grad = - grad/grad_norm
        is_adv = pred_labels!=labels
        already_adversarial = is_adv
        i=0
        #Stage 1: finding an adversarial sample quickly
        ever_found = already_adversarial #help control stage 1 in case an image is already adversarial
        while is_adv.sum()!=batch_size and i<self.steps:
            #print(adv.view(batch_size,-1).norm(dim=1), inputs.view(batch_size,-1).norm(dim=1))
            gammas = teddy_decay(i,self.steps,self.gamma)
            normalized_grad = grad/grad_norm
            adv_temp = inputs + self.alpha*multiplier*gammas*normalized_grad
            adv_temp = torch.clamp(adv_temp,0,255)
            adv_temp = torch.autograd.Variable(adv_temp, requires_grad=True)
            pred_labels, grad = classif_loss(model, adv_temp, labels, self.num_classes, self.device)
            grad_norm = grad.view(batch_size,-1).norm(dim=1).view(batch_size,1,1,1)
            is_adv = pred_labels!=labels
            adv = adv_temp*(ever_found==0).view(batch_size,1,1,1) + adv*(ever_found!=0).view(batch_size,1,1,1)
            ever_found = ever_found+is_adv
            i+=1

            #print(adv.view(batch_size,-1).norm(dim=1), inputs.view(batch_size,-1).norm(dim=1))
        best_norm = best_norm*(~is_adv)+(adv-inputs).view(batch_size,-1).norm(dim=1)*(is_adv)
        i=0
        #Stage 2
        while i<self.steps: #our sample is still adversarial -> reduce distortion
            gammas = teddy_decay(i,self.steps,self.gamma)
            delta = adv-inputs
            #print(delta.view(batch_size,-1).norm(dim=1), delta.shape, delta.view(batch_size,-1).shape, delta[1,:,:,:].norm())
            delta_norm = delta.view(batch_size,-1).norm(dim=1).view(batch_size,1,1,1)
            normalized_delta = delta/delta_norm

            adv = torch.autograd.Variable(adv, requires_grad=True)
            pred_labels, grad = classif_loss(model, adv, labels, self.num_classes, self.device)
            grad_norm = grad.view(batch_size,-1).norm(dim=1).view(batch_size,1,1,1)
            normalized_grad = - grad/grad_norm

            r_scale = (delta*normalized_grad).view(batch_size,-1).sum(1).view(batch_size,1,1,1)

            is_adv = pred_labels!=labels
            better_norm = (adv-inputs).view(batch_size,-1).norm(dim=1)<best_norm
            better_norm_and_adv = better_norm*is_adv
            best_adv = adv*(better_norm_and_adv.view(batch_size,1,1,1)) + best_adv*(~better_norm_and_adv.view(batch_size,1,1,1))
            best_norm = (adv-inputs).view(batch_size,-1).norm(dim=1)*better_norm_and_adv + best_norm*(~better_norm_and_adv)

            # samples that are still adversarial  -> decrease distortion
            epsilons = gammas*delta_norm
            v_star = inputs - multiplier*r_scale.view(batch_size,1,1,1)*normalized_grad
            v_adv_diff = (adv-v_star)
            diff_norm = (v_adv_diff.view(batch_size,-1).norm(dim=1).view(batch_size,1,1,1))
            #change diff_norm avoid nan if original image is already adversarial
            diff_norm[diff_norm==0]=1
            diff_normed = v_adv_diff/diff_norm
            dir = torch.max(torch.zeros(batch_size).to(r_scale.device), epsilons[:,0,0,0]**2-r_scale[:,0,0,0]**2)
            z_vector_out = v_star + diff_normed*(dir.view(batch_size,1,1,1)**0.5)


            # samples that are not adversarial anymore -> increase distortion
            epsilons_in = delta_norm/gammas
            z_vector_in = adv - multiplier*(r_scale+(epsilons_in**2-delta_norm**2+r_scale**2)**0.5)*normalized_grad

            adv = z_vector_out*(is_adv.view(batch_size,1,1,1))+z_vector_in*(~is_adv.view(batch_size,1,1,1))
            adv = torch.clamp(adv,0,255)
            i+=1
        best_adv = best_adv*(~already_adversarial.view(batch_size,1,1,1))+inputs*(already_adversarial.view(batch_size,1,1,1))
        return best_adv.detach()
