import torch
import numpy as np
from costs.Tools import variance_estimation_2D, wiener_filter, im2col, idct2, find_local_var
from scipy.signal import convolve2d


def best_quantifization_gina(quantizer, adversarial_images, images_orig, init_labels):
    """
    This is the main function for the steganographic quantization with the Gina method.
    adversarial_images = Tensor of the already attacked images
    images_orig = Tensor of the original images before attack
    init_labels = Ground Truth labels for images
    """
    initial_shape = adversarial_images.shape
    batch_size, height, width, _  = initial_shape
    perturbations_adv = adversarial_images - images_orig
    grads, adv_labels = quantizer.find_grads(init_labels, adversarial_images)
    #Each lattice has to contribute to a 1/12 of the classification loss.
    loss = quantizer.classif_loss(init_labels, adv_labels, images_orig)

    local_variance = find_local_var(images_orig, adversarial_images, quantizer.pytorch_device)
    lambda_max = find_lambda_max(quantizer, grads, local_variance, perturbations_adv)

    lattice_masks = torch.zeros(height,width,4).to(quantizer.pytorch_device)
    for i in range(height//2):
        for j in range(width//2):
            lattice_masks[2*i,2*j,0]=1
            lattice_masks[2*i+1,2*j,1]=1
            lattice_masks[2*i,2*j+1,2]=1
            lattice_masks[2*i+1,2*j+1,3]=1

    quantized_perturbation = torch.zeros(initial_shape).to(quantizer.pytorch_device)
    best_adversarial = adversarial_images.clone()

    """
    This is the loop of quantization on every lattice. Every lattice has to contribute to a roughly equal part
    of decreasing the classification loss until the last one ensures the loss is negative (==adversarial)
    To quantize a lattice, only the perturbation and gradient of this lattice is kept and the rest is set to 0
    """
    iterations = 0
    for channel_nb in range(3):
        for lattice_nb in range(4):
            perturbation_lattice = torch.zeros(initial_shape).to(quantizer.pytorch_device)
            perturbation_lattice[:,:,:,channel_nb]=perturbations_adv[:,:,:,channel_nb]*lattice_masks[:,:,lattice_nb]
            grads_lattice = torch.zeros(initial_shape).to(quantizer.pytorch_device)
            grads_lattice[:,:,:,channel_nb]=grads[:,:,:,channel_nb]*lattice_masks[:,:,lattice_nb]
            #
            local_variance = update_variance(perturbation_lattice, quantized_perturbation, channel_nb, local_variance)
            local_variance = local_variance.to(quantizer.pytorch_device)

            loss_contribution = (11-iterations)*loss/(12-iterations)#-1e4
            loss_init = loss.clone()
            quantized_lattice = binary_search_lambda(quantizer, init_labels, adv_labels, quantized_perturbation, images_orig, grads_lattice, local_variance, perturbation_lattice, lambda_max, loss_contribution)
            quantized_perturbation = quantized_perturbation+quantized_lattice
            loss = quantizer.classif_loss(init_labels, adv_labels, images_orig.clone()+quantized_perturbation.clone())
            iterations+=1

    best_adversarial = images_orig+quantized_perturbation
    return(best_adversarial)


def quantize_lambda(quantizer, perturbations, gradients, variance_map, lambda_val):
    """
    This function returns the quantized perturbation for a given value of lambda.
    """
    maximum_distortion = quantizer.max_dist
    batch_size = perturbations.shape[0]

    #Applying the quantization method and keeping it inside the maximum distortion boundaries
    quantized = (-variance_map*lambda_val.view(batch_size,1,1,1)*gradients/2-perturbations)
    quantized = (quantized+perturbations).round()
    quantized = quantized.to(quantizer.pytorch_device)
    quantized_max = torch.max((perturbations.floor()-(maximum_distortion-1)), quantized)
    quantized_min = torch.min((perturbations.ceil()+(maximum_distortion-1)), quantized_max)

    #This ensures that the quantization doesn't go backward and cancels the perturbation
    quantized_min[perturbations>0] = torch.max((quantized_min[perturbations>0]), torch.zeros(perturbations[perturbations>0].shape).to(quantizer.pytorch_device))
    quantized_min[perturbations<0] = torch.min((quantized_min[perturbations<0]), torch.zeros(perturbations[perturbations<0].shape).to(quantizer.pytorch_device))
    quantized = quantized_min
    return(quantized)

def find_lambda_max(quantizer, gradients, local_var, adv_perturbation):
    """
    This function finds for each image the maximum value of lambda.
    This corresponds to the value for which the quantization will have the biggest distortion.
    """
    qminus_matrix= torch.floor(adv_perturbation).to(quantizer.pytorch_device)
    maximum_distortion = quantizer.max_dist
    batch_size = gradients.shape[0]
    lambda_max = torch.zeros(batch_size, device=quantizer.pytorch_device)
    for k in range(2*maximum_distortion+1):
        polynomial_minimum = - 2/(gradients*local_var)*(adv_perturbation+(qminus_matrix-maximum_distortion+k))
        polynomial_minimum = polynomial_minimum.view(batch_size,-1)
        lambda_max_iter = polynomial_minimum.max(axis=1).values
        lambda_max = (lambda_max_iter>lambda_max).float()*lambda_max_iter + (lambda_max_iter<=lambda_max).float()*lambda_max
    return(lambda_max)

def update_variance(p_lattice, quantized_perturbation, channel_nb ,variance):
    """
    This function updates the local variance map.
    The local variance is increased (meaning it will be more likely to be quantized with maximum distortion)
    IF : The perturbation and the quantized neighbors share the same sign.
    Eventually this allows for an increased local smoothness == lower difficulty
    """
    batch_size = quantized_perturbation.shape[0]
    convolve_filter = torch.ones(3,3)
    convolve_filter[1,1] = 0
    cpu_perturbation = quantized_perturbation.cpu()
    cpu_variance = variance.cpu()
    cpu_lattice = p_lattice.cpu()
    for batch_image in range(batch_size):
        averaging = convolve2d(cpu_perturbation[batch_image,:,:,channel_nb], convolve_filter,'same')

        averaging = torch.tensor(averaging)

        p_and_v_same_sign = (averaging.sign()==cpu_lattice[batch_image,:,:,channel_nb].sign()).float()
        p_and_v_same_sign[averaging==0]=0
        p_and_v_same_sign[cpu_lattice[batch_image,:,:,channel_nb]==0]=0
        cpu_variance[batch_image,:,:,channel_nb] = cpu_variance[batch_image,:,:,channel_nb]+8*p_and_v_same_sign*cpu_variance[batch_image,:,:,channel_nb]

    return(cpu_variance)

def binary_search_lambda(quantizer, init_labels, adv_labels, quantized_perturbation, images_orig, gradients, local_var, adv_perturbation, lambda_max, loss_contribution):
    """
    This function performs the binary search on a given lattice. The stopping criterion is that the loss is just inferior
    to the loss contribution == the amount of loss that lattice has to decrease
    """
    batch_size = gradients.shape[0]
    lower_lambda = torch.zeros(batch_size, device= quantizer.pytorch_device)
    upper_lambda = lambda_max

    best_quantiz = quantize_lambda(quantizer, adv_perturbation, gradients, local_var, lambda_max)
    lambda_search = (lower_lambda+upper_lambda)/2
    previous = torch.zeros(batch_size, device= quantizer.pytorch_device)

    for binary_search_step in range(30):
        lattice_quantized = quantize_lambda(quantizer, adv_perturbation, gradients, local_var, lambda_search)
        adversarial_samples = torch.clamp(quantized_perturbation+lattice_quantized+images_orig.clone(), 0, 255)
        clamped_lattice = adversarial_samples-images_orig-quantized_perturbation

        loss = quantizer.classif_loss(init_labels, adv_labels, adversarial_samples)
        contributed = (loss<=loss_contribution).float().view(batch_size,1,1,1)
        best_quantiz = contributed*clamped_lattice + (1-contributed)*best_quantiz

        lower_lambda = (1-contributed).view(batch_size)*lambda_search + contributed.view(batch_size)*lower_lambda
        upper_lambda = contributed.view(batch_size)*lambda_search + (1-contributed).view(batch_size)*upper_lambda
        lambda_search = (lower_lambda+upper_lambda)/2
        previous = loss

    return(best_quantiz)
