import torch
import numpy as np
from costs.Tools import variance_estimation_2D, wiener_filter, im2col, idct2, find_local_var

def best_quantization_variance(quantizer, adversarial_images, images_orig, init_labels):
    """
    This is the main function for the  steganographic quantization with the quadratic method.
    adversarial_images = Tensor of the already attacked images
    images_orig = Tensor of the original images before attack
    init_labels = Ground Truth labels for images
    """
    batch_size, height, width, channels = adversarial_images.shape
    perturbations_adv = adversarial_images - images_orig

    grads, adv_labels = quantizer.find_grads(init_labels, adversarial_images)
    loss = quantizer.classif_loss(init_labels, adv_labels, images_orig)
    local_variance = find_local_var(images_orig, adversarial_images, quantizer.pytorch_device)
    lambda_max = find_lambda_max(quantizer, grads, local_variance, perturbations_adv)

    quantization_mat = quantize_lambda(quantizer, perturbations_adv, grads, local_variance, lambda_max)
    adversarial = torch.clamp(quantization_mat+images_orig, 0, 255)
    best_adversarial = adversarial

    lower_lambda = torch.zeros(batch_size, device=quantizer.pytorch_device)
    upper_lambda = lambda_max
    lambda_search = (upper_lambda+lower_lambda)/2

    for binary_search_step in range(40):
        quantization_mat = quantize_lambda(quantizer, perturbations_adv, grads, local_variance, lambda_search)
        adversarial = torch.clamp(quantization_mat+images_orig, 0, 255)
        loss = quantizer.classif_loss(init_labels, adv_labels, adversarial)
        distortion = (quantization_mat+perturbations_adv).norm().item()

        best_adversarial = (loss<=0).float().view(batch_size,1,1,1)*adversarial + (loss>0).float().view(batch_size,1,1,1)*best_adversarial

        lower_lambda = (loss>0).float()*lambda_search + (loss<=0).float()*lower_lambda
        upper_lambda = (loss<=0).float()*lambda_search + (loss>0).float()*upper_lambda
        lambda_search = (lower_lambda+upper_lambda)/2
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
