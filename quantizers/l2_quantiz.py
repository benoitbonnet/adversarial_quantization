import torch
import numpy as np

def best_quantization_l2(quantizer, adversarial_images, images_orig, init_labels):
    """
    This is the main function for the proposed best_quantization
    adversarial_images = Tensor of the already attacked images
    images_orig = Tensor of the original images before attack
    quantizer.model = The DNN used to craft the attack
    init_labels = Ground Truth labels for images
    currently batch is not supported so the algorithm is iterating over the batch size (WIP)
    """
    quantizer.model, torch_device = quantizer.model, quantizer.pytorch_device
    batch_size = adversarial_images.shape[0]
    perturbations_adv = adversarial_images - images_orig
    grads, adv_labels = quantizer.find_grads(init_labels, adversarial_images)

    if quantizer.jpeg_quant!=0:
        for i in range(batch_size):
            adversarial_images[i,:,:,:] = jpeg.spatial_to_jpeg(adversarial_images[i,:,:,:].double(), quantizer.jpeg_quant, quantize=True).float()
            perturbations_adv[i,:,:,:]= jpeg.spatial_to_jpeg(perturbations_adv[i,:,:,:].double(), quantizer.jpeg_quant, quantize=False).float()
            images_orig[i,:,:,:] =  jpeg.spatial_to_jpeg(images_orig[i,:,:,:].double(), quantizer.jpeg_quant, quantize=True).float()
    lambdas = find_lambdas(quantizer, perturbations_adv, grads)

    perturbations_quantized = lambda_binary_search(quantizer, lambdas, perturbations_adv, grads, init_labels, adv_labels, images_orig)
    best_adversarials = perturbations_quantized + images_orig

    return(best_adversarials)

def quantize(quantizer, lambada, unquantized_perturbations, gradients):
    """
    This function creates the quantized perturbation by quantizing with respect to the given value of Lambda
    (called "lambada" for Python's sake ...)
    """

    initial_shape = unquantized_perturbations.shape
    batch_size = initial_shape[0]

    ceiled_perturbations = torch.ceil(unquantized_perturbations)
    floored_perturbations = torch.floor(unquantized_perturbations)

    delta = ceiled_perturbations + lambada.view(batch_size,1,1,1) * gradients - 1/2
    quantized_perturbations = (delta>0).float()*floored_perturbations+(delta<0).float()*ceiled_perturbations

    return(quantized_perturbations)

def find_lambdas(quantizer, perturbation, gradients):
    """
    This function calculates all the possibles values of Lambda for which the quantization of a pixel
    will swap from minimum distortion to maximum distortion
    """
    gradients[gradients==0]+=1e-7
    batch_size = perturbation.shape[0]
    lambdas = (-torch.ceil(perturbation)+1/2)/gradients
    lambdas = lambdas.view(batch_size,-1)
    lambdas = lambdas.cpu().detach().numpy()
    lambda_list = []
    for ind in range(batch_size):
        batch_lambdas = lambdas[ind,:]
        batch_lambdas = batch_lambdas[~np.isnan(batch_lambdas)]
        batch_lambdas.sort()
        batch_lambdas = batch_lambdas[batch_lambdas>0]
        batch_lambdas = np.unique(batch_lambdas)
        lambda_list.append(batch_lambdas)
    return(lambda_list)


def lambda_binary_search(quantizer, lambdas, perturbations, grads, init_label, adv_label, image_orig):
    batch_size = len(lambdas)
    lowers = torch.zeros(batch_size).int()
    uppers = torch.tensor([lambda_array.shape[0] for lambda_array in lambdas]).int()
    indexes = ((lowers+uppers)//2)

    lambda_values = torch.tensor([lambdas[i][indexes[i]] for i in range(batch_size)], device = quantizer.pytorch_device)
    perturbation_quantized = quantize(quantizer, lambda_values, perturbations, grads)
    best_quantization = perturbation_quantized
    adversarial_quantized = perturbation_quantized + image_orig.clone()
    new_loss = quantizer.classif_loss(init_label, adv_label, adversarial_quantized)

    for lambda_search_step in range(18): #on a 224*224*3 image, solution is found in maximum 18 steps
            #When the loss is >0, then we need to increase distortion == increase lambda
            #When the loss is <0, then we are adversarial, we save the current quantization as the best and try to improve it
            lowers = (new_loss>0).int().cpu()*indexes + (new_loss<=0).int().cpu()*lowers
            uppers = (new_loss>0).int().cpu()*uppers + (new_loss<=0).int().cpu()*indexes
            indexes = ((lowers+uppers)//2)
            lambda_values = torch.tensor([lambdas[i][indexes[i]] for i in range(batch_size)], device = quantizer.pytorch_device)

            best_quantization = perturbation_quantized*(new_loss<=0).float().view(batch_size,1,1,1)+best_quantization*(new_loss>0).float().view(batch_size,1,1,1)

            perturbation_quantized = quantize(quantizer, lambda_values, perturbations, grads)
            adversarial_quantized = perturbation_quantized + image_orig.clone()
            new_loss = quantizer.classif_loss(init_label, adv_label, adversarial_quantized)

    return best_quantization
