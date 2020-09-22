import torch
import numpy as np
from costs.HILL import do_process_color

def best_quantifization_hill(quantizer, adversarial_images, images_orig, init_labels):
    """
    This is the main function for the steganographic quantization with the Hill method.
    adversarial_images = Tensor of the already attacked images
    images_orig = Tensor of the original images before attack
    init_labels = Ground Truth labels for images
    """
    batch_size = adversarial_images.shape[0]

    perturbations_adv = adversarial_images - images_orig
    costs_maps = do_process_color(images_orig.cpu())
    costs_maps = torch.Tensor(costs_maps).float().to(quantizer.pytorch_device)

    grads, adv_labels = quantizer.find_grads(init_labels, adversarial_images)
    lambdas = find_lambdas(quantizer, perturbations_adv, grads, costs_maps)

    perturbations_quantized = lambda_binary_search_stega(quantizer, lambdas, perturbations_adv, grads, init_labels, adv_labels, images_orig, costs_maps)
    best_adversarials = perturbations_quantized + images_orig

    return(best_adversarials)

def find_lambdas(quantizer, perturbation, gradients, costs):
    """
    This function calculates all the possibles values of Lambda for which the quantization of a pixel
    will swap from minimum distortion to maximum distortion
    """
    maximum_distortion = quantizer.max_dist
    batch_size = perturbation.shape[0]

    flat_costs = costs.view(batch_size,-1)
    flat_gradients = gradients.view(batch_size,-1)
    flat_perturbation = perturbation.view(batch_size,-1)

    lambdas = torch.zeros(batch_size, 2*maximum_distortion, flat_gradients.shape[1]).to(quantizer.pytorch_device)
    flat_floored = torch.floor(flat_perturbation)
    for i in range(2*maximum_distortion):
        q_mat = flat_floored+i-(maximum_distortion-1)
        lambdas[:,i,:] = -flat_costs*(q_mat**2)/(flat_gradients*(q_mat-flat_perturbation))

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

def quantize(quantizer, lambada, unquantized_perturbations, gradient, costs_mat):
    """
    This function creates the quantized perturbation by quantizing with respect to the given value of Lambda
    (called "lambada" for Python's sake ...)
    """
    maximum_distortion = quantizer.max_dist
    initial_shape = unquantized_perturbations.shape
    batch_size = initial_shape[0]

    flat_perturbations = unquantized_perturbations.view(batch_size,-1)
    flat_floored = torch.floor(flat_perturbations)
    flat_costs = costs_mat.view(batch_size,-1)
    flat_gradients = gradient.view(batch_size,-1)

    solutions = torch.zeros(batch_size,2*maximum_distortion, flat_floored.shape[1])
    possible_quants = torch.zeros(batch_size,2*maximum_distortion, flat_floored.shape[1])

    for i in range(int(2*maximum_distortion)):
        q_mat = flat_floored+i-(maximum_distortion-1)
        solutions[:,i,:] = flat_costs*(q_mat**2) + lambada.view(batch_size,1)*flat_gradients*(q_mat-flat_perturbations)
        possible_quants[:,i,:] = q_mat

    min_quantized_perturbations = solutions.argmin(axis=1)

    one_hot = torch.nn.functional.one_hot(min_quantized_perturbations,2*maximum_distortion)

    test = possible_quants*one_hot.transpose(1,2).float()
    test = test.sum(axis=1)

    test = test.view(initial_shape).to(quantizer.pytorch_device)

    return(test)

def lambda_binary_search_stega(quantizer, lambdas, perturbation_adv, grads, init_label, adv_label, image_orig, costs):
    maximum_distortion = quantizer.max_dist

    batch_size = len(lambdas)
    lowers = torch.zeros(batch_size).int()
    uppers = torch.tensor([lambda_array.shape[0] for lambda_array in lambdas]).int()
    indexes = ((lowers+uppers)//2)

    lambda_values = torch.tensor([lambdas[i][indexes[i]] for i in range(batch_size)], device = quantizer.pytorch_device)
    perturbation_quantized = quantize(quantizer, lambda_values, perturbation_adv, grads, costs)
    best_quantization = perturbation_quantized
    adversarial_quantized = perturbation_quantized + image_orig.clone()
    adversarial_quantized = torch.clamp(adversarial_quantized,0,255)
    new_loss = quantizer.classif_loss(init_label, adv_label, adversarial_quantized)
    best_adversarial = adversarial_quantized


    for lambda_search_step in range(18): #on a 224*224*3 image, solution is found in maximum 18 steps
            #When the loss is >0, then we need to increase distortion == increase lambda
            #When the loss is <0, then we are adversarial, we save the current quantization as the best and try to improve it
            lowers = (new_loss>0).int().cpu()*indexes + (new_loss<=0).int().cpu()*lowers
            uppers = (new_loss>0).int().cpu()*uppers + (new_loss<=0).int().cpu()*indexes
            indexes = ((lowers+uppers)//2)
            lambda_values = torch.tensor([lambdas[i][indexes[i]] for i in range(batch_size)], device = quantizer.pytorch_device)

            best_quantization = perturbation_quantized*(new_loss<=0).float().view(batch_size,1,1,1)+best_quantization*(new_loss>0).float().view(batch_size,1,1,1)

            perturbation_quantized = quantize(quantizer, lambda_values, perturbation_adv, grads, costs)
            adversarial_quantized = perturbation_quantized + image_orig.clone()
            adversarial_quantized = torch.clamp(adversarial_quantized,0,255)
            new_loss = quantizer.classif_loss(init_label, adv_label, adversarial_quantized)

    return best_quantization
