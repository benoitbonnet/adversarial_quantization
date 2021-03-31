import torch
import numpy as np
from scipy.fftpack import dct, idct

def quantization_tables(quality_factor):

    if quality_factor == 50:
        quant_table = np.array([[ 16,  11,  10,  16,  24,  40,  51,  61],\
                               [ 12,  12,  14,  19,  26,  58,  60,  55],\
                               [ 14,  13,  16,  24,  40,  57,  69,  56],\
                               [ 14,  17,  22,  29,  51,  87,  80,  62],\
                               [ 18,  22,  37,  56,  68, 109, 103,  77],\
                               [ 24,  35,  55,  64,  81, 104, 113,  92],\
                               [ 49,  64,  78,  87, 103, 121, 120, 101],\
                               [ 72,  92,  95,  98, 112, 100, 103,  99]]).astype(np.int32)

        quant_table2 = np.array([[17, 18, 24, 47, 99, 99, 99, 99],\
                               [18, 21, 26, 66, 99, 99, 99, 99],\
                               [24, 26, 56, 99, 99, 99, 99, 99],\
                               [47, 66, 99, 99, 99, 99, 99, 99],\
                               [99, 99, 99, 99, 99, 99, 99, 99],\
                               [99, 99, 99, 99, 99, 99, 99, 99],\
                               [99, 99, 99, 99, 99, 99, 99, 99],\
                               [99, 99, 99, 99, 99, 99, 99, 99]]).astype(np.int32)

    elif quality_factor == 92:
        quant_table = np.array([[ 6,  4,  4,  6, 10, 16, 20, 24],\
                               [ 5,  5,  6,  8, 10, 23, 24, 22],\
                               [ 6,  5,  6, 10, 16, 23, 28, 22],\
                               [ 6,  7,  9, 12, 20, 35, 32, 25],\
                               [ 7,  9, 15, 22, 27, 44, 41, 31],\
                               [ 10, 14, 22, 26, 32, 42, 45, 37],\
                               [ 20, 26, 31, 35, 41, 48, 48, 40],\
                               [ 29, 37, 38, 39, 45, 40, 41, 40]]).astype(np.int32)

        quant_table2 = np.array([[ 7,  7, 10, 19, 40, 40, 40, 40],\
                               [7,  8, 10, 26, 40, 40, 40, 40],\
                               [10, 10, 22, 40, 40, 40, 40, 40],\
                               [19, 26, 40, 40, 40, 40, 40, 40],\
                               [40, 40, 40, 40, 40, 40, 40, 40],\
                               [40, 40, 40, 40, 40, 40, 40, 40],\
                               [40, 40, 40, 40, 40, 40, 40, 40],\
                               [40, 40, 40, 40, 40, 40, 40, 40]]).astype(np.int32)

    elif quality_factor == 90:
        quant_table = np.array([[ 3,  2,  2,  3,  5,  8, 10, 12],\
                               [ 2,  2,  3,  4,  5, 12, 12, 11],\
                               [ 3,  3,  3,  5,  8, 11, 14, 11],\
                               [ 3,  3,  4,  6, 10, 17, 16, 12],\
                               [ 4,  4,  7, 11, 14, 22, 21, 15],\
                               [ 5,  7, 11, 13, 16, 21, 23, 18],\
                               [10, 13, 16, 17, 21, 24, 24, 20],\
                               [14, 18, 19, 20, 22, 20, 21, 20]]).astype(np.int32)

        quant_table2 = np.array([[ 3,  4,  5,  9, 20, 20, 20, 20],\
                               [ 4,  4,  5, 13, 20, 20, 20, 20],\
                               [ 5,  5, 11, 20, 20, 20, 20, 20],\
                               [ 9, 13, 20, 20, 20, 20, 20, 20],\
                               [20, 20, 20, 20, 20, 20, 20, 20],\
                               [20, 20, 20, 20, 20, 20, 20, 20],\
                               [20, 20, 20, 20, 20, 20, 20, 20],\
                               [20, 20, 20, 20, 20, 20, 20, 20]]).astype(np.int32)

    quant_table_tot = np.zeros((8,8,3))
    quant_table_tot[:,:,0] = quant_table
    quant_table_tot[:,:,1] = quant_table2
    quant_table_tot[:,:,2] = quant_table2
    return(quant_table_tot)

def dct2(a):
    return dct(dct(a, axis=0, norm='ortho' ), axis=1, norm='ortho')

def idct2(a):
    return idct(idct(a, axis=0 , norm='ortho'), axis=1 , norm='ortho')

def dct2_8_8(image, quant_tables, quantization=True):
    imsize = image.shape
    dct = np.zeros(imsize)
    for batch_nb in range(image.shape[0]):
        for channel in range(imsize[3]):
            for i in np.r_[:imsize[1]:8]:
                for j in np.r_[:imsize[2]:8]:
                    dct[batch_nb, i:(i+8),j:(j+8), channel] = dct2(image[batch_nb, i:(i+8),j:(j+8), channel])
                    if quantization:
                        dct[batch_nb, i:(i+8),j:(j+8), channel] =  dct[batch_nb, i:(i+8),j:(j+8), channel]/quant_tables[:,:,channel]
    return dct

def idct2_8_8(dct, quant_tables, quantization=True):
    im_dct = np.zeros(dct.shape)
    for batch_im in range(dct.shape[0]):
        for channel in range(dct.shape[3]):
            for i in np.r_[:dct.shape[1]:8]:
                for j in np.r_[:dct.shape[2]:8]:
                    if quantization:
                        im_dct[batch_im, i:(i+8),j:(j+8), channel] = dct[batch_im, i:(i+8),j:(j+8), channel]*quant_tables[:,:,channel]
                    else:
                        im_dct[batch_im, i:(i+8),j:(j+8), channel] = dct[batch_im, i:(i+8),j:(j+8), channel]
                    im_dct[batch_im, i:(i+8),j:(j+8), channel] = idct2(im_dct[batch_im, i:(i+8),j:(j+8), channel] )
    return im_dct


def rgb_to_ycbcr(bgr_tensor):
    b, g, r = bgr_tensor[:,:,:,0], bgr_tensor[:,:,:,1], bgr_tensor[:,:,:,2]
    ycbcr_tensor = torch.zeros(bgr_tensor.shape).double()
    ycbcr_tensor[:,:,:,0] = 0.299*r + 0.587*g + 0.114*b
    ycbcr_tensor[:,:,:,1] = 128-0.168736*r-0.331264*g+0.5*b
    ycbcr_tensor[:,:,:,2] = 128+0.5*r-0.418688*g-0.081312*b

    return(ycbcr_tensor)

def ycbcr_to_rgb(ycbcr_tensor):
    Y, Cb, Cr = ycbcr_tensor[:,:,:,0],ycbcr_tensor[:,:,:,1],ycbcr_tensor[:,:,:,2]
    bgr_tensor = torch.zeros(ycbcr_tensor.shape)
    bgr_tensor[:,:,:,2] = Y + 1.40200 * (Cr-128)
    bgr_tensor[:,:,:,1] = Y - 0.34414 * (Cb-128) - 0.71414 *(Cr-128)
    bgr_tensor[:,:,:,0] = Y + 1.77200 * (Cb-128)
    return(bgr_tensor)

def rgb_to_ycbcr_grad(bgr_tensor):
    b, g, r = bgr_tensor[:,:,:,0], bgr_tensor[:,:,:,1], bgr_tensor[:,:,:,2]
    ycbcr_tensor = torch.zeros(bgr_tensor.shape).double()
    ycbcr_tensor[:,:,:,0] = 0.299*r + 0.587*g + 0.114*b
    ycbcr_tensor[:,:,:,1] = -0.168736*r-0.331264*g+0.5*b
    ycbcr_tensor[:,:,:,2] = 0.5*r-0.418688*g-0.081312*b

    return(ycbcr_tensor)

def ycbcr_to_rgb_grad(ycbcr_tensor):
    Y, Cb, Cr = ycbcr_tensor[:,:,:,0],ycbcr_tensor[:,:,:,1],ycbcr_tensor[:,:,:,2]
    bgr_tensor = torch.zeros(ycbcr_tensor.shape)
    bgr_tensor[:,:,:,2] = Y + 1.40200 * Cr
    bgr_tensor[:,:,:,1] = Y - 0.34414 * Cb - 0.71414 * Cr
    bgr_tensor[:,:,:,0] = Y + 1.77200 * Cb
    return(bgr_tensor)

def spatial_to_jpeg_grad(spatial_tensor, quant_tables):

    jfif_tensor = rgb_to_ycbcr_grad(spatial_tensor)
    jpeg_array = dct2_8_8(jfif_tensor.numpy(), quant_tables, False)
    jpeg_tensor = torch.tensor(jpeg_array)
    #jpeg_tensor = jfif_tensor
    return(jpeg_tensor.float())

def jpeg_to_spatial_grad(jpeg_tensor, quant_tables):

    jpeg_array = jpeg_tensor.detach().cpu().numpy()
    jfif_tensor = idct2_8_8(jpeg_array, quant_tables, False)
    #jfif_tensor = jpeg_tensor
    spatial_tensor = ycbcr_to_rgb_grad(torch.tensor(jfif_tensor))
    return(spatial_tensor.float())

def jpeg_to_spatial(jpeg_tensor, quant_tables, quantization=True):
    jpeg_array = jpeg_tensor.detach().cpu().numpy()
    jfif_tensor = idct2_8_8(jpeg_array, quant_tables)
    jfif_tensor = jfif_tensor+128
    spatial_tensor = ycbcr_to_rgb(torch.tensor(jfif_tensor))
    if quantization:
        spatial_tensor = spatial_tensor.round()
    spatial_tensor = torch.clamp(spatial_tensor,0,255)
    return(spatial_tensor.float())

def spatial_to_jpeg(spatial_tensor, quant_tables, quantization=True):
    jfif_tensor = rgb_to_ycbcr(spatial_tensor)
    jfif_tensor = jfif_tensor.clamp(0,255)
    jfif_tensor = jfif_tensor-128
    jpeg_array = dct2_8_8(jfif_tensor.numpy(), quant_tables)
    if quantization:
        jpeg_array = jpeg_array.round()
    jpeg_tensor = torch.tensor(jpeg_array)
    return(jpeg_tensor.float())


def switch_to_dct(adversarial_images_png, orig_images_png, quantization_tables):
    #perturbations_png = adversarial_images_png-orig_images_png
    origins_dct = spatial_to_jpeg(orig_images_png, quantization_tables, quantization=False).to(orig_images_png.device)
    #perturbations_dct = spatial_to_jpeg(perturbations_png, quantization, quantization=False).to(adversarial_images_png.device)
    adversarial_dct = spatial_to_jpeg(adversarial_images_png, quantization_tables, quantization=False).to(orig_images_png.device)
    # origins_dct = adversarial_dct.round()
    perturbations_dct = adversarial_dct-origins_dct
    return(origins_dct, perturbations_dct, adversarial_dct)

def best_quantization_dct(quantizer, adversarial_images, images_orig):
    """
    This is the main function for the steganographic quantization with the Hill method.
    adversarial_images = Tensor of the already attacked images
    images_orig = Tensor of the original images before attack
    init_labels = Ground Truth labels for images
    """
    batch_size = adversarial_images.shape[0]

    quant_tables = quantization_tables(quantizer.jpeg_quant)

    images_orig_dct, perturbations_adv_dct, adversarial_images_dct = switch_to_dct(adversarial_images, images_orig,quant_tables)
    grads = quantizer.find_grads(jpeg_to_spatial(adversarial_images_dct, quant_tables, False))
    grads_dct = spatial_to_jpeg_grad(grads, quant_tables).to(quantizer.pytorch_device)
    grads_dct = grads_dct#/0.01#grads_dct.mean()
    grads_dct = grads_dct.to(quantizer.pytorch_device)

    lambdas = find_lambdas(quantizer, perturbations_adv_dct, grads_dct, quant_tables)

    #perturbations_quantized = lambda_binary_search_stega(quantizer, lambdas, perturbations_adv, grads, init_labels, adv_labels, images_orig, costs_maps)
    best_adversarials = lambda_binary_search_stega(quantizer, lambdas, perturbations_adv_dct, grads_dct, images_orig_dct, quant_tables, images_orig)
    #best_adversarials = perturbations_quantized + images_orig
    return(best_adversarials)

def find_lambdas(quantizer, perturbation, gradients, quantization):
    """
    This function calculates all the possibles values of Lambda for which the quantization of a pixel
    will swap from minimum distortion to maximum distortion
    """
    maximum_distortion = quantizer.max_dist
    batch_size = perturbation.shape[0]
    quantization = torch.tensor(quantization).repeat(48,64,1).to(quantizer.pytorch_device).float()

    flat_quant = quantization.view(-1).unsqueeze(0).repeat(batch_size,1)
    flat_gradients = gradients.view(batch_size,-1)
    flat_perturbation = perturbation.view(batch_size,-1)

    lambdas_max = 2*flat_quant*flat_perturbation/flat_gradients
    lambdas_max = lambdas_max.abs()
    lambda_list = lambdas_max.max(axis=1).values
    return(lambda_list)

def quantize(quantizer, lambada, unquantized_perturbations, gradient, quant_mat):
    """
    This function creates the quantized perturbation by quantizing with respect to the given value of Lambda
    (called "lambada" for Python's sake ...)
    """
    maximum_distortion = quantizer.max_dist
    initial_shape = unquantized_perturbations.shape
    batch_size = initial_shape[0]
    quantization = torch.tensor(quant_mat).repeat(48,64,1).unsqueeze(0).repeat(batch_size,1,1,1).to(quantizer.pytorch_device).float()


    solutions =  - lambada.view(batch_size,1,1,1)*gradient/(2*quantization)+unquantized_perturbations*2
    solutions = solutions.float().round()

    max_pert = unquantized_perturbations.floor() + quantizer.max_dist + 1
    min_pert = unquantized_perturbations.floor() - quantizer.max_dist
    sol_min = torch.min(max_pert, solutions)
    sol_max = torch.max(min_pert, sol_min)
    sol_max = torch.clamp(solutions, -quantizer.max_dist, quantizer.max_dist)

    saturated = (sol_max==-quantizer.max_dist).sum()+(sol_max==quantizer.max_dist).sum()
    # max_pert = unquantized_perturbations + quantizer.max_dist/quantization
    # min_pert = unquantized_perturbations - quantizer.max_dist/quantization
    #
    # sol_min = torch.min(max_pert, solutions)
    # sol_max = torch.max(min_pert, sol_min)
    #
    # sol_floor = unquantized_perturbations.floor()
    # sol_ceil = unquantized_perturbations.ceil()
    # jdiff = 3*quantization+lambada*gradient
    # sol_max = (jdiff>0).float()*sol_floor+(jdiff<0).float()*sol_ceil
    # sol_max = (jdiff<0).float()*sol_floor+(jdiff>0).float()*sol_ceil

    return(sol_max)

def lambda_binary_search_stega(quantizer, lambdas, perturbation_adv, grads, image_orig, quantization, orig_images_png):
    maximum_distortion = quantizer.max_dist
    image_orig = image_orig.float()
    batch_size = perturbation_adv.shape[0]
    lambda_values = torch.zeros(batch_size).to(quantizer.pytorch_device)

    perturbation_quantized = quantize(quantizer, lambda_values, perturbation_adv, grads, quantization)

    adversarial_quantized =  image_orig.clone() + perturbation_quantized
    adversarial_quantized = jpeg_to_spatial(adversarial_quantized, quantization).to(quantizer.pytorch_device)
    adversarial_quantized = torch.clamp(adversarial_quantized,0,255)
    #print("2", adversarial_quantized.norm())

    new_loss = quantizer.classif_loss(adversarial_quantized)
    best_adversarial = adversarial_quantized
    criterion = 28.0
    leslambdas = torch.ones(1).float().to(quantizer.pytorch_device)
    for lambda_search_step in range(70): #on a 224*224*3 image, solution is found in maximum 18 steps
            #When the loss is >0, then we need to increase distortion == increase lambda
            #When the loss is <0, then we are adversarial, we save the current quantization as the best and try to improve it
            lambda_values = leslambdas*10**(7-lambda_search_step/10)
            lambda_values = torch.tensor(lambda_values).repeat(batch_size,1,1,1)
            adversarial = (new_loss<=criterion).float().view(batch_size,1,1,1)
            # if lambda_search_step%5==0:
            #     print(new_loss, (adversarial_quantized-orig_images_png).norm())
            best_adversarial = adversarial_quantized*adversarial+best_adversarial*(1-adversarial)

            perturbation_quantized = quantize(quantizer, lambda_values, perturbation_adv, grads, quantization)
            #print("2", adversarial_quantized.norm())
            adversarial_quantized = perturbation_quantized + image_orig.clone()
            #print("in the loop", image_orig.norm(), perturbation_quantized.norm(), (perturbation_quantized-perturbation_adv).norm())

            #old_loss = quantizer.classif_loss(adversarial_quantized)
            adversarial_quantized = jpeg_to_spatial(adversarial_quantized, quantization).to(quantizer.pytorch_device)
            adversarial_quantized = torch.clamp(adversarial_quantized,0,255)

            new_loss = quantizer.classif_loss(adversarial_quantized)
            #print(new_loss, (adversarial_quantized-orig_images_png).norm() , lambda_values)
    #best_adversarial = jpeg_to_spatial(best_adversarial, quantization).to(quantizer.pytorch_device)
    #best_adversarial = torch.clamp(best_adversarial,0,255)
    return best_adversarial
