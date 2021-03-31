import torch
import numpy as np
import cv2
import sys
import os
from attacks.attack_gen import attack_generator
from quantizers.quantizer import Quantizer
from data_utils.data_utils import  AdversarialDataset,AttackParser, load_model
import argparse
import foolbox as fb

parser = AttackParser(argparse.ArgumentParser())

if not os.path.isdir(parser.save_path+'/measures'):
    os.makedirs(parser.save_path+'/measures')
if not os.path.isdir(parser.save_path+'/images'):
    os.makedirs(parser.save_path+'/images')
measures_path = parser.save_path+'/measures/'
images_path = parser.save_path+'/images/'

orig_set = AdversarialDataset(parser)
orig_loader = torch.utils.data.DataLoader(orig_set, batch_size=parser.batch_size, shuffle=False)

results_cpt=0
results_array = np.zeros(len(orig_set))
results_array_unquant = np.zeros(len(orig_set))
print(len(orig_loader), " batches")

cpt3=0

for model_cpt,model_name in enumerate(parser.models):
    model, device = load_model(parser, model_cpt)
    fmodel = fb.PyTorchModel(model, bounds=[0,255])
    for attack_cpt, attack_name in enumerate(parser.attacks):

        results_cpt=0
        results_array = np.zeros(len(orig_set))
        results_array_unquant = np.zeros(len(orig_set))

        if parser.attack_type=='foolbox':
            attack = getattr(fb.attacks, attack_name, fb.attacks.FGSM)()
        else:
            attack = attack_generator(parser,attack_cpt)

        quantizer = Quantizer(model, parser, number_classes=1000)


        for i, data_batch in enumerate(orig_loader, 0):
            orig_batch, initial_label, image_nbs = data_batch
            criterion = fb.criteria.Misclassification(initial_label)
            batch_size = orig_batch.shape[0]
            if i%5==0:
                print(i*batch_size)

            adversarial_image,_,_ = attack(fmodel, orig_batch, criterion, epsilons = parser.epsilon)
            quantized_adv = quantizer.quantize_samples(adversarial_image, orig_batch, initial_label)

            pred_adv = model(adversarial_image)
            adv_label = torch.argmax(pred_adv,axis=-1)

            pred_quant = model(quantized_adv)
            quant_label = torch.argmax(pred_quant,axis=-1)

            disto_quantized = (quantized_adv-orig_batch).view(batch_size,-1).norm(dim=1).cpu().numpy()/387.9794
            disto_unquantized = (adversarial_image-orig_batch).view(batch_size,-1).norm(dim=1).cpu().numpy()/387.9794

            is_adv_quant = (quant_label!=initial_label).cpu()
            is_adv_unquant = (adv_label!=initial_label).cpu()
            results_array[results_cpt:results_cpt+batch_size] = is_adv_quant*disto_quantized + (~is_adv_quant)*1e6
            results_array_unquant[results_cpt:results_cpt+batch_size] = is_adv_unquant*disto_unquantized + (~is_adv_unquant)*1e6
            #print(results_array[results_cpt:results_cpt+batch_size], results_array_unquant[results_cpt:results_cpt+batch_size])
            for batch_image in range(quantized_adv.shape[0]):
                if initial_label[batch_image].item()!=quant_label[batch_image].item():
                     if parser.jpeg_quality==0:
                         img_bgr = cv2.cvtColor(np.float32(quantized_adv[batch_image].cpu()), cv2.COLOR_RGB2BGR)
                         cv2.imwrite(images_path+"/{}.png".format(image_nbs[0][batch_image]),img_bgr)
                     else:
                         np.save(images_path+"/{}.npy".format(image_nbs[0][batch_image]),np.float32(quantized_adv[batch_image].cpu()) )

            results_cpt+=batch_size
        np.save('{}{}_{}_quantized.npy'.format(measures_path, model_name, attack_name), results_array)
        np.save('{}{}_{}_unquantized.npy'.format(measures_path, model_name, attack_name), results_array_unquant)
#np.save('/nfs/nas4/bbonnet/bbonnet/distortion_curves/deepfool/{}2.npy'.format(model_name), results_array_unquant)
