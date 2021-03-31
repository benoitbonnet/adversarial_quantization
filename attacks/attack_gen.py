from .fgsm_attack import FGSM
from .pgd_attack import PGD
from .bp_attack import BP
import sys

def attack_generator(arg_parser, attack_cpt, epsilons=1):
    attack_name, pytorch_device = arg_parser.attacks[attack_cpt], arg_parser.device
    print('running {}'.format(attack_name))
    if attack_name == 'FGSM':
        return(FGSM(epsilon=epsilons, max_epsilon = 2., num_classes=1000, device=pytorch_device, binary_search=True))
    elif attack_name == 'PGD':
        return(PGD(upper_radius=1000, num_classes=1000, device=pytorch_device, binary_search=True))
    elif attack_name == 'BP':
        return(BP(num_classes=1000, device=pytorch_device))
