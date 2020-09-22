from .fgsm_attack import FGSM
from .pgd_attack2 import PGD
from .cw_attack import CarliniWagnerL2 as CW2
from .ddn_attack import DDN


# attack = DDN(init_norm = 100.,device=device)
# attack2 = DDN(init_norm = 200.,device=device)
# attack3 = DDN(init_norm = 500.,device=device)

def attack_generator(attack_name, pytorch_device):
    if attack_name == 'FGSM':
        return(FGSM(epsilon=1, max_epsilon = 15., num_classes=1000, device=pytorch_device, binary_search=True))
    elif attack_name == 'PGD':
        return(PGD(upper_radius=1000, num_classes=1000, device=pytorch_device, binary_search=True))
    elif attack_name == 'DDN':
        return(DDN(init_norm = 500.,device=device))
    elif attack_name == 'CW2':
        return(CW2(image_constraints=(0.0,255.0),num_classes=1000, device=pytorch_device))
