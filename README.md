## About

Code for the article "What if Adversarial Samples were Digital Images?" (https://hal.archives-ouvertes.fr/hal-02553006v2), IH&MMSec 2020

Implementation is done and tested PyTorch 1.5.1 and Python 3.7

This code provides quantization for adversarial samples as well as the possibility of saving the resulting images. 

This code also provides custom versions of PGD_2 and FGSM by adding an optimized binary search on distortion as well as BP (https://hal.archives-ouvertes.fr/hal-02931493) by adding an optimized initialization. With few iterations, BP beats every attack in their L2 form even ressource-greedy attacks such as C&W2


## Other requirements

Models: 
**torchvision** (0.6.1)
**timm** (https://github.com/rwightman/pytorch-image-models) (0.3.4).
Attacks:
**foolbox** (https://github.com/bethgelab/foolbox) (3.2.1)


## Adversarial training with DDN

The following commands were used to adversarially train the models:

MNIST:
```
python -m fast_adv.defenses.mnist --lr=0.01 --lrs=30 --adv=0 --max-norm=2.4 --sn=mnist_adv_2.4
```

CIFAR-10 (adversarial training starts at epoch 200):
```
python -m fast_adv.defenses.cifar10 -e=230 --adv=200 --max-norm=1 --sn=cifar10_wrn28-10_adv_1
```

### Adversarially trained models 

* MNIST: https://www.dropbox.com/s/9onr3jfsuc3b4dh/mnist.pth
* CIFAR10: https://www.dropbox.com/s/ppydug8zefsrdqn/cifar10_wrn28-10.pth
