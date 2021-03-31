## About

Code for the article "What if Adversarial Samples were Digital Images?" (https://hal.archives-ouvertes.fr/hal-02553006v2), IH&MMSec 2020\

Implementation is done and tested PyTorch 1.5.1 and Python 3.7\

This code provides quantization for adversarial samples as well as the possibility of saving the resulting images. \

This code also provides custom versions of PGD_2 and FGSM by adding an optimized binary search on distortion as well as BP (https://hal.archives-ouvertes.fr/hal-02931493) by adding an optimized initialization. With few iterations, BP beats every attack in their L2 form even ressource-greedy attacks such as C&W2


## Other requirements

Models: \
**torchvision** (0.6.1)\
**timm** (https://github.com/rwightman/pytorch-image-models) (0.3.4).\
Attacks:\
**foolbox** (https://github.com/bethgelab/foolbox) (3.2.1)


## Getting started

To run an attack and quantize it simply run:

```
python quantize.py 
```
This will run with all defaults parameters (attacking ResNet50 with BP), storing measures in outputs/measures and images in outputs/images. \

You can run multiple attacks on multiple models:
```
python quantize.py --models resnet50,vgg16,alexnet --attacks BP,FGSM
```
\

By default, attacks are pulled from our custom attacks and models from torchvision. To use an attack from foolbox and a model from timm:
```
python quantize.py --model_type timm --models efficientnet_b0 --attack_type foolbox --attacks DDNAttack,L2CarliniWagnerAttack
```
This code will attack efficientnet-b0 with DDN and C&W (with default parameters). Attack names can be found at: https://foolbox.readthedocs.io/en/stable/modules/attacks.html#


### Other parameters

```
--batch_size
```
Change batch size (default 6)
```
--gpu false
```
Define usage of gpu (default  `true`)
```
--inputs path_to_inputs --labels_path path_to_labels --outputs path_to_outputs
```
Where to pull data and labels from, where to store results. (default respectively `./inputs`, `./labels` and `./outputs`)

### Misc

To test the accuracy of a given model you can run 

```
python accuracy_test.py --model_type timm --models efficientnet_b0,mobilenetv2_100
```
Note that some models require a different preprocessing and implementation can be clanky at times. Code also defaults for an image size of 224 x 224 x 3\
To plot measures obtained from an experiment
```
python draw_curves.py --upper 1 --inputs path_to_measures --outputs path_to_store
```
`--upper` corresponds to the maximum range of the x-axis (default 1= an average distortion of 1 on a pixel).
