import torch
import numpy as np
import pandas
from torchvision import models, datasets
import cv2
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import glob
import os

class AdversarialDataset(Dataset):
    def __init__(self, arg_parser,img_size = 224):
        """
        Args:
            arg_parser: All the parsed arguments given at execution
            img_size: input size of the model
        """
        self.nat_images = glob.glob(arg_parser.input_path+'/*png')#[:10]
        self.device=arg_parser.device
        self.img_size = img_size
        self.labels = self.get_labels(arg_parser.labels_path)

    def __len__(self):
        return len(self.nat_images)

    def image_name(self, idx):
        file_name_index = self.nat_images[idx][::-1].index('/')
        return self.nat_images[idx][-file_name_index:-4]

    def get_labels(self, labels_path):
        labels = pandas.read_csv(labels_path)
        labels_val = labels.values
        labels_dict = {}
        for value_tuple in labels_val:
            labels_dict[value_tuple[0]]=value_tuple[1]
        return(labels_dict)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        nat_path = self.nat_images[idx]
        nat_image = cv2.imread(nat_path)[..., ::-1]
        nat_image = cv2.resize(nat_image, (self.img_size, self.img_size))
        nat_tensor = torch.autograd.Variable(torch.Tensor(nat_image),requires_grad=False)

        image_name = self.image_name(idx)
        label = int(self.labels['{}.png'.format(image_name)][1:-1])

        label = torch.tensor(label).to(self.device)
        image_name = [image_name]

        return (nat_tensor.to(self.device), label, image_name)


class Preprocessing_Layer(torch.nn.Module):
    """
    This is an added layer to the pytorch model, it allows to back-propagate gradients through the preprocessing of the images
    and then work on [0,255] images instead of the preprocessed domain. -> model = nn.Sequential(Preprocessing_Layer(), model)
    This preprocessing works on ResNet models as well as EfficientNet (when not trained with adv prop) with the following values:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    For EfficientNet trained with adv_prop, see Preprocessing_Layer_robust
    """
    def __init__(self, mean, std, torch_device='cpu'):
        super(Preprocessing_Layer, self).__init__()
        self.mean = mean
        self.std = std
        self.device = torch_device

    def preprocess(self, img, mean, std):
        image = img.clone().to(self.device)
        image /= 255.0

        image = image.transpose(1, 3).transpose(2, 3)
        image[:,0,:,:] = (image[:,0,:,:] - mean[0]) / std[0]
        image[:,1,:,:] = (image[:,1,:,:] - mean[1]) / std[1]
        image[:,2,:,:] = (image[:,2,:,:] - mean[2]) / std[2]

        return(image)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        res = self.preprocess(x, self.mean, self.std)
        return res

class Preprocessing_Layer_robust(torch.nn.Module):
    """
    This is an added layer to the pytorch model, it allows to back-propagate gradients through the preprocessing of the images
    and then work on [0,255] images instead of the preprocessed domain. -> model = nn.Sequential(Preprocessing_Layer(), model)
    This preprocessing works on EfficientNet trained with adv prop
    """
    def __init__(self,torch_device='cpu'):
        super(Preprocessing_Layer_robust, self).__init__()
        self.device = torch_device

    def forward(self, x):
        image = x/255.0

        image = image.transpose(1, 3).transpose(2, 3)
        res = image * 2.0 - 1.0
        res = res.to(self.device)
        return res

class AdversarialSample(torch.nn.Module):
    def __init__(self, initial_label, adversarial_label, adv_image, orig_image, torch_device='cpu'):
        super(AdversarialSample, self).__init__()
        self.initial_label = initial_label
        self.adversarial_label = adversarial_label
        self.adv_image = adv_image
        self.orig_image = orig_image
        self.device = torch_device

    def get_gradients(self, py_model):
        return(find_grads(initial_labels, adversarial_image, py_model))

def load_robust(init_model, weight_path):
    state_dict = torch.load(weight_path)
    checkpoint = state_dict['model']
    new_checkpoint = {}
    robust_fc, natural_fc = [], []
    for check_key in checkpoint.keys():
        if "module.attacker.model" in check_key:
                new_checkpoint[check_key.replace("module.attacker.model.","")]=checkpoint[check_key].to('cuda:0')

    init_model.load_state_dict(new_checkpoint)
    return init_model

def load_model(arg_parser, model_number):

    device = arg_parser.device
    print("loading {}".format(arg_parser.models[model_number]))
    if arg_parser.model_type=='timm':
        if 'timm' not in sys.modules:
            import timm
        model_init = timm.create_model(arg_parser.models[model_number], pretrained=True)
    else:
        model_init = getattr(models, arg_parser.models[model_number])(pretrained=True)
    if 'ap' in arg_parser.models[model_number]:
        preprocess_layer = Preprocessing_Layer_robust(torch_device=arg_parser.device)
    else:
        preprocess_layer = Preprocessing_Layer(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225], torch_device=arg_parser.device)

    softmax = nn.Softmax(dim=1)
    model = nn.Sequential(preprocess_layer, model_init, softmax)
    model.eval();
    model.to(device)

    # param_count = 0
    # for parameter in model.parameters():
    #     if len(parameter.shape)==1:
    #         param_count+=parameter.shape[0]
    #     else:
    #         temp = parameter.shape[0]
    #         for i in range(1,len(parameter.shape)):
    #             temp = temp*parameter.shape[i]
    #         param_count+=temp
    #
    # print(param_count, " parameters")

    return(model, device)

class AttackParser:
    def __init__(self, parser_obj):

        parser_obj.add_argument('--inputs', type=str, default='inputs/', help='path to folder containing images')
        parser_obj.add_argument('--outputs', type=str, default='outputs/', help='path to store adversarial images')
        parser_obj.add_argument('--model_type', type=str, default='torchvision', choices=['torchvision', 'timm'] )
        parser_obj.add_argument('--models', type=str, default='resnet50')#, choices=['alexnet','vgg16', 'googlenet','resnet18', 'resnet50','robust', 'efficientnet', 'regnety_032'], required=False, help="network to attack")
        parser_obj.add_argument('--attacks', type=str, default='BP', required=False, help="attack used")
        parser_obj.add_argument('--epsilon', type=float, default=1, required=False, help="epsilon for foolbox attacks")
        parser_obj.add_argument('--attack_type', type=str, default='custom', choices=['custom', 'foolbox'], required=False, help="attack framework used")
        parser_obj.add_argument('--quantizer', type=str, default='L2', choices=['L2', 'GINA', 'HILL', 'QUAD'], required=False, help="quantizer used")
        parser_obj.add_argument('--max_degree', type=int, default=1, required=False, help="maximum distortion on a given pixel from the adversarial perturbation")
        parser_obj.add_argument('--batch_size', type=int, default='6', required=False, help="batch size")
        parser_obj.add_argument('--labels_path', type=str, default='/nfs/nas4/bbonnet/bbonnet/datasets/labels/imagenet_2012_val/processed.csv', required=False, help="labels of the images, default will result in attacking the image regardless of the initial prediction")
        parser_obj.add_argument('--gpu', type=str, default='true', choices=['true', 'false'], required=False, help="determines usage of gpu device")
        parser_obj.add_argument('--jpeg', type=int, default='0', required=False, help="quality of target jpeg. If 0, save as png")
        args = parser_obj.parse_args()

        self.input_path = args.inputs
        self.models = args.models.split(',')
        self.save_path  = args.outputs
        self.model_type = args.model_type
        self.attacks = args.attacks.split(',')
        self.attack_type = args.attack_type
        self.quantization_method  = args.quantizer
        self.max_deg   = args.max_degree
        self.device       = ('cuda:0') if args.gpu=='true' else ('cpu:0')
        self.labels_path  = args.labels_path
        self.batch_size   = args.batch_size
        self.jpeg_quality = args.jpeg
        self.epsilon = args.epsilon

        print("Starting {} with {} quantization (max distortion from adversarial perturbation = {})".format(self.attacks, self.quantization_method, self.max_deg))
        print("attacking images in {}".format(self.input_path))
        print("storing results in {}".format(self.save_path))
        print("models used {} ".format(self.models))
        print("batch size {}".format(self.batch_size))
