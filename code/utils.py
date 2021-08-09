"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from torch.autograd import Variable

from torchvision import transforms
from model import  network_9layers,network_29layers,LightCNN_29Layers_v2
import torch
import os
import math
import torchvision.utils as vutils
import yaml
import numpy as np


def load_lightcnn(model_dir):
    """ Use the model from https://github.com/abhiskk/fast-neural-style/blob/master/neural_style/utils.py """
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    checkpoint_path = os.path.join(model_dir, 'LightCNN_9Layers_checkpoint.pth.tar')
    if os.path.exists(checkpoint_path):
        if os.path.isfile(checkpoint_path):
            print("=> loading checkpoint '{}'".format(checkpoint_path))

            lightcnn = network_9layers()
            lightcnn = torch.nn.DataParallel(lightcnn,device_ids=[torch.cuda.current_device()]).cuda()
            pretrained_dict = torch.load(checkpoint_path)
            lightcnn_dict = lightcnn.state_dict()
            pretrained_dict_temp = pretrained_dict['state_dict']
             
            pretrained_dict_remove = pretrained_dict_temp.pop('module.features.0.filter.weight')
            pretrained_dict_remove = pretrained_dict_temp.pop('module.features.0.filter.bias')
            pretrained_dict_remove = pretrained_dict_temp.pop('module.features.2.conv_a.filter.weight')
            pretrained_dict_remove = pretrained_dict_temp.pop('module.features.2.conv_a.filter.bias')
            pretrained_dict_remove = pretrained_dict_temp.pop('module.features.2.conv.filter.weight')
            pretrained_dict_remove = pretrained_dict_temp.pop('module.features.2.conv.filter.bias')
            pretrained_dict_remove = pretrained_dict_temp.pop('module.features.4.conv_a.filter.weight')
            pretrained_dict_remove = pretrained_dict_temp.pop('module.features.4.conv_a.filter.bias')
            pretrained_dict_remove = pretrained_dict_temp.pop('module.features.4.conv.filter.weight')
            pretrained_dict_remove = pretrained_dict_temp.pop('module.features.4.conv.filter.bias')
            pretrained_dict_remove = pretrained_dict_temp.pop('module.fc2.weight')
            pretrained_dict_remove = pretrained_dict_temp.pop('module.fc2.bias')
            pretrained_dict_temp["module.features.1.conv_a.filter.weight"] = pretrained_dict_temp.pop("module.features.6.conv_a.filter.weight")
            pretrained_dict_temp["module.features.1.conv_a.filter.bias"] = pretrained_dict_temp.pop("module.features.6.conv_a.filter.bias")
            pretrained_dict_temp["module.features.1.conv.filter.weight"] = pretrained_dict_temp.pop("module.features.6.conv.filter.weight")
            pretrained_dict_temp["module.features.1.conv.filter.bias"] = pretrained_dict_temp.pop("module.features.6.conv.filter.bias")
            pretrained_dict_temp["module.features.2.conv_a.filter.weight"] = pretrained_dict_temp.pop("module.features.7.conv_a.filter.weight")
            pretrained_dict_temp["module.features.2.conv_a.filter.bias"] = pretrained_dict_temp.pop("module.features.7.conv_a.filter.bias")
            pretrained_dict_temp["module.features.2.conv.filter.weight"] = pretrained_dict_temp.pop("module.features.7.conv.filter.weight")
            pretrained_dict_temp["module.features.2.conv.filter.bias"] = pretrained_dict_temp.pop("module.features.7.conv.filter.bias")
            pretrained_dict_temp["module.fc1.filter.weight"] = pretrained_dict_temp.pop("module.fc1.filter.weight")
            pretrained_dict_temp["module.fc1.filter.bias"] = pretrained_dict_temp.pop("module.fc1.filter.bias")
            # pretrained_dict_temp["module.fc2.weight"] = pretrained_dict_temp.pop("module.fc2.weight")
            # pretrained_dict_temp["module.fc2.bias"] = pretrained_dict_temp.pop("module.fc2.bias")
            # 1. filter out unnecessary keys
           # pretrained_dict1 = {k: v for k, v in pretrained_dict_temp.items() if k in lightcnn_dict}

            # 2. overwrite entries in the existing state dict
          #  lightcnn_dict.update(pretrained_dict1)

            # 3. load the new state dict
            lightcnn.load_state_dict(lightcnn_dict)
    else:
            print("=> no checkpoint found at '{}'".format(checkpoint_path))
    return lightcnn
