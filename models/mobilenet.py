#Filename:	mobilenet.py
#Institute: IIT Roorkee

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, ".")
from utils.initialize import *
from torchvision import datasets, models, transforms
from collections import OrderedDict
from models.metadata_fusion import meta_fusion

class mobilenet_fusion(nn.Module):
    def __init__(self, im_size, num_classes, init_weights = "kaimingNormal"):
        super((mobilenet_fusion), self).__init__()
        self.im_size = im_size
        self.num_classes = num_classes
        self.init_weights = init_weights
        self.meta_fusion = meta_fusion( 1280, 73) 

        self.model = models.mobilenet_v2(pretrained=True)

        for param in self.model.parameters(): 
            param.requires_grad = True

        self.part = self.model.features 
        self.conv_block1 = self._conv_block(1280, 512, 2)
        self.adaptive_conv = nn.Conv2d(512, 1280, kernel_size = 4, padding = 0, bias = True)

        self.fc_block = nn.Sequential(
            nn.Linear(1280 , 512, bias = True),
            nn.BatchNorm1d(512, eps = 1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256, bias = True),
            nn.BatchNorm1d(256, eps = 1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64, bias = True),
            nn.BatchNorm1d(64, eps = 1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.classifier = nn.Linear(64, self.num_classes, bias = True)


        # weights initialization
        if init_weights == "kaimingUniform":
            weights_init_kaimingUniform(self)
        elif init_weights == "kaimingNormal":
            print('Initializing Weights')
            weights_init_kaimingNormal(self.conv_block1)
            weights_init_kaimingNormal(self.adaptive_conv) 
            weights_init_kaimingNormal(self.fc_block)
            weights_init_kaimingNormal(self.classifier)
        elif init_weights == "xavierUniform":
            weights_init_kaimingNormal(self)
        elif init_weights == "xavierNormal":
            weights_init_kaimingNormal(self)
        else:
            raise NotImplementedError("Invalid type of initialization")

    def forward(self, input_x,metafeat_x):
        input_x = self.part(input_x)
        input_x = self.conv_block1(input_x) 
        input_x = F.max_pool2d(input_x, kernel_size = 2, stride = 2, padding = 1,dilation=1, ceil_mode=False) 
        g = self.adaptive_conv(input_x) 
        gmeta = self.meta_fusion(g, metafeat_x)
        gmeta = gmeta.view(gmeta.size(0), -1)
        x = self.fc_block(gmeta) 
        x = self.classifier(x)

        return [x, None, None, None]

    def _conv_block(self, input_depth, num_filters, num_layers, is_pool = False):
        layers = []
        layers.append(nn.Conv2d(input_depth, num_filters, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)))
        layers.append(nn.BatchNorm2d(num_filters, eps = 1e-05, momentum=0.1, affine=True, track_running_stats=True))
        layers.append(nn.Dropout(0.20))
        layers.append(nn.ReLU(inplace = True))
        if is_pool:
            layers.append(nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, dilation = 1, ceil_mode = False))

        for i in range(num_layers - 1):
            layers.append(nn.Conv2d(num_filters, num_filters, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)))
            layers.append(nn.BatchNorm2d(num_filters, eps = 1e-05, momentum=0.1, affine=True, track_running_stats=True))
            layers.append(nn.Dropout(0.20))
            layers.append(nn.ReLU(inplace = True))
            if is_pool:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))

        return nn.Sequential(*layers) 