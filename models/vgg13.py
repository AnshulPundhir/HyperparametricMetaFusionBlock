#Filename:	vgg13.py
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


class vgg13_fusion(nn.Module):
    def __init__(self, im_size, num_classes, init_weights = "kaimingNormal"):
        super(vgg13_fusion, self).__init__()
        self.im_size = im_size
        self.num_classes = num_classes
        self.init_weights = init_weights

        self.meta_fusion = meta_fusion(512, 73)

        self.model = models.vgg13_bn(pretrained=True)

        for param in self.model.parameters(): 
            param.requires_grad = True

        self.part = self.model.features   

        self.avgpool = self.model.avgpool
        self.adaptive_conv = nn.Conv2d(512, 512, kernel_size = int(im_size / 32), padding = 0, bias = True)

        self.fc_block = nn.Sequential(
            nn.Linear(512, 256, bias = True),
            nn.BatchNorm1d(256, eps = 1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128, bias = True),
            nn.BatchNorm1d(128, eps = 1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64, bias = True),
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
        input_x = F.max_pool2d(input_x,kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        input_x = self.avgpool(input_x)  
        g = self.adaptive_conv(input_x)
        g_meta = self.meta_fusion(g, metafeat_x)
        x = self.fc_block(g_meta)
        x = self.classifier(x)

        return [x, None, None, None] 