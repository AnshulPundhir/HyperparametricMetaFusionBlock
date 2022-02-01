'''

This is the partial code for efficient net model 
Full Code will be available soon after acceptance of our research paper

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from efficientnet_pytorch import EfficientNet
from model.MetaFusion import MetaFusion            # will be released soon after acceptance of our research paper


class effnet_mid_AM(nn.Module):
    def __init__(self, im_size, num_classes):
        super(effnet_mid_AM, self).__init__()

        self.im_size = im_size
        self.num_classes = num_classes

        self.model = EfficientNet.from_pretrained('efficientnet-b4')

        self.part = nn.Sequential(
            self.model._conv_stem,
            self.model._bn0,
            *self.model._blocks,
            self.model._conv_head,
            self.model._bn1,
            self.model._avg_pooling,
            self.model._dropout
        )

        self.metafusion = MetaFusion(1792, 73)       # will be released soon after acceptance of our research paper

        self.fc_block = nn.Sequential(
                # will be released soon after acceptance of our research paper
        ) 
            
        self.classifier = nn.Linear(256, self.num_classes, bias = True)

    def forward(self, input_x, metafeat_x): 
        input_x = self.part(input_x) 

        g_meta = self.metafusion(input_x, metafeat_x) 
        x = self.fc_block(g_meta) 
        x = self.classifier(x)

        return [x]


