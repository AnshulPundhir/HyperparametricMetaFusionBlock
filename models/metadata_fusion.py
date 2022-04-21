#Filename:	metadata_fusion.py
#Institute: IIT Roorkee

import torch.nn as nn
import torch

class meta_fusion(nn.Module):

    def __init__(self, V, U):
        super(meta_fusion, self).__init__()
        self.fb = nn.Sequential(nn.Linear(U, V), nn.BatchNorm1d(V))
        self.gb = nn.Sequential(nn.Linear(U, V), nn.BatchNorm1d(V))

    def forward(self, imgs, metadata):
        lambda_1 = 0.7
        lambda_2 = 0.3
        t1 = self.fb(metadata)
        t2 = self.gb(metadata)
        imgs = imgs.view(imgs.size(0),-1)
        act_obj = nn.LeakyReLU() 
        act_res = act_obj(imgs * t1)
        act_silu = nn.SiLU()
        V = act_silu(lambda_1 * act_res + lambda_2 * t2)
        
        return V