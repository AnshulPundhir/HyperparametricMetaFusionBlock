import torch
import torch.nn as nn
import torch.nn.functional as F

class binary_classifier(nn.Module):
    def __init__(self, num_layers, drop_batchnorm= False):
        super(binary_classifier, self).__init__()
        self.num_layers= num_layers
        self.layer_config=[[37, 30, 22,  14,1], [37,  26, 15 ,1 ], [37, 18, 1]]
        #You can define your own layer_config depending on the number of features (encoded or otherwise) you are using to predict another feature.  
        self.drop_batchnorm= drop_batchnorm
        
        if self.num_layers==3:
            self.hidden_1= nn.Linear(self.layer_config[0][0], self.layer_config[0][1])
            self.hidden_2= nn.Linear(self.layer_config[0][1], self.layer_config[0][2]) 
            self.hidden_3= nn.Linear(self.layer_config[0][2], self.layer_config[0][3])
            self.batchnorm1 = nn.BatchNorm1d(self.layer_config[0][1])
            self.batchnorm2 = nn.BatchNorm1d(self.layer_config[0][2])
            self.batchnorm3 = nn.BatchNorm1d(self.layer_config[0][3])
            self.output= nn.Linear(self.layer_config[0][3], self.layer_config[0][4])   
            
        elif self.num_layers==2:
            self.hidden_1= nn.Linear(self.layer_config[1][0], self.layer_config[1][1])
            self.hidden_2= nn.Linear(self.layer_config[1][1], self.layer_config[1][2]) 
            self.batchnorm1 = nn.BatchNorm1d(self.layer_config[1][1])
            self.batchnorm2 = nn.BatchNorm1d(self.layer_config[1][2])     
            self.output= nn.Linear(self.layer_config[1][2], self.layer_config[1][3])
            
        elif self.num_layers==1:
            self.hidden_1= nn.Linear(self.layer_config[2][0], self.layer_config[2][1])    
            self.output= nn.Linear(self.layer_config[2][1], self.layer_config[2][2])
            
            
    def forward(self, x):
        if self.drop_batchnorm== False and self.num_layers!=1:
            if self.num_layers==3:
                x = F.relu(self.hidden_1(x))
                x = F.relu(self.hidden_2(x))
                x = F.relu(self.hidden_3(x))
                x = self.output(x)
                
            elif self.num_layers==2:
                x = F.relu(self.hidden_1(x))
                x = F.relu(self.hidden_2(x))
                x = self.output(x)
                
                

        elif self.drop_batchnorm== True and self.num_layers!=1:
            if self.num_layers==3:
                x = F.relu(self.batchnorm1(self.hidden_1(x)))
                x = F.dropout(F.relu(self.batchnorm2(self.hidden_2(x))), p=0.3)
                x = F.dropout(F.relu(self.batchnorm3(self.hidden_3(x))), p=0.3)
                x = self.output(x)
                
            elif self.num_layers==2:
                x = F.relu(self.batchnorm1(self.hidden_1(x)))
                x = F.dropout(F.relu(self.batchnorm2(self.hidden_2(x))), p=0.3)
                x = self.output(x)
                
        elif self.num_layers==1:
            x = F.relu(self.hidden_1(x))
            x = self.output(x)         
                
                
        return x
             
        
                