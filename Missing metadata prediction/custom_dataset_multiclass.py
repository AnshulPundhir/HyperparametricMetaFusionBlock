import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np 

class metadataset(Dataset):
    def __init__(self, X_filepath, Y_filepath):
        
         
        X_metadata= pd.read_csv(X_filepath, index_col=False)
        
        #below you can specify column names of the features you wish to use for predicting other features. 
        X_metadata= X_metadata[[ 'smoke', 'drink',
       'pesticide', 'gender_0', 'gender_1', 'skin_cancer_history',
       'cancer_history', 'has_piped_water', 'has_sewage_system', 'fitspatrick',
       'region_0', 'region_1', 'region_2', 'region_3', 'region_4', 'region_5',
       'region_6', 'region_7', 'region_8', 'region_9', 'region_10',
       'region_11', 'region_12', 'region_13', 
       'diagnostic_0', 'diagnostic_1', 'diagnostic_2', 'diagnostic_3',
       'diagnostic_4', 'diagnostic_5', 'itch', 'hurt', 'bleed', 'elevation','changed', 'grew',
       'norm_age', 'norm_diameter_1', 'norm_diameter_2']]
        
        self.Y_metadata= pd.read_csv(Y_filepath, index_col=False)
        X_data= X_metadata.values
        Y_data= self.Y_metadata.values
        self.Y_labels_column= self.Y_metadata.columns.item()
        
        self.X= torch.tensor(X_data, dtype= torch.float64) 
        self.Y = torch.tensor(Y_data ) 
        
    def __len__(self):
        return(len(self.Y))
    
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    
    def get_train_labels_weights(self):
        label_freq= np.array(self.Y_metadata[self.Y_labels_column].value_counts().sort_index(ascending= True))
        weights= 1/label_freq
        return weights
        
        
        
        










