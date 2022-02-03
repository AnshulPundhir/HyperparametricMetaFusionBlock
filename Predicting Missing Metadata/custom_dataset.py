'''
This is the partial code for the custom dataset made to predict missing metafeatures.  
Full Code will be available soon after acceptance of our research paper

'''


import pandas as pd
from torch.utils.data import Dataset
import numpy as np 

class metadataset(Dataset):
    def __init__(self, X_filepath, Y_filepath, multiclass=False):
        #multiclass Flag denotes whether the feature we want to predict values for is binary or multiclass. 
        #X_filepath is the path of the csv file containg the encoded features to be used for predicting the target feature. 
        #Y_filepath is the path of the csv file containg the true values of the target feature, where the true values are label encoded. 
        
        X_metadata= pd.read_csv(X_filepath)
        X_metadata= X_metadata[]  #specify the feature-names as a list, which you want to use to predict the target feature. 
        self.Y_metadata= pd.read_csv(Y_filepath)
        self.multiclass= multiclass
        X_data= X_metadata.values
        Y_data= self.Y_metadata.values
        self.Y_labels_column= self.Y_metadata.columns.item()
        
        self.X= torch.tensor(X_data, dtype= torch.float64) 
        if self.multiclass:
            self.Y = torch.tensor(Y_data) 
        else: 
            self.Y = torch.tensor(Y_data, dtype= torch.float32) 

        
    def __len__(self):
        return(len(self.Y))
    
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    
    def get_binary_weights(self):
        label_freq= np.array(self.Y_metadata[self.Y_labels_column].value_counts().sort_index(ascending= True))
        pos_wt= label_freq[0]/label_freq[1] 
        return pos_wt

    def get_multiclass_weights(self):
        label_freq= np.array(self.Y_metadata[self.Y_labels_column].value_counts().sort_index(ascending= True))
        weights= 1/label_freq
        return weights
