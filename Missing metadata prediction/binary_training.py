from custom_dataset_binary import metadataset
from binary_classifier import binary_classifier

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("The device used for training is ", device)


def accuracy(y_pred, y_actual):
    y_pred_sigmoid = torch.sigmoid(y_pred)   
    
    y_pred_labels= (y_pred_sigmoid>0.7)*1         #you can experiment with different thresholds for predicting accuracy for different features. 
    correct_pred = (y_pred_labels == y_actual)*1
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc * 100)
    
    return acc

#in the list below, specify list of features you want to make binary (i.e. 0 or 1) predictions for. 
for feature in ['smoke', 'drink', 'pesticide', 'gender_encoded', 'skin_cancer_history', 'cancer_history', 'has_piped_water', 'has_sewage_system']: 
    for lr in [0.001, 0.0001]:
        for optim_fn in ['Adam', 'SGD']:
            for loss_fn in ['weightedBCE', 'nonweightedBCE']:                
                for num_hidden_layers in [1,2,3]:
                    for batchnorm_dropout in [True, False]:                         
                        
                        train_set= metadataset('automatic input of filepath of X train set for the particular feature, based on feature name', 'automatic input of filepath of Y train set for the particular feature, based on feature name')
                        train_loader= DataLoader(train_set, batch_size=8, shuffle= True, drop_last=True)
                        
                        test_set= metadataset('automatic input of filepath of X test set for the particular feature, based on feature name', 'automatic input of filepath of Y test set for the particular feature, based on feature name')
                        test_loader= DataLoader(test_set, batch_size=1)
                        
                        if num_hidden_layers==1:
                            batchnorm_dropout=False
                        model= binary_classifier(num_hidden_layers, batchnorm_dropout)
                
                
                        accuracy_stats = {
                            'train': [],
                            "test": []
                        }
                        loss_stats = {
                            'train': [],
                            "test": []
                        }
                        
                        model.to(device)
                        loss_weights= train_set.get_pos_weights()
                        class_weights = torch.from_numpy(np.array(loss_weights)).to(device)
                        
                        if loss_fn=='nonweightedBCE':
                            criterion = nn.BCEWithLogitsLoss()
                        elif loss_fn=='weightedBCE':
                            criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights, reduction="mean")
                        
                        if optim_fn=='Adam':
                            optimizer = optim.Adam(model.parameters(), lr=lr)
                        else:
                            optimizer=optim.SGD(model.parameters(), lr=lr, )  
                
                        epochs = 30
            
                        for e in range(epochs):
                            
                            # CODE FOR TRAINING
                            model.train()
                            train_epoch_loss = 0
                            train_epoch_acc=0
                            
                            for X_train_batch, y_train_batch in train_loader:
                                
                                # TODO: Training pass
                                X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
                                optimizer.zero_grad()
                                #print(X_train_batch.shape)
                                output = model(X_train_batch.float())
                                #print('output:', output, '\n')
                                loss = criterion(output, y_train_batch)
                                acc= accuracy(output, y_train_batch)
                                loss.backward()
                                optimizer.step()
                                
                                train_epoch_loss += loss.item()
                                train_epoch_acc += acc.item()
                                                            
                            #VALIDATION    
                            with torch.no_grad():
                                
                                val_epoch_loss = 0
                                val_epoch_acc = 0
                                
                                model.eval()
                                for X_val_batch, y_val_batch in test_loader:
                                    X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                                    
                                    y_val_pred = model(X_val_batch.float())
                                                
                                    val_loss = criterion(y_val_pred, y_val_batch)
                                    val_acc = accuracy(y_val_pred, y_val_batch)
                                    
                                    val_epoch_loss += val_loss.item()
                                    val_epoch_acc += val_acc.item()
                                    
                            loss_stats['train'].append(train_epoch_loss/len(train_loader))
                            loss_stats['test'].append(val_epoch_loss/len(test_loader))
                            accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
                            accuracy_stats['test'].append(val_epoch_acc/len(test_loader))
                                
                            if e!=0:
                                if (accuracy_stats['test'][e]>accuracy_stats['test'][e-1]):
                                    
                                    checkpoint = {'num_layers': num_hidden_layers,
                                    'drop_batchnorm': batchnorm_dropout,
                                    'state_dict': model.state_dict()}                            
                    
                                    torch.save(checkpoint, 'filepath_of_weights_to_be_saved'.pth)
                                    print("Weights saved for epoch ", e, '\n')                                  
                            print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(test_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(test_loader):.3f}', '\n')
                
                        train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
                        train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
                            
                        # Plot the dataframes into plots. 
                        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,7))
                        acc_plot=sns.lineplot(data=train_val_acc_df, x = "epochs", y="value", hue="variable",  ax=axes[0]).set_title('Train-Val Accuracy/Epoch')
                        loss_plot=sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable", ax=axes[1]).set_title('Train-Val Loss/Epoch')
                        acc_fig= acc_plot.get_figure()
                        loss_fig= loss_plot.get_figure()
                        acc_fig.savefig("specify_filepath"+".jpg")
                        loss_fig.savefig("specify_filepath" +".jpg")
                        plt.close('all')

                        
                        if num_hidden_layers==1:
                            break    