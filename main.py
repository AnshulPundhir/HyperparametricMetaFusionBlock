#Filename:	main.py
#Institute: IIT Roorkee

import argparse
import os
import copy 
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from datetime import datetime
from tensorboardX import SummaryWriter 
from train_with_metadata import *
from utils.custom_dataset import *
import gc


def parse_param():
    """
    parse the arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-cls", type = int, default = 6, help = "dataset classes")
    parser.add_argument("-gpu", type = bool, default = True, help = "Use gpu to accelerate")
    parser.add_argument("-batch_size", type = int, default = 32, help = "batch size for dataloader")
    parser.add_argument("-lr", type = float, default = 0.01, help = "initial learning rate")
    parser.add_argument("-epoch", type = int, default = 150, help = "training epoch")
    parser.add_argument("-optimizer", type = str, default = "sgd", help = "optimizer")
    args = parser.parse_args()

    return args

def print_param(args):          
    """
    print the arguments
    """
    print("-" * 15 + "training configuration" + "-" * 15)
    print("class number:{}".format(args.cls))
    print("batch size:{}".format(args.batch_size))
    print("gpu used:{}".format(args.gpu))
    print("learning rate:{}".format(args.lr))
    print("training epoch:{}".format(args.epoch))
    print("optimizer used:{}".format(args.optimizer))
    print("-" * 53)

def run(model, train_loader, test_loader, optimizer, loss_func,  writer, train_scheduler, epoch, _output_path, _model_name):

    best_acc = 0
    best_top5 = 0
    best_model = model
    tn_loss, tn_acc, val_loss, val_acc = [], [], [], []

    for i in range(epoch):
        gc.collect()
        torch.cuda.empty_cache() 
        torch.cuda.memory_summary(device=None, abbreviated=False)    
        print("Epoch {}".format(i))

        # performance on training set
        model, train_loss, train_acc, time_elapsed = train(model, train_loader, loss_func, optimizer, True)
        print("Training set: Epoch {}, Loss {}, Accuracy {}, Time Elapsed {}".format(i, train_loss / len(train_loader.dataset), train_acc / len(train_loader.dataset), time_elapsed))
        writer.add_scalar("Train/loss", train_loss / len(train_loader.dataset), i)
        writer.add_scalar("Train/acc", train_acc / len(train_loader.dataset), i)

        tn_acc.append(train_acc / len(train_loader.dataset))
        tn_loss.append(train_loss / len(train_loader.dataset))

        # record the layers' gradient
        for name, param in model.named_parameters():
            if "weight" in name and not isinstance(param.grad, type(None)):
                layer, attr = os.path.splitext(name)
                attr = attr[1:]
                writer.add_histogram("{}/{}_grad".format(layer, attr), param.grad.clone().cpu().data.numpy(), i)

        # record the weights distribution
        for name, param in model.named_parameters():
            if "weight" in name:
                layer, attr = os.path.splitext(name)
                attr = attr[1:]
                writer.add_histogram("{}/{}".format(layer, attr), param.clone().cpu().data.numpy(), i)

        # performance on test set
        test_loss, test_acc, top5, time_elapsed = test(model, test_loader, loss_func, True)
        print("Validation set: Epoch {}, Loss {}, Accuracy {}, Top 5 {}, Time Elapsed {}".format(i, test_loss / len(test_loader.dataset), test_acc / len(test_loader.dataset), top5 / len(test_loader.dataset), time_elapsed))
        writer.add_scalar("Val/loss", test_loss / len(test_loader.dataset), i)
        writer.add_scalar("Val/acc", test_acc / len(test_loader.dataset), i)
        writer.add_scalar("Val/top5", top5 / len(test_loader.dataset), i)
    
        test_acc = float(test_acc) / len(test_loader.dataset)
        top5 = float(top5) / len(test_loader.dataset)

        val_acc.append(test_acc)
        val_loss.append(test_loss / len(test_loader.dataset))

        train_scheduler.step(test_loss / len(test_loader.dataset))
        
        print('Current Lr: ',optimizer.param_groups[0]['lr'])

        if test_acc > best_acc:
            best_acc = test_acc
            best_model = copy.deepcopy(model)
            best_iters = i

        if top5 > best_top5:
            best_top5 = top5    

    result = pd.DataFrame({'Train Accuracy':tn_acc,'Train Loss':tn_loss,'Validation Accuracy': val_acc,'Validation Loss':val_loss})
    result.to_csv(os.path.join(_output_path,'epoch_{}.csv'.format(_model_name)),index  = False)

    return best_model, best_acc, best_top5, best_iters



if __name__ == "__main__":

    # Clear occupied memory
    gc.collect()
    torch.no_grad()
    torch.cuda.empty_cache() 
    torch.cuda.memory_summary(device=None, abbreviated=False)

    args = parse_param()
    print_param(args)

    _folder = 1
    _sched_factor = 0.1
    _sched_min_lr = 1e-6
    _sched_patience = 20

    _dir_path = ""             # Path to Working Directory 
    _base_path = os.path.join(_dir_path, "PAD_UFES_20")  
    _model_name = ""           # Name of the Model 
    _output_path = os.path.join(_dir_path,'weights',_model_name)

    _imgs_folder_train = os.path.join(_base_path, "final_imgs_all_PAD")
    _csv_path_train_val_encoded = os.path.join(_base_path, "metadata_encoded_train_val.csv") 
    _csv_path_test_encoded = os.path.join(_base_path, "metadata_encoded_test.csv")  

    metadata_test_encoded      = pd.read_csv(_csv_path_test_encoded)  
    metadata_train_val_encoded = pd.read_csv(_csv_path_train_val_encoded)
      
    val_csv_folder =   metadata_train_val_encoded[ (metadata_train_val_encoded['folder'] == _folder) ]
    train_csv_folder = metadata_train_val_encoded[ metadata_train_val_encoded['folder'] != _folder ]
    
    initial_features= ['diameter_1', 'age',
    'diameter_2', 'background_father_0',
    'background_father_1',
    'background_father_2',
    'background_father_3',
    'background_father_4',
    'background_father_5',
    'background_father_6',
    'background_father_7',
    'background_father_8',
    'background_father_9',
    'background_father_10',
    'background_father_11', 'background_mother_0',
    'background_mother_1',
    'background_mother_2',
    'background_mother_3',
    'background_mother_4',
    'background_mother_5',
    'background_mother_6',
    'background_mother_7',
    'background_mother_8',
    'background_mother_9', 'gender_0',
    'gender_1', 'region_0',
    'region_1',
    'region_2',
    'region_3',
    'region_4',
    'region_5',
    'region_6',
    'region_7',
    'region_8',
    'region_9',
    'region_10',
    'region_11',
    'region_12',
    'region_13', 'fitspatrick_0',
    'fitspatrick_1',
    'fitspatrick_2',
    'fitspatrick_3',
    'fitspatrick_4',
    'fitspatrick_5', 'smoke_0',
    'smoke_1',
    'drink_0',
    'drink_1',
    'pesticide_0',
    'pesticide_1',
    'skin_cancer_history_0',
    'skin_cancer_history_1',
    'cancer_history_0',
    'cancer_history_1', 'has_piped_water_0',
    'has_piped_water_1',
    'has_sewage_system_0',
    'has_sewage_system_1',
    'itch_0',
    'itch_1',
    'grew_0',
    'grew_1',
    'hurt_0',
    'hurt_1',
    'changed_0',
    'changed_1', 
    'bleed_0',
    'bleed_1',
    'elevation_0',
    'elevation_1'
    ]

    metadata_train_encoded = metadata_train_val_encoded[ metadata_train_val_encoded['folder'] != _folder ][initial_features]
    
    train_age_mu = metadata_train_encoded['age'].mean()
    train_D1_mu  = metadata_train_encoded['diameter_1'].mean()
    train_D2_mu  = metadata_train_encoded['diameter_2'].mean()
    
    train_age_sigma = metadata_train_encoded['age'].std()
    train_D1_sigma  = metadata_train_encoded['diameter_1'].std()
    train_D2_sigma  = metadata_train_encoded['diameter_2'].std()
    
    metadata_train_encoded['norm_age']= (metadata_train_encoded['age']-train_age_mu)/train_age_sigma
    metadata_train_encoded['norm_D1']= (metadata_train_encoded['diameter_1']-train_D1_mu)/train_D1_sigma
    metadata_train_encoded['norm_D2']= (metadata_train_encoded['diameter_2']-train_D2_mu)/train_D2_sigma
    
    final_features = ['norm_D1','norm_D2', 'norm_age',
    'background_father_0',
    'background_father_1',
    'background_father_2',
    'background_father_3',
    'background_father_4',
    'background_father_5',
    'background_father_6',
    'background_father_7',
    'background_father_8',
    'background_father_9',
    'background_father_10',
    'background_father_11', 'background_mother_0',
    'background_mother_1',
    'background_mother_2',
    'background_mother_3',
    'background_mother_4',
    'background_mother_5',
    'background_mother_6',
    'background_mother_7',
    'background_mother_8',
    'background_mother_9', 'gender_0',
    'gender_1', 'region_0',
    'region_1',
    'region_2',
    'region_3',
    'region_4',
    'region_5',
    'region_6',
    'region_7',
    'region_8',
    'region_9',
    'region_10',
    'region_11',
    'region_12',
    'region_13', 'fitspatrick_0',
    'fitspatrick_1',
    'fitspatrick_2',
    'fitspatrick_3',
    'fitspatrick_4',
    'fitspatrick_5', 'smoke_0',
    'smoke_1',
    'drink_0',
    'drink_1',
    'pesticide_0',
    'pesticide_1',
    'skin_cancer_history_0',
    'skin_cancer_history_1',
    'cancer_history_0',
    'cancer_history_1', 'has_piped_water_0',
    'has_piped_water_1',
    'has_sewage_system_0',
    'has_sewage_system_1',
    'itch_0',
    'itch_1',
    'grew_0',
    'grew_1',
    'hurt_0',
    'hurt_1',
    'changed_0',
    'changed_1', 
    'bleed_0',
    'bleed_1',
    'elevation_0',
    'elevation_1'
    ]

    metadata_train_norm_encoded= metadata_train_encoded[final_features]
    
    metadata_val_encoded = metadata_train_val_encoded[ metadata_train_val_encoded['folder'] == _folder ][initial_features]
    metadata_val_encoded['norm_age'] = (metadata_val_encoded['age']-train_age_mu)/train_age_sigma
    metadata_val_encoded['norm_D1']  = (metadata_val_encoded['diameter_1']-train_D1_mu)/train_D1_sigma
    metadata_val_encoded['norm_D2']  = (metadata_val_encoded['diameter_2']-train_D2_mu)/train_D2_sigma
    metadata_val_norm_encoded= metadata_val_encoded[final_features]
    
    metadata_test_encoded = metadata_test_encoded[initial_features]
    metadata_test_encoded['norm_age'] = (metadata_test_encoded['age']-train_age_mu)/train_age_sigma
    metadata_test_encoded['norm_D1']  = (metadata_test_encoded['diameter_1']-train_D1_mu)/train_D1_sigma
    metadata_test_encoded['norm_D2']  = (metadata_test_encoded['diameter_2']-train_D2_mu)/train_D2_sigma
    metadata_test_norm_encoded= metadata_test_encoded[final_features]

    print('Size of training Data: ',len(train_csv_folder))
    print('Size of Validaton Data: ',len(val_csv_folder))

    train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(),]), p=0.1),
            transforms.RandomPerspective(distortion_scale=0.6, p=0.05), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        ]) 

    train_imgs_id = train_csv_folder['img_id'].values
    train_imgs_path = ["{}/{}".format(_imgs_folder_train, img_id)[0:-3] + 'jpg' for img_id in train_imgs_id]
    train_labels = list(train_csv_folder['diagnostic_number'].values)
    metadata_train_norm_encoded= list(metadata_train_norm_encoded.values)

    print('number of classes:  ',train_csv_folder['diagnostic_number'].nunique())

    train_dataset = meta_img_dataset(train_imgs_path, metadata_train_norm_encoded, train_labels, train_transform) 
    train_loader = DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 20)

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

    val_imgs_id = val_csv_folder['img_id'].values
    val_imgs_path = ["{}/{}".format(_imgs_folder_train, img_id)[0:-3] + 'jpg' for img_id in val_imgs_id]
    val_labels = val_csv_folder['diagnostic_number'].values   
    metadata_val_norm_encoded = list(metadata_val_norm_encoded.values)  

    # create evaluation data
    val_dataset = meta_img_dataset_test(val_imgs_path, metadata_val_norm_encoded, val_labels, val_transform)
    val_loader = DataLoader(dataset = val_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 20)

    csv_test = pd.read_csv(_csv_path_test_encoded) 

    print('Size of Test Data: ',len(csv_test))
    
    test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ]) 

    test_imgs_id = csv_test['img_id'].values
    test_imgs_path = ["{}/{}".format(_imgs_folder_train, img_id)[0:-3] + 'jpg' for img_id in test_imgs_id]
    test_labels = csv_test['diagnostic_number'].values  
    metadata_test_norm_encoded = list(metadata_test_norm_encoded.values)  
   
    # create evaluation data
    test_dataset = meta_img_dataset_test(test_imgs_path, metadata_test_norm_encoded, test_labels, test_transform)
    test_loader = DataLoader(dataset = test_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 20)

    # weighted cross-entropy loss function
    ser_lab_freq = train_csv_folder.groupby("diagnostic")["img_id"].count() 
    _labels_name = ser_lab_freq.index.values 
    _freq = ser_lab_freq.values
    _weights = (_freq.sum() / _freq).round(3) 
    print('ser_lab_freq:    ',ser_lab_freq)
    # specify the loss function
    loss_func = torch.nn.CrossEntropyLoss(weight=torch.Tensor(_weights).cuda())
    
    # specify the model
    if _model_name == 'densenet':
        from models.densenet import densenet_fusion
        model = densenet_fusion(224, 6)
    elif _model_name == 'effnet':
        from models.effnet import effnet_fusion
        model = effnet_fusion(224, 6)
    elif _model_name == 'resnet':
        from models.resnet import resnet_fusion
        model = resnet_fusion(224, 6)
    elif _model_name == 'mobilenet':
        from models.mobilenet import mobilenet_fusion
        model = mobilenet_fusion(224, 6)    
    else:
        from models.vgg13 import vgg13_fusion
        model = vgg13_fusion(224, 6)          

    # specify gpu used
    if args.gpu == True:
        model = model.cuda()
        loss_func = loss_func.cuda()
    
    # specify optimizer
    if args.optimizer == 'sgd':
        print('SGD Optimizer')
        optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum = 0.9, weight_decay = 0.001)
    else:
        optimizer = optim.momentum(model.parameters(), lr = args.lr, momentum = 0.9, weight_decay = 0.002)
    
    scheduler_lr = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=_sched_factor, min_lr=_sched_min_lr,
                                                                    patience=_sched_patience)

    # specify the epoches
    epoch = args.epoch

    Time = "{}".format(datetime.now().isoformat(timespec='seconds')).replace(':', '-')
    writer = SummaryWriter(log_dir = os.path.join("./log/", Time))
    best_model, best_acc, best_top5, best_iters = run(model, train_loader, val_loader, optimizer, loss_func, writer, scheduler_lr, epoch, _output_path, _model_name)

    print("Best acc {} at iteration {}, Top 5 {}".format(best_acc, best_iters, best_top5))

    # ON TEST dATASET:
    test_loss, test_acc, test_bacc, top5, time_elapsed, prediction, real_labels = test(best_model, test_loader, loss_func, True, True)
    print("Test set: Loss {}, Accuracy {}, BACC {}, Top 5 {}, Time Elapsed {}".format(test_loss / len(test_loader.dataset), test_acc / len(test_loader.dataset), test_bacc, top5 / len(test_loader.dataset), time_elapsed))
    
    pred_csv = pd.DataFrame({'Labels':real_labels,'Prediction':prediction}) 
    record = {0:'ACK',1:'BCC',2:'MEL',3:'NEV',4:'SCC',5:'SEK'}
    pred_csv['Diagnostic'] = -1
    for i in range(len(pred_csv)):
        pred_csv.iloc[i,2] = record[pred_csv.iloc[i,0]]
        
    pred_csv_path = os.path.join(_output_path,'predict_{}.csv'.format(_model_name))
    pred_csv.to_csv(pred_csv_path,index = False)

    # save model
    model_name = os.path.join(_output_path, str(_model_name) + ".pkl")
    torch.save(best_model.state_dict(), model_name)
