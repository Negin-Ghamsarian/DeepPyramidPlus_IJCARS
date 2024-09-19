#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 13:43:53 2022

@author: negin
"""
from __future__ import print_function
import random 
import argparse
import logging 
import os 
import sys 
import csv

import numpy as np 
import torch 
import torch.nn as nn 
from torch import optim 
from tqdm import tqdm 

from utils.eval_dice_IoU import eval_dice_IoU
from utils.save_metrics import save_metrics


from utils.dataset_PyTorch import BasicDataset as BasicDataset
from utils.dataset_PyTorch_CSV import BasicDataset as BasicDataset_CSV
from torch.utils.data import DataLoader

from torchvision import transforms
from utils.losses_binary_ReduceMean import DiceBCELoss
from utils.losses_MultiClass_ReduceMean import Dice_CELoss


import wandb
from utils.seed_initialization import seed_all, seed_worker

from utils.import_helper import import_config
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
import importlib

from utils.TrainUtils import create_directory, polynomial_LR





class printer(nn.Module):
        def __init__(self, global_dict=globals()):
            super(printer, self).__init__()
            self.global_dict = global_dict
            self.except_list = []
        def debug(self,expression):
            frame = sys._getframe(1)

            print(expression, '=', repr(eval(expression, frame.f_globals, frame.f_locals)))

        def namestr(self,obj, namespace):
            return [name for name in namespace if namespace[name] is obj]     
        
        def forward(self,x):
            for i in x:
                if i not in self.except_list:
                    name = self.namestr(i, globals())
                    if len(name)>1:
                        self.except_list.append(i)
                        for j in range(len(name)):
                            self.debug(name[j])
                    else:  
                        self.debug(name[0])


           


def train_net(net,
              epochs=30,
              batch_size=1,
              lr=0.001,
              device='cuda',
              save_cp=True
              ):

    TESTS_source = []
    TESTS_target = []

    if dataset_mode == 'folder':
        train_dataset = BasicDataset(dir_train_img, dir_train_mask)
        test_dataset = BasicDataset(dir_test_img, dir_test_mask, doTransform = False)

    elif dataset_mode == 'csv':   
        train_dataset = BasicDataset_CSV(train_IDs_CSV, num_classes = num_classes)
        test_dataset = BasicDataset_CSV(test_IDs_CSV, doTransform = False, num_classes = num_classes)
        sourceTest_dataset = BasicDataset_CSV(SourceTest_IDs_CSV, doTransform = False, num_classes = num_classes)

    n_train = len(train_dataset)
    if n_train%batch_size == 1:
        drop_last = True
    else:
        drop_last = False    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=drop_last)
    
    inference_step = np.floor(np.ceil(n_train/batch_size)/test_per_epoch)
    print(f'Inference Step:{inference_step}')

    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=False, drop_last=False)
    if dataset_mode == 'csv':
        n_test = len(test_dataset)
    else:
        n_test = 'N/A due to one domain'    

 
    SourceTest_loader = DataLoader(sourceTest_dataset, batch_size=1, shuffle=False, pin_memory=False, drop_last=False)
    n_SourceTest = len(sourceTest_dataset)





    
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Source test size:{n_SourceTest}
        Target test size:{n_test}
        Checkpoints:     {save_cp}
        Device:          {device.type}
    ''')

    optimizer = optim.SGD(net.module.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 2, gamma = 0.9)

    if num_classes == 1:
        criterion = DiceBCELoss()
    else:
        criterion = Dice_CELoss()  
    test_counter = 1
    for epoch in range(epochs):
        net.train()
        

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:

                imgs = batch['image']
                true_masks = batch['mask']
                
            
                
                assert imgs.shape[1] == net.module.n_channels, \
                    f'Network has been defined with {net.module.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                
                masks_pred = net(imgs)
                loss_main = criterion(masks_pred, true_masks)
                loss_wandb = loss_main.detach().item()
                
        
                loss = loss_main
                epoch_loss += loss.detach().item()

                

                pbar.set_postfix(**{'loss (batch)': loss.detach().item()})

                optimizer.zero_grad()
            
                (loss_main).backward()
                nn.utils.clip_grad_value_(net.module.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                
                #######################################################################
                #######################################################################
                if (global_step) % (inference_step) == 0: # Should be changed if the condition that the n_train%batch_size != 0
                    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
                    print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
                    print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
                    for tag, value in net.module.named_parameters():
                        tag = tag.replace('.', '/')
                        

                    val1, val2, val3, val4, val5, val6, val7, val8, inference_time = eval_dice_IoU(net, SourceTest_loader, device, test_counter, save_dir=save_test, num_classes = num_classes, save=False)
                    TESTS_source.append([val1, val2, val3, val4, val5, val6, val7, val8, inference_time, epoch_loss])
                    del val5, val6, val7, val8, inference_time
                    
                    print(f'Validation Dice:{val1}')
                    print(f'Validation IoU:{val3}')
                    

                    val1_target = 0
                    val3_target = 0
                    if dataset_mode == 'csv':

                        if 'Endovis' in project_name or 'Endometriosis' in project_name or 'Anatomy' in project_name or 'instrument_MultiClass' in project_name or 'Syn' in project_name:
                            print("one domain")
                        else:   

                            val1_target, val2_target, val3_target, val4_target, val5_target, val6_target, val7_target, val8_target, inference_time_target = eval_dice_IoU(net, test_loader, device, test_counter, save_dir=save_test, num_classes = num_classes, save=False)
                            TESTS_target.append([val1_target, val2_target, val3_target, val4_target, val5_target, val6_target, val7_target, val8_target, inference_time_target, epoch_loss])
                            del val5_target, val6_target, val7_target, val8_target, inference_time_target
                            
                            print(f'Source Validation Dice:{val1_target}')
                            print(f'Source Validation IoU:{val3_target}')
                    

                    test_counter = test_counter+1
                    
                    


                    logging.info('Validation Dice Coeff: {}'.format(val1))
                    logging.info('Validation IoU: {}'.format(val3))

                    if 'Anatomy' in project_name:
                        wandb.log({'Train_Loss': loss_wandb,
                            'SourceTest_Dice_Cornea': val2[0],
                            'SourceTest_IoU_Cornea': val4[0],
                            'SourceTest_Dice_Pupil': val2[1],
                            'SourceTest_IoU_Pupil': val4[1],
                            'SourceTest_Dice_Lens': val2[2],
                            'SourceTest_IoU_Lens': val4[2],
                            'SourceTest_Dice_Instruments': val2[3],
                            'SourceTest_IoU_Instruments': val4[3],
                            'SourceTest_Dice_avg': val1,
                            'SourceTest_IoU_avg': val3})

                    elif 'Syn_multi' in project_name:
                        wandb.log({'Train_Loss': loss_wandb,
                            'SourceTest_Dice_shaft': val2[0],
                            'SourceTest_IoU_shaft': val4[0],
                            'SourceTest_Dice_wrist': val2[1],
                            'SourceTest_IoU_wrist': val4[1],
                            'SourceTest_Dice_jaw': val2[2],
                            'SourceTest_IoU_jaw': val4[2],
                            'SourceTest_Dice_avg': val1,
                            'SourceTest_IoU_avg': val3})

                    
                    else:
                        wandb.log({'Train_Loss': loss_wandb,
                            'TargetTest_Dice': val1_target,
                            'TargetTest_IoU': val3_target,
                            'SourceTest_Dice': val1,
                            'SourceTest_IoU': val3})
                    
                    

        scheduler.step()
           
        if save_cp:
            if True:
                try:
                    os.mkdir(dir_checkpoint)
                    logging.info('Created checkpoint directory')
                except OSError:
                    pass
        if (epoch+1)%50 == 0:
            torch.save(net.module.state_dict(),
                    dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

        if (epoch+1)%30 == 0:
            torch.save(net.module.state_dict(),
                    dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')    
            
    val1, val2, val3, val4, val5, val6, val7, val8, inference_time = eval_dice_IoU(net, test_loader, device, test_counter, save_dir=save_test, num_classes = num_classes, save=False)
    save_metrics(TESTS_source, csv_name_source)
    save_metrics(TESTS_target, csv_name_target)
     

   


if __name__ == '__main__':
    args = parser.parse_args()
    config_file = args.config
    my_conf = importlib.import_module(config_file)
    Categories, Learning_Rates_init, epochs, batch_size, size,\
             Dataset_Path_Train, Dataset_Path_SemiTrain, Dataset_Path_Test,\
                  mask_folder, Results_path, Visualization_path,\
                 CSV_path, project_name, load, load_path, net_name,\
                      test_per_epoch, Checkpoint_path, Net1,\
                         hard_label_thr, ensemble_batch_size, SemiSup_initial_epoch,\
                             image_transforms, affine, affine_transforms, LW,\
                                 EMA_decay, Alpha, strategy, GCC, supervised_share, num_classes\
                     = import_config.execute(my_conf)

    print("inside main")
    print('Hello Ubelix')
    print(f'Cuda Availability: {torch.cuda.is_available()}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device : {device}')
    print(f'Cuda Device Number: {torch.cuda.current_device()}')
    print(f'Cuda Device Name: {torch.cuda.get_device_name(0)}')
    num_gpus = torch.cuda.device_count()
    print('num_gpus: ', num_gpus)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Using device {device}')

    printer1 = printer()       
    
    print('CONFIGS:________________________________________________________')
        
    printer1([Categories, Learning_Rates_init, epochs, batch_size, size,\
                Dataset_Path_Train, Dataset_Path_SemiTrain, Dataset_Path_Test,\
                    mask_folder, Results_path, Visualization_path,\
                    CSV_path, project_name, load, load_path, net_name,\
                        test_per_epoch, Checkpoint_path, Net1,\
                            hard_label_thr, ensemble_batch_size, SemiSup_initial_epoch,\
                                image_transforms, affine, affine_transforms, LW,\
                                    EMA_decay, Alpha, strategy, GCC, supervised_share, num_classes]) 
    
    
    try:
        for c in range(len(Categories)):      
            for LR in range(len(Learning_Rates_init)):

                print(f'Initializing the learning rate: {Learning_Rates_init[LR]}')

                wandb.init(project=project_name+'_'+str(Learning_Rates_init[LR])+'_'+str(batch_size), entity="negin_gh",
                name=net_name)
                wandb.config = {
                "learning_rate": Learning_Rates_init[LR],
                "epochs": epochs,
                "batch_size": batch_size,
                "net_name": net_name,
                "Dataset": "Fold"+str(c),

                }
                    

                if 'RETOUCH' in project_name:
                    if num_classes == 1:

                        dataset_mode = 'csv'


                        train_IDs_CSV = Dataset_Path_Train+str(Categories[c])+'.csv'    
                                        
                        test_IDs_CSV = Dataset_Path_Train+str(Categories[c])+'_test.csv'    

                        semi_train_IDs_CSV = Dataset_Path_Train+str(Categories[c])+'_SemiSup.csv'

                        SourceTest_IDs_CSV = Dataset_Path_Train+str(Categories[c])+'_SourceTest.csv' 

                    else:

                        dataset_mode = 'csv'


                        train_IDs_CSV = Dataset_Path_Train+str(Categories[c])+'_SourceFull.csv'    
                                        
                        test_IDs_CSV = Dataset_Path_Train+str(Categories[c])+'_TestFull.csv'    

                        semi_train_IDs_CSV = Dataset_Path_Train+str(Categories[c])+'_SemiSupFull.csv'

                        SourceTest_IDs_CSV = Dataset_Path_Train+str(Categories[c])+'_SourceTestFull.csv'     

                
                elif 'Cataract' in project_name or 'MRI' in project_name or 'Dataset_Cat3KVsCaDIS' in project_name:

                    dataset_mode = 'csv'


                    train_IDs_CSV = Dataset_Path_Train+str(Categories[c])+'.csv'    
                                       
                    test_IDs_CSV = Dataset_Path_Train+str(Categories[c])+'_test.csv'    

                    semi_train_IDs_CSV = Dataset_Path_Train+str(Categories[c])+'_SemiSup.csv'

                    SourceTest_IDs_CSV = Dataset_Path_Train+str(Categories[c])+'_SourceTest.csv' 

                elif 'Endovis' in project_name:

                    dataset_mode = 'csv'


                    train_IDs_CSV = Dataset_Path_Train+str(Categories[c])+'.csv'    
                                       
                    test_IDs_CSV = Dataset_Path_Train+str(Categories[c])+'_test.csv'    

                    semi_train_IDs_CSV = Dataset_Path_Train+str(Categories[c])+'_SemiSup.csv'
                    
                    SourceTest_IDs_CSV = Dataset_Path_Train+str(Categories[c])+'_test.csv'  

                elif 'Endometriosis' in project_name: 

                    dataset_mode = 'csv'


                    train_IDs_CSV = Dataset_Path_Train+str(Categories[c])+'.csv'    
                                       
                    test_IDs_CSV = Dataset_Path_Train+str(Categories[c])+'_test.csv'    

                    semi_train_IDs_CSV = Dataset_Path_Train+str(Categories[c])+'.csv'
                    
                    SourceTest_IDs_CSV = Dataset_Path_Train+str(Categories[c])+'_test.csv' 

                elif 'Anatomy' in project_name or 'instrument_MultiClass' in project_name or 'Syn' in project_name: 

                    dataset_mode = 'csv'


                    train_IDs_CSV = Dataset_Path_Train+str(Categories[c])+'_train.csv'    
                                       
                    test_IDs_CSV = Dataset_Path_Train+str(Categories[c])+'_test.csv'    

                    semi_train_IDs_CSV = Dataset_Path_Train+str(Categories[c])+'_train.csv'
                    
                    SourceTest_IDs_CSV = Dataset_Path_Train+str(Categories[c])+'_test.csv'     


                else:   #CATARACT

                    dataset_mode = 'folder'

                    dir_train_img = Dataset_Path_Train+str(Categories[c])+'/imgs'  
                    dir_train_mask = Dataset_Path_Train+str(Categories[c])+ mask_folder 
                    
                    dir_test_img = Dataset_Path_Test+'/imgs'
                    dir_test_mask = Dataset_Path_Test+mask_folder

                    dir_SemiTrain_img = Dataset_Path_SemiTrain+'/imgs'
                    dir_SemiTrain_mask = Dataset_Path_SemiTrain+mask_folder


                save_test = Results_path + Visualization_path + project_name + '_' + strategy +'_Thr_'+str(hard_label_thr)+'_'+net_name +"_GCC_"+ str(GCC) +'_init_epoch_'+str(SemiSup_initial_epoch)+'_'+str(Learning_Rates_init[LR])+'_'+str(batch_size)+'_'+str(Categories[c])+'_'+'Affine_'+str(affine)+'/'
                dir_checkpoint = Results_path + Checkpoint_path + project_name + '_'+ strategy + '_'+net_name +'_'+str(Learning_Rates_init[LR])+'_'+str(batch_size)+'_'+str(Categories[c])+'/'
                csv_name_source = Results_path + CSV_path + project_name + '_' + strategy +'_Thr_'+str(hard_label_thr)+'_'+net_name +"_GCC_"+ str(GCC) +'_init_epoch_'+str(SemiSup_initial_epoch)+'_'+str(Learning_Rates_init[LR])+'_'+str(batch_size)+'_'+str(Categories[c])+'_'+'Affine_'+str(affine)+'_source.csv'
                csv_name_target = Results_path + CSV_path + project_name + '_' + strategy +'_Thr_'+str(hard_label_thr)+'_'+net_name +"_GCC_"+ str(GCC) +'_init_epoch_'+str(SemiSup_initial_epoch)+'_'+str(Learning_Rates_init[LR])+'_'+str(batch_size)+'_'+str(Categories[c])+'_'+'Affine_'+str(affine)+'_target.csv'
                
            
                create_directory(Results_path + Visualization_path)
                create_directory(Results_path + Checkpoint_path)
                create_directory(Results_path + CSV_path)


                

                net = Net1(n_classes=num_classes, n_channels=3)
                net = torch.nn.parallel.DataParallel(net, device_ids=list(range(num_gpus)), dim=0)
                logging.info(f'Network:\n'
                             f'\t{net.module.n_channels} input channels\n'
                             f'\t{net.module.n_classes} output channels (classes)\n')


                net.to(device=device)
                
                train_net(net=net,
                          epochs=epochs,
                          batch_size=batch_size,
                          lr=Learning_Rates_init[LR],
                          device=device)
            
    except KeyboardInterrupt:
        torch.save(net.module.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)