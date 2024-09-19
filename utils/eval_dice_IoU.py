# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 13:11:39 2020

@author: Negin
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms
from .Metrics_ReduceMean import Dice_binary, IoU_binary, Dice_MultiClass, IoU_MultiClass


from PIL import Image
import numpy as np
import os
import timeit


def eval_dice_IoU(net, loader, device, test_counter, save_dir, num_classes = 1, save=True, Pyramid_Loss = False):
    """Evaluation without the densecrf with the dice coefficient"""

    if num_classes == 1:

        dice_coeff = Dice_binary()
        jaccard_index = IoU_binary()

    else:

        dice_coeff = Dice_MultiClass(num_classes=num_classes)
        jaccard_index = IoU_MultiClass(num_classes=num_classes)


    net.eval()
    
    #mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    dice = []
    IoU = []
    Inference_time = []

    try:
       os.mkdir(save_dir)
    except OSError:
       pass

    test_batch = False
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            test_batch = True

            imgs, true_masks, name = batch['image'], batch['mask'], batch['name']


            start = timeit.default_timer()
            with torch.no_grad():
                if Pyramid_Loss:
                    mask_pred, NI1, NI2, NI3 = net(imgs)
                    del  NI1, NI2, NI3
                else:
                    mask_pred = net(imgs)
            stop = timeit.default_timer()
            Inference_time.append(stop - start)



            if num_classes == 1:    

                val_Dice = dice_coeff(mask_pred, true_masks)
                val_IoU = jaccard_index(mask_pred, true_masks)

                dice.append(val_Dice)
                IoU.append(val_IoU)

            else:

                dice_coeff(mask_pred, true_masks)
                jaccard_index(mask_pred, true_masks)

                val_Dice, val_Dice_avg = dice_coeff.evaluate()
                val_IoU, val_IoU_avg = jaccard_index.evaluate()



            pbar.set_postfix(**{'val_Dice (batch)': val_Dice})
            pbar.set_postfix(**{'val_IoU (batch)': val_IoU})

            if save: #test_counter//10 == test_counter/10:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                probs = pred.squeeze(0)


                tf = transforms.Compose(
                  [
                   transforms.ToPILImage(),
                   transforms.Resize(512),
                   transforms.ToTensor()
                  ]
                  )

                probs = tf(probs.cpu())
                full_mask = Image.fromarray(probs.squeeze().cpu().numpy()*255).convert('RGB')
                full_mask = full_mask.save(save_dir+ str(name[0]) + '_' + str(test_counter) + '.png')

            pbar.update()
    if test_batch:
        del imgs, true_masks, name, mask_pred
    else:    
        return 0, 0, 0, 0, 0, 0, 0, 0, 0
    net.train()
    if num_classes == 1:    
        return sum(dice)/n_val, np.std(dice), sum(IoU)/n_val, np.std(IoU), min(dice), min(IoU), max(dice), max(IoU), sum(Inference_time)/n_val
    else:
        return val_Dice_avg, val_Dice, val_IoU_avg, val_IoU,  'N/A',  'N/A',   'N/A',   'N/A',  sum(Inference_time)/n_val        