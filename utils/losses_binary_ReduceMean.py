# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 16:39:03 2020

@author: Negin
"""



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function



# For FocalLoss:
ALPHA = 0.8
GAMMA = 2    

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        inputs = F.sigmoid(inputs)
        
        intersection = torch.sum(inputs * targets, dim=(1,2,3))
        total = torch.sum(inputs + targets, dim=(1,2,3))
        dice = (2*intersection + smooth)/(total + smooth)
        
        return 1 - torch.mean(dice)




class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        

        
        inputs = F.sigmoid(inputs)
    
        
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        
        intersection = torch.sum(inputs * targets, dim=(1,2,3))
        total = torch.sum(inputs + targets, dim=(1,2,3))
        dice = (2*intersection + smooth)/(total + smooth)
        
        Dice_BCE = 0.8*BCE - 0.2*torch.log(torch.mean(dice))
        
        return Dice_BCE
    
    
    
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth = 1):

        inputs = torch.sigmoid(inputs)
        
        intersection = torch.sum(inputs * targets, dim=(1,2,3))
        total = torch.sum(inputs + targets, dim=(1,2,3))
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
        
        
                
        return 1 - torch.mean(IoU) 

    
class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss    
    
    