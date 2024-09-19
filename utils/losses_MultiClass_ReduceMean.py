# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 16:39:03 2020

@author: Negin
"""



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Optional
from .one_hot import one_hot


# For FocalLoss:
ALPHA = 0.8
GAMMA = 2  



class Dice_CELoss(nn.Module):
    def __init__(self, ignore_first=True, apply_softmax=True):
        super(Dice_CELoss, self).__init__()
        self.eps = 1
        self.ignore_first = ignore_first
        self.apply_softmax = apply_softmax
        self.CE = nn.CrossEntropyLoss()

    def forward(self, input, target):
        '''
        input: torch.Tensor. Predicted tensor. Shape: [BxCxHxW]. Before softmax,
        it includes the raw outputs of the network. Softmax converts them into
        probabilities.
        target: torch.Tensor. Ground truth tensor. Shape: [BxHxW]
        target_one_hot: torch.Tensor. Conversion of target into a one hot tensor
        '''

        CE_loss = self.CE(input, target)

        if self.apply_softmax:
            input = input.softmax(dim=1)

        target_one_hot = F.one_hot(target.long(), num_classes=input.shape[1]).permute(0,3,1,2)

        if self.ignore_first:
            input = input[:, 1:]
            target_one_hot = target_one_hot[:, 1:]


        intersection= torch.sum(target_one_hot*input,dim=(2,3))
        cardinality= torch.sum(target_one_hot+input,dim=(2,3))

         
        dice=(2*intersection+self.eps)/(cardinality+self.eps)

        dice = torch.mean(torch.sum(dice, dim=1)/input.size(dim=1))

        loss = 0.8*CE_loss-0.2*torch.log(dice)
        return loss

