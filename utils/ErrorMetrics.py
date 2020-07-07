"""
This script is modified from the work of Abdelrahman Eldesokey.
Find more details from https://github.com/abdo-eldesokey/nconv
"""

########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class iMAE(nn.Module):
    def __init__(self):
        super(iMAE, self).__init__()

    def forward(self, outputs, target, *args):
        outputs = outputs / 1000.
        target = target / 1000.
        outputs[outputs == 0] = -1
        target[target == 0] = -1
        outputs = 1. / outputs
        target = 1. / target
        outputs[outputs == -1] = 0
        target[target == -1] = 0
        val_pixels = (target > 0).float().cuda()
        err = torch.abs(target * val_pixels - outputs * val_pixels)
        loss = torch.sum(err.view(err.size(0), 1, -1), -1, keepdim=True)
        cnt = torch.sum(val_pixels.view(val_pixels.size(0), 1, -1), -1, keepdim=True)
        return torch.mean(loss / cnt)

class MAE(nn.Module):
    def __init__(self):
        super(MAE, self).__init__()

    def forward(self, outputs, target, *args):
        val_pixels = ((target > 0).float()*(outputs>0).float()).cuda()
        err = torch.abs(target * val_pixels - outputs * val_pixels)
        loss = torch.sum(err.view(err.size(0), 1, -1), -1, keepdim=True)
        cnt = torch.sum(val_pixels.view(val_pixels.size(0), 1, -1), -1, keepdim=True)
        return torch.mean(loss / cnt) * 1000

class RMSE(nn.Module):
    def __init__(self):
        super(RMSE, self).__init__()

    def forward(self, outputs, target, *args):
        val_pixels = ((target > 0).float()*(outputs>0).float()).cuda()
        err = (target * val_pixels - outputs * val_pixels) ** 2
        loss = torch.sum(err.view(err.size(0), 1, -1), -1, keepdim=True)
        cnt = torch.sum(val_pixels.view(val_pixels.size(0), 1, -1), -1, keepdim=True)
        #return torch.sqrt(torch.mean(loss / cnt))
        return torch.mean(torch.sqrt(loss / cnt))  * 1000

class iRMSE(nn.Module):
    def __init__(self):
        super(iRMSE, self).__init__()

    def forward(self, outputs, target, *args):

        outputs = outputs / 1000.
        target = target / 1000.
        outputs[outputs==0] = -1
        target[target==0] = -1
        outputs = 1. / outputs
        target = 1. / target
        outputs[outputs == -1] = 0
        target[target == -1] = 0
        val_pixels = (target > 0).float().cuda()
        err = (target * val_pixels - outputs * val_pixels) ** 2
        loss = torch.sum(err.view(err.size(0), 1, -1), -1, keepdim=True)
        cnt = torch.sum(val_pixels.view(val_pixels.size(0), 1, -1), -1, keepdim=True)
        #return torch.sqrt(torch.mean(loss / cnt))
        return torch.mean(torch.sqrt(loss / cnt))
