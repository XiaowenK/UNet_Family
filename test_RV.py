
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Author   : Xiaowen Ke
## Email    : xiaowen.herman@gmail.com
## Version  : v0.1.0
## Date     : 2020/04/30
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import cv2
import argparse
import numpy as np
import albumentations as A
import torch
import torchvision
import torch.nn as nn

from tqdm import tqdm
from albumentations.pytorch import ToTensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from models import unet
from models import att_unet
from models import unet_pp
from models import resunet
from models import resunext



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



parser = argparse.ArgumentParser(description='Testing on Retina Vessel Dataset.')
parser.add_argument('-data', default=r"./database/Retina_Vessel/organized/48x48/patch", type=str, metavar='DATA', help='path to dataset')
parser.add_argument('-txt', default=r"./labels/Retina_Vessel", type=str, metavar='TXT', help='path to txt files')
parser.add_argument('-num_class', default=1, type=int, metavar='NUM_CLASS', help='number of class')
parser.add_argument('-height', default=48, type=int, metavar='HEIGHT', help='height of input')
parser.add_argument('-width', default=48, type=int, metavar='WIDTH', help='width of input')
parser.add_argument('-bs', default=1, type=int, metavar='BATCH SIZE', help='batch size')
parser.add_argument('-fold', default=5, type=int, metavar='FOLD', help='k-fold CV')
parser.add_argument('arch', type=str, metavar='ARCH', help='model architecture')
parser.add_argument('pretrained', type=bool, metavar='PRETRAINED', help='if pretrained on ImageNet')

args = parser.parse_args()



class RetinaVesselDataset(Dataset):

    def __init__(self, path_data, path_txt, h, w, pretrained):
        self.h = h
        self.w = w
        self.T_raw, self.T_mask = self._getT(h, w, pretrained)
        self.list_path_raw = []
        self.list_path_mask = []
        fileDescriptor = open(path_txt, "r")
        line = True
        while line:
            line = fileDescriptor.readline()
            if line:
                lineItems = line.split()
                path_raw = r"{}/raw/{}".format(path_data, lineItems[0])
                path_mask = r"{}/mask/{}".format(path_data, lineItems[0])
                self.list_path_raw.append(path_raw)
                self.list_path_mask.append(path_mask)
        fileDescriptor.close()

    def __getitem__(self, idx):
        raw = cv2.imread(self.list_path_raw[idx], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.list_path_mask[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.w, self.h))
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        raw = self.T_raw(image=raw)['image']
        mask = self.T_mask(image=mask)['image']
        data_pair = {"image": raw, "mask": mask}
        return data_pair

    def __len__(self):
        return len(self.list_path_raw)

    def _getT(self, h, w, pretrained):
        if pretrained:
            T_raw = A.Compose([
                A.Resize(height=h, width=w, interpolation=1, always_apply=False, p=1),
                A.Normalize(max_pixel_value=255.0, always_apply=False, p=1.0),
                ToTensor()
                ])
        else:
            T_raw = A.Compose([
                A.Resize(height=h, width=w, interpolation=1, always_apply=False, p=1),
                ToTensor()
                ])
        T_mask = A.Compose([
            ToTensor()
            ])
        return T_raw, T_mask



def soft_dice_coef_loss(y_pred, y_true, smooth=1.0):
    y_pred_f = y_pred.contiguous().view(-1)
    y_true_f = y_true.contiguous().view(-1)
    intersection = (y_pred_f * y_true_f).sum()
    A_sum = torch.sum(y_pred_f * y_pred_f)
    B_sum = torch.sum(y_true_f * y_true_f)
    loss = 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))
    return loss



def dice_coef_loss(y_pred, y_true, smooth=1.0):
    y_pred_f = y_pred.contiguous().view(-1)
    y_true_f = y_true.contiguous().view(-1)
    intersection = (y_pred_f * y_true_f).sum()
    A_sum = torch.sum(y_pred_f)
    B_sum = torch.sum(y_true_f)
    loss = 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))
    return loss



def dice_coef(y_pred, y_true, smooth=1.0):
    y_pred_f = y_pred.contiguous().view(-1)
    y_true_f = y_true.contiguous().view(-1)
    intersection = (y_pred_f * y_true_f).sum()
    A_sum = torch.sum(y_pred_f)
    B_sum = torch.sum(y_true_f)
    dsc = (2. * intersection + smooth) / (A_sum + B_sum + smooth)
    return dsc



def calMeanStdDev(list_loss_fold):
    mean_numerator = 0
    for loss_i in list_loss_fold:
        mean_numerator += loss_i
    mean = mean_numerator / len(list_loss_fold)

    variance_numerator = 0
    for loss_i in list_loss_fold:
        variance_numerator += ((loss_i - mean) ** 2)
    std_dev = (variance_numerator / len(list_loss_fold)) ** 0.5
    return mean, std_dev



def test():

    args = parser.parse_args()

    # ---- start testing ----
    list_loss_fold = []
    list_dsc_fold = []
    for k in range(1, args.fold+1):
        print("========== Fold {} ==========".format(k))
        # ---- setting model ----
        if args.arch == "UNet":
            model = unet.UNet(args.num_class).to(device)
        elif args.arch == "Att_UNet":
            model = att_unet.Att_UNet(args.num_class).to(device)
        elif args.arch == "UNet_PP":
            model = unet_pp.UNet_PP(args.num_class).to(device)
        elif args.arch == "ResUNet50":
            model = resunet.ResUNet50(args.num_class, args.pretrained).to(device)
        elif args.arch == "ResUNet101":
            model = resunet.ResUNet101(args.num_class, args.pretrained).to(device)
        elif args.arch == "ResUNext101":
            model = resunext.ResUNext101(args.num_class, args.pretrained).to(device)

        model = torch.nn.DataParallel(model).to(device)

        model_ckpt = torch.load(r"./checkpoints/fold_{}.pth.tar".format(k))
        model.load_state_dict(model_ckpt)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        # ---- loading data ----
        path_txt_test = r"{}/fold_{}/test.txt".format(args.txt, k)
        dataset = RetinaVesselDataset(args.data, path_txt_test, args.height, args.width, args.pretrained)
        data = DataLoader(dataset=dataset, batch_size=args.bs, num_workers=12, pin_memory=True)
        # ---- init loss ----
        loss_sum = 0
        dsc_sum = 0
        # ---- start validating ----
        for idx_batch, batch_data in enumerate(tqdm(data)):
            # ---- inputs & masks ----
            inputs = batch_data['image'].to(device, dtype=torch.float)
            masks = batch_data['mask'].to(device, dtype=torch.float)
            # ---- fp ----
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)
            # ---- bp ----
            loss = soft_dice_coef_loss(outputs, masks)
            # loss = dice_coef_loss(outputs, masks)
            dsc = dice_coef(outputs, masks)
            loss_sum += loss.item()
            dsc_sum += dsc.item()
        # ---- average loss ----
        loss_test = loss_sum / len(data)
        list_loss_fold.append(loss_test)
        # ---- average DSC ----
        dsc_test = dsc_sum / len(data)
        list_dsc_fold.append(dsc_test)

    # ---- Dice Coefficient (DSC) and Standard Deviation ----
    mean_loss, std_dev_loss = calMeanStdDev(list_loss_fold)
    mean_dsc, std_dev_dsc = calMeanStdDev(list_dsc_fold)
    print("Loss: Mean({:.3f}), Standard Deviation({:.3f}) ".format(mean_loss, std_dev_loss))
    print("DSC: Mean({:.3f}), Standard Deviation({:.3f}) ".format(mean_dsc, std_dev_dsc))
    # ---- mIoU (Jaccard Index) ----
    # ---- PA ----
    # ---- ROC ----



if __name__ == "__main__":

    test()


