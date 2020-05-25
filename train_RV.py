
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
import datetime
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from apex import amp
from albumentations.pytorch import ToTensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import unet
from models import att_unet
from models import unet_pp
from models import resunet
from models import resunext



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



parser = argparse.ArgumentParser(description='Training on Retina Vessel Dataset.')
parser.add_argument('-data', default=r"./database/Retina_Vessel/organized/48x48/patch", type=str, metavar='DATA', help='path to dataset')
parser.add_argument('-txt', default=r"./labels/Retina_Vessel", type=str, metavar='TXT', help='path to txt files')
parser.add_argument('-num_class', default=1, type=int, metavar='NUM_CLASS', help='number of class, output channels')
parser.add_argument('-height', default=48, type=int, metavar='HEIGHT', help='height of input')
parser.add_argument('-width', default=48, type=int, metavar='WIDTH', help='width of input')
parser.add_argument('-bs', default=64, type=int, metavar='BATCH SIZE', help='batch size')
parser.add_argument('-epoch', default=60, type=int, metavar='EPOCH', help='max training epoch')
parser.add_argument('-lr', default=0.0001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('-optm', default="Adam", type=str, metavar='OPTM', help='optimizer')
parser.add_argument('-apex', default=True, type=bool, metavar='APEX', help='nvidia apex module')
parser.add_argument('-fold', default=5, type=int, metavar='FOLD', help='k-fold CV')
parser.add_argument('arch', type=str, metavar='ARCH', help='model architecture')
parser.add_argument('pretrained', type=bool, metavar='PRETRAINED', help='if pretrained on ImageNet')



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



def train():

    args = parser.parse_args()

    for k in range(1, args.fold+1):
        print("========== Fold {} ==========".format(k))
        # ---- setting model & optimizer ----
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

        if args.optm == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=args.lr)
        elif args.optm == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True, eps=1e-6)

        if args.apex:
            amp.register_float_function(torch, 'sigmoid')
            amp.register_float_function(F, 'softmax')
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

        model = torch.nn.DataParallel(model)
        model.train()

        # ---- start training ----
        path_txt_train = r"{}/fold_{}/train.txt".format(args.txt, k)
        loss_min = 100
        for epoch in range(1, args.epoch+1):
            # ---- timer ----
            starttime = datetime.datetime.now()
            # ---- loading data ----
            dataset = RetinaVesselDataset(args.data, path_txt_train, args.height, args.width, args.pretrained)
            data = DataLoader(dataset=dataset, batch_size=args.bs, shuffle=True, num_workers=12, pin_memory=True)
            # ---- loop for all train data ----
            loss_train_sum = 0
            for step, batch_data in enumerate(data):
                # ---- inputs & masks ----
                inputs = batch_data['image'].to(device, dtype=torch.float)
                masks = batch_data['mask'].to(device, dtype=torch.float)
                # ---- fp ----
                outputs = model(inputs)
                outputs = torch.sigmoid(outputs)
                # ---- bp ----
                loss = soft_dice_coef_loss(outputs, masks)
                # loss = dice_coef_loss(outputs, masks)
                loss_train_sum += loss.item()
                if args.apex:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            # ---- train loss ----
            loss_train = loss_train_sum / len(data)
            # ---- validation ----
            loss_val = validation(model, k, args)
            scheduler.step(loss_val)
            # ---- saving ckpt ----
            if loss_val < loss_min:
                loss_min = loss_val
                print("Best model saved at epoch {}!".format(epoch))
                torch.save(model.state_dict(), r"./checkpoints/fold_{}.pth.tar".format(k))
            # ---- timer ----
            endtime = datetime.datetime.now()
            elapsed = (endtime - starttime).seconds
            # ---- printing ----
            print("fold #{}, epoch #{}, train: {:.4f}, val: {:.4f}, elapsed: {}s"
                  .format(k, epoch, loss_train, loss_val, elapsed))
            print('-' * 60)



def validation(model, k, args):

    # ---- setting model ----
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    # ---- loading data ----
    path_txt_val = r"{}/fold_{}/val.txt".format(args.txt, k)
    dataset = RetinaVesselDataset(args.data, path_txt_val, args.height, args.width, args.pretrained)
    data = DataLoader(dataset=dataset, batch_size=args.bs, num_workers=12, pin_memory=True)
    # ---- loop for all val data ----
    loss_val_sum = 0
    for idx_batch, batch_data in enumerate(data):
        # ---- inputs & labels ----
        inputs = batch_data['image'].to(device, dtype=torch.float)
        masks = batch_data['mask'].to(device, dtype=torch.float)
        # ---- fp ----
        outputs = model(inputs)
        outputs = torch.sigmoid(outputs)
        # ---- bp ----
        loss = soft_dice_coef_loss(outputs, masks)
        # loss = dice_coef_loss(outputs, masks)
        loss_val_sum += loss.item()
    # ---- val loss ----
    loss_val = loss_val_sum / len(data)
    # ---- reset the model's mode ----
    model.train()
    for param in model.parameters():
        param.requires_grad = True
    return loss_val



if __name__ == "__main__":

    train()


