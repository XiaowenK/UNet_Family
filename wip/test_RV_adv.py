
__author__ = "Xiaowen"

##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Xiaowen Ke
## Shenzhen Smart Imaging Healthcare Co.,Ltd.
## Email: xiaowen.herman@gmail.com
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import cv2
import numpy as np
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
import torchvision.transforms as transforms
import shutil

from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensor
from PIL import Image
from models import unet
from models import resunet
from models import rexunet
from models import attention_unet
from models import unet_pp
from models import att_unet



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global Variables
path_data = r"/home/kxw/Develop/0430_UNet_Family/database/Retina_Vessel_Segmentation/patch"
path_txt_test = r"/home/kxw/Develop/0430_UNet_Family/labels/RV/fold_1/test.txt"
path_results_preds = r"./results/preds"
path_results_masks = r"./results/masks"

M = "RexUNet"    # UNet, ResUNet, RexUNet, Att_UNet, UNet_PP
num_class = 1
H = 96
W = 96
bs = 1
pretrained = True
epoch_test = 5



T_raw_val = A.Compose([
    A.Resize(
        height=H,
        width=W,
        interpolation=1,
        always_apply=False,
        p=1
        ),
    A.Normalize(
        max_pixel_value=255.0,
        always_apply=False,
        p=1.0
        ),
    ToTensor()
    ])



T_mask = A.Compose([
    ToTensor()
    ])



class SegDataset(Dataset):

    def __init__(self, path_data, path_txt, mode):
        if mode == "train":
            self.T = T_raw_train
        elif mode == "val" or mode == "test":
            self.T = T_raw_val
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
        mask = cv2.resize(mask, (W, H))
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        raw = self.T(image=raw)['image']
        mask = T_mask(image=mask)['image']
        data_pair = {"image": raw, "mask": mask}
        return data_pair

    def __len__(self):
        return len(self.list_path_raw)



def getModel(num_class, pretrained):
    if M == "UNet":
        model = unet.UNet(num_class=num_class).to(device)
    elif M == "ResUNet":
        model = resunet.ResUNet(num_class=num_class, pretrained=pretrained).to(device)
    elif M == "RexUNet":
        model = rexunet.RexUNet(num_class=num_class, pretrained=pretrained).to(device)
    elif M == "Att_UNet":
        # model = attention_unet.Att_UNet(num_class=num_class).to(device)
        model = att_unet.Att_UNet(num_class=num_class).to(device)
    elif M == "UNet_PP":
        model = unet_pp.UNet_PP(num_class=num_class).to(device)
    model = torch.nn.DataParallel(model).to(device)
    return model



def setModel(model, mode, test_epoch=None):
    model_ckpt = torch.load(r"./checkpoints_adv/G/"+str(test_epoch)+".pth.tar")
    model.load_state_dict(model_ckpt['state_dict'])
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model



def soft_dice_coef_loss(y_pred, y_true, smooth=1.0):

    # y_pred = torch.sigmoid(y_pred)
    y_pred_f = y_pred.contiguous().view(-1)
    y_true_f = y_true.contiguous().view(-1)

    intersection = (y_pred_f * y_true_f).sum()

    A_sum = torch.sum(y_pred_f * y_pred_f)
    B_sum = torch.sum(y_true_f * y_true_f)

    loss = 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))

    return loss



def save_pred(pred, labels, dataset, idx_batch):

    #
    raw = cv2.imread(dataset.list_path_raw[idx_batch])
    h = raw.shape[0]
    w = raw.shape[1]
    name = dataset.list_path_raw[idx_batch].split("/")[-1]

    #
    pred = pred.cpu().data.numpy()
    pred = pred[0, 0, :, :] * 255
    pred = pred.T
    pred = cv2.resize(pred, (w, h))
    cv2.imwrite(r"{}/{}".format(path_results_preds, name), np.uint8(pred))

    #
    labels = labels.cpu().data.numpy()
    labels = labels[0, :, :] * 255
    labels = labels.T
    labels = cv2.resize(labels, (w, h))
    cv2.imwrite(r"{}/{}".format(path_results_masks, name), np.uint8(labels))



def test():

    if os.path.exists(path_results_preds):
        shutil.rmtree(path_results_preds)
    os.mkdir(path_results_preds)
    if os.path.exists(path_results_masks):
        shutil.rmtree(path_results_masks)
    os.mkdir(path_results_masks)

    model = getModel(num_class, pretrained)
    model = setModel(model=model, mode="test", test_epoch=epoch_test)
    # ---- loading data ----
    dataset = SegDataset(path_data, path_txt_test, mode="test")
    data = DataLoader(dataset=dataset, batch_size=bs, num_workers=12, pin_memory=True)
    # ---- init loss ----
    loss_test_sum = 0
    loss_test = 0
    # ---- start validating ----
    for idx_batch, batch_data in enumerate(tqdm(data)):
        # ---- inputs & labels ----
        inputs = batch_data['image'].to(device, dtype=torch.float)
        labels = batch_data['mask'].to(device, dtype=torch.float)
        # ---- fp ----
        preds = model(inputs)
        preds = torch.sigmoid(preds)
        # ---- draw contour ----
        save_pred(preds, labels, dataset, idx_batch)
        # ---- bp ----
        loss_test = soft_dice_coef_loss(preds, labels)
        loss_test_sum += loss_test.item()
    # ---- val loss & acc ----
    loss_test_avg = loss_test_sum / len(data)
    print("Testing at epoch {}, dice loss is {}.".format(epoch_test, round(loss_test_avg, 5)))



if __name__ == "__main__":

    test()


