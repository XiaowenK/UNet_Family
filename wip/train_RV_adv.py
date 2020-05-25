
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

from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensor
from PIL import Image
from apex import amp
from models import unet
from models import resunet
from models import rexunet
from models import attention_unet
from models import unet_pp
from models import att_unet
from models import model_discriminative



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global Variables
path_data = r"/home/kxw/Develop/0430_UNet_Family/database/Retina_Vessel_Segmentation/patch"
path_txt_train = r"/home/kxw/Develop/0430_UNet_Family/labels/RV/fold_1/train.txt"
path_txt_val = r"/home/kxw/Develop/0430_UNet_Family/labels/RV/fold_1/val.txt"

M_G = "RexUNet"    # UNet, ResUNet, RexUNet, Att_UNet, UNet_PP
num_class = 1
pretrained = True
H = 96
W = 96
bs = 128
max_epoch = 50
lr_g = 0.0001
lr_d = 0.00001
optm_G = "Adam"    # Adam or SGD
optm_D = "Adam"    # Adam or SGD

apex_on = True



T_raw_train = A.Compose([
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
        elif mode == "val":
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



def getCurrent(path_ckpt):
    list_epoch_all = os.listdir(path_ckpt)
    list_epoch = []
    for epoch in list_epoch_all:
        if epoch.endswith(".tar"):
            list_epoch.append(epoch)
    current_epoch = int(len(list_epoch)) + 1
    return current_epoch



def getModel(num_class, pretrained):
    if M_G == "UNet":
        model_g = unet.UNet(num_class=num_class).to(device)
    elif M_G == "ResUNet":
        model_g = resunet.ResUNet(num_class=num_class, pretrained=pretrained).to(device)
    elif M_G == "RexUNet":
        model_g = rexunet.RexUNet(num_class=num_class, pretrained=pretrained).to(device)
    elif M_G == "Att_UNet":
        model_g = att_unet.Att_UNet(num_class=num_class).to(device)
    elif M_G == "UNet_PP":
        model_g = unet_pp.UNet_PP(num_class=num_class).to(device)
    model_d = model_discriminative.Discriminator().to(device)
    return model_g, model_d



def setModel(model_g, model_d, mode, test_epoch=None):
    crn_eph = getCurrent(r"./checkpoints_adv/G")
    if crn_eph == 1:
        is_from_begin = True
    else:
        is_from_begin = False
    if mode == 'train':
        if not is_from_begin:
            model_ckpt_g = torch.load(r"./checkpoints_adv/G/{}.pth.tar".format(str(crn_eph-1)))
            model_g.load_state_dict(model_ckpt_g['state_dict'])
            model_ckpt_d = torch.load(r"./checkpoints_adv/D/{}.pth.tar".format(str(crn_eph-1)))
            model_d.load_state_dict(model_ckpt_d['state_dict'])
        model_g.train()
        model_d.train()
    elif mode == 'test':
        model_ckpt_g = torch.load(r"./checkpoints_adv/G/{}.pth.tar".format(str(test_epoch)))
        model_g.load_state_dict(model_ckpt_g['state_dict'])
        model_g.eval()
        for param in model_g.parameters():
            param.requires_grad = False
    elif mode == 'val':
        model_g.eval()
        # model_d.eval()
        for param_g in model_g.parameters():
            param_g.requires_grad = False
        # for param_d in model_d.parameters():
        #     param_d.requires_grad = False
    return model_g, model_d



def setOptimizer(model_g, model_d, lr_g, lr_d):
    # ---- G ----
    if optm_G == "SGD":
        optimizerG = optim.SGD(model_g.parameters(), lr=lr_g)
    elif optm_G == "Adam":
        optimizerG = optim.Adam(model_g.parameters(), lr=lr_g, betas=(0.5, 0.999))
    # ---- D ----
    if optm_D == "SGD":
        optimizerD = optim.SGD(model_d.parameters(), lr=lr_d)
    elif optm_D == "Adam":
        optimizerD = optim.Adam(model_d.parameters(), lr=lr_d, betas=(0.5, 0.999))
    return optimizerG, optimizerD



def saveCKPT(epoch, model_g, model_d, optim_g, optim_d):
    path_ckpt_g = r"./checkpoints_adv/G/{}.pth.tar".format(str(epoch))
    path_ckpt_d = r"./checkpoints_adv/D/{}.pth.tar".format(str(epoch))
    torch.save({"epoch":epoch, "state_dict":model_g.state_dict(), 
                "optimizer":optim_g.state_dict()}, path_ckpt_g)
    torch.save({"epoch":epoch, "state_dict":model_d.state_dict(), 
                "optimizer":optim_d.state_dict()}, path_ckpt_d)



def soft_dice_coef_loss(y_pred, y_true, smooth=1.0):

    y_pred_f = y_pred.contiguous().view(-1)
    y_true_f = y_true.contiguous().view(-1)

    intersection = (y_pred_f * y_true_f).sum()

    A_sum = torch.sum(y_pred_f * y_pred_f)
    B_sum = torch.sum(y_true_f * y_true_f)

    loss = 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))

    return loss



def setApex(model_g, model_d, optimG, optimD):
    amp.register_float_function(torch, 'sigmoid')
    amp.register_float_function(F, 'softmax')
    model_g, optimG = amp.initialize(model_g, optimG, opt_level="O1")
    model_d, optimD = amp.initialize(model_d, optimD, opt_level="O1")
    return model_g, model_d, optimG, optimD



def train():
    # ---- get current epoch ----
    current_epoch = getCurrent(r"./checkpoints_adv/G")
    # ---- model & optimizer initialization ----
    model_g, model_d = getModel(num_class, pretrained)
    optimizerG, optimizerD = setOptimizer(model_g, model_d, lr_g, lr_d)
    if apex_on:
        # model, optimizer = setApex(model, optimizer)
        #
        model_g, model_d, optimizerG, optimizerD = setApex(model_g, model_d, optimizerG, optimizerD)
    model_g = torch.nn.DataParallel(model_g)
    model_d = torch.nn.DataParallel(model_d)
    model_g, model_d = setModel(model_g, model_d, mode="train")

    for epoch in range(current_epoch, max_epoch+1):
        print("Running epoch {}".format(epoch))
        # ---- loading data ----
        dataset = SegDataset(path_data, path_txt_train, mode="train")
        data = DataLoader(dataset=dataset, batch_size=bs, shuffle=True, num_workers=12, pin_memory=True)
        # ---- init loss ----
        loss_d_sum = 0
        loss_g_sum = 0
        loss_dice = 0
        # ---- start training ----
        for step, batch_data in enumerate(tqdm(data)):
            #
            real_label = 1
            fake_label = 0
            # ---- Update D network: maximize log(D(x)) + log(1 - D(G(z))) ----
            # Train with all-real batch
            model_d.zero_grad()
            input_raw = batch_data["image"].to(device)
            input_raw_maskgt = torch.cat((batch_data["image"], torch.unsqueeze(batch_data["mask"],dim=1)), 1)
            label = torch.full((input_raw.size(0),), real_label, device=device)
            output = model_d(input_raw_maskgt).view(-1)
            errD_real = F.binary_cross_entropy(torch.sigmoid(output), label)
            if apex_on:
                with amp.scale_loss(errD_real, optimizerD) as scaled_errD_real:
                    scaled_errD_real.backward()
            else:
                errD_real.backward()
            # Train with all-fake batch
            mask_pd = model_g(input_raw)
            input_raw_maskpd = torch.cat((batch_data["image"], mask_pd.cpu()), 1)
            label.fill_(fake_label)
            output = model_d(input_raw_maskpd.detach()).view(-1)
            errD_fake = F.binary_cross_entropy(torch.sigmoid(output), label)
            if apex_on:
                with amp.scale_loss(errD_fake, optimizerD) as scaled_errD_fake:
                    scaled_errD_fake.backward()
            else:
                errD_fake.backward()

            if step % 10 == 0:
                optimizerD.step()

            errD = errD_real + errD_fake

            # ---- Update G network: maximize log(D(G(z))) ----
            model_g.zero_grad()
            label.fill_(real_label)
            output = model_d(input_raw_maskpd.detach()).view(-1)
            errG = F.binary_cross_entropy(torch.sigmoid(output), label)

            # ---- For Dice loss ----
            P_flat = torch.sigmoid(mask_pd).contiguous().view(-1).cuda()
            M_flat = batch_data["mask"].contiguous().view(-1).cuda()
            errG_dice = soft_dice_coef_loss(P_flat, M_flat).to(device)

            errG = errG + errG_dice

            if apex_on:
                with amp.scale_loss(errG, optimizerD) as scaled_errG:
                    scaled_errG.backward()
            else:
                errG.backward()
            optimizerG.step()

            loss_d_sum += errD
            loss_g_sum += errG
            loss_dice += errG_dice

        loss_d_avg = loss_d_sum / len(data)
        loss_g_avg = loss_g_sum / len(data)
        loss_dice_avg = loss_dice / len(data)

        # ---- saving ckpt ----
        saveCKPT(epoch, model_g, model_d, optimizerG, optimizerD)
        # ---- train loss & acc ----
        print('Train (loss_d: {:.4f}, loss_g: {:.4f}, loss_dice: {:.4f})'.format(loss_d_avg, loss_g_avg, loss_dice_avg))
        # ---- val loss & acc ----
        loss_val_avg = val(model_g, model_d)
        print('Val (loss_dice: {:.4f})'.format(loss_val_avg))
        # ---- printing ----
        print('-' * 50)



def val(model_g, model_d):

    model_g, model_d = setModel(model_g, model_d, mode="val")
    # ---- loading data ----
    dataset = SegDataset(path_data, path_txt_val, mode="val")
    data = DataLoader(dataset=dataset, batch_size=bs, num_workers=12, pin_memory=True)
    # ---- init loss ----
    loss_val_sum = 0
    loss_val = 0
    # ---- start validating ----
    for idx_batch, batch_data in enumerate(tqdm(data)):
        # ---- inputs & labels ----
        inputs = batch_data['image'].to(device, dtype=torch.float)
        labels = batch_data['mask'].to(device, dtype=torch.float)
        # ---- fp ----
        preds = model_g(inputs)
        preds = torch.sigmoid(preds)
        # ---- bp ----
        loss_val = soft_dice_coef_loss(preds, labels)
        loss_val_sum += loss_val.item()
    # ---- val loss ----
    loss_val_avg = loss_val_sum / len(data)
    # ---- reset the model's mode ----
    model_g.train()
    for param_g in model_g.parameters():
        param_g.requires_grad = True
    return loss_val_avg



if __name__ == "__main__":

    train()


