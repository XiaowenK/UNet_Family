 
__author__ = "Xiaowen"

import os
import cv2
import numpy as np
import shutil
import random

from PIL import Image



# ---- step #1: STARE, unzip data ----
# unzip the following .tar files to corresponding folders:
# --- STARE/                           --- STARE/
#       |--- stare-images.tar   --->         |--- stare-images/
#       |--- labels-ah.tar                   |--- labels-ah/
#       |--- labels-vk.tar                   |--- labels-vk/



# ---- step #2: STARE, convert .ppm files to .png and rename them to format <0001_STARE.png> ----
def STARE_ppm2png_rename(path_data):
    list_ppm = os.listdir(path_data)
    for ppm in list_ppm:
        if ppm.endswith(".ppm"):
            png_new = "{}_STARE.png".format(ppm.split(".")[0].split("m")[1])
            im = Image.open(r"{}/{}".format(path_data, ppm))
            im.save(r"{}/{}".format(path_data, png_new)) 
            os.remove(r"{}/{}".format(path_data, ppm))



# ---- step #3: DRIVE, unzip data ----
# --- DRIVE/                       --- DRIVE/
#       |--- datasets.zip   --->         |--- datasets.zip
#                                        |--- training/
#                                        |--- test/                                             



# ---- step #4: DRIVE, convert .gif mask files to .png and rename them to format <01_DRIVE.png> ----
def DRIVE_gif2png_rename(path_data):
    list_gif = os.listdir(path_data)
    for gif in list_gif:
        if gif.endswith(".gif"):
            png_new = "{}_DIRVE.png".format(gif.split(".")[0].split("_")[0])
            im = Image.open(r"{}/{}".format(path_data, gif))
            im.save(r"{}/{}".format(path_data, png_new))
            os.remove(r"{}/{}".format(path_data, gif))



# ---- step #5: DRIVE, convert .tif image files to .png and rename them to format <01_DRIVE.png> ----
def DRIVE_tif2png_rename(path_data):
    list_tif = os.listdir(path_data)
    for tif in list_tif:
        if tif.endswith(".tif"):
            png_new = "{}_DIRVE.png".format(tif.split(".")[0].split("_")[0])
            im = Image.open(r"{}/{}".format(path_data, tif))
            im.save(r"{}/{}".format(path_data, png_new))
            os.remove(r"{}/{}".format(path_data, tif))



# ---- step #6: copy all images to <organized folder> ----
# before_organized/STARE/stare-images/..
# before_organized/DRIVE/training/images/..    --->    organized/48x48/whole/raw
# before_organized/DRIVE/test/images/..



# ---- step #6: copy all masks to <organized folder> ----
# before_organized/STARE/labels-ah/..
# before_organized/DRIVE/training/1st_manual/..    --->    organized/48x48/whole/mask
# before_organized/DRIVE/test/1st_manual/..



# ---- step #8: crop and split the whole image to small patch 48 x 48 ----
def whole2patch(path_raw_src, path_mask_src, path_raw_dst, path_mask_dst):
    list_name_whole = os.listdir(path_raw_src)
    for name_whole in list_name_whole:
        img = cv2.imread(r"{}/{}".format(path_raw_src, name_whole), 1)
        mask = cv2.imread(r"{}/{}".format(path_mask_src, name_whole), 0)
        d = name_whole.split(".")[0].split("_")[1]
        size = "{}/{}".format(img.shape[0], img.shape[1])

        if d == "STARE":
            img_new = img[10:(10+576), 14:(14+672), :]
            mask_new = mask[10:(10+576), 14:(14+672)]
        elif d == "DRIVE":
            img_new = img[4:(4+576), 18:(18+528), :]
            mask_new = mask[4:(4+576), 18:(18+528)]

        # 48x48
        for idx_0 in range(0, int(img_new.shape[0]/48)):
            y_start = idx_0 * 48
            y_end = idx_0 * 48 + 48
            for idx_1 in range(0, int(img_new.shape[1]/48)):
                x_start = idx_1 * 48
                x_end = idx_1 * 48 + 48
                patch_img = img_new[y_start:y_end, x_start:x_end, :]
                patch_mask = mask_new[y_start:y_end, x_start:x_end]
                name_patch = name_whole.split(".")[0]+"_"+str(idx_0)+str(idx_1)+".png"
                cv2.imwrite(r"{}/{}".format(path_raw_dst, name_patch), np.uint8(patch_img))
                cv2.imwrite(r"{}/{}".format(path_mask_dst, name_patch), np.uint8(patch_mask)) 



# def whole2patch_v2(path_raw_src, path_mask_src, path_raw_dst, path_mask_dst):
#     list_name_whole = os.listdir(path_raw_src)
#     for name_whole in list_name_whole:
#         img = cv2.imread(r"{}/{}".format(path_raw_src, name_whole), 1)
#         mask = cv2.imread(r"{}/{}".format(path_mask_src, name_whole), 0)
#         d = name_whole.split(".")[0].split("_")[1]
#         size = "{}/{}".format(img.shape[0], img.shape[1])    # h/w

#         if d == "STARE":
#             img_new = img[10:(10+576), 14:(14+672), :]
#             mask_new = mask[10:(10+576), 14:(14+672)]
#         elif d == "DRIVE":
#             img_t = img[4:(4+576), :, :]
#             mask_t = mask[4:(4+576), :]
#             # padding
#             img_new = np.pad(img_t, ((0,0),(6,5),(0,0)), 'constant', constant_values=(0,0))
#             mask_new = np.pad(mask_t, ((0,0),(6,5)), 'constant', constant_values=(0,0))

#         # 96x96
#         for idx_0 in range(0, int(img_new.shape[0]/96)):
#             y_start = idx_0 * 96
#             y_end = idx_0 * 96 + 96
#             for idx_1 in range(0, int(img_new.shape[1]/96)):
#                 x_start = idx_1 * 96
#                 x_end = idx_1 * 96 + 96
#                 patch_img = img_new[y_start:y_end, x_start:x_end, :]
#                 patch_mask = mask_new[y_start:y_end, x_start:x_end]
#                 name_patch = name_whole.split(".")[0]+"_"+str(idx_0)+str(idx_1)+".png"
#                 cv2.imwrite(r"{}/{}".format(path_raw_dst, name_patch), np.uint8(patch_img))
#                 cv2.imwrite(r"{}/{}".format(path_mask_dst, name_patch), np.uint8(patch_mask))




if __name__ == "__main__":

    path_data = r"./database/Retina_Vessel/before_organized/STARE/stare-images"
    # path_data = r"./database/Retina_Vessel/before_organized/STARE/labels-ah"
    STARE_ppm2png_rename(path_data)


    path_data = r"./database/Retina_Vessel/before_organized/DRIVE/training/1st_manual"
    # path_data = r"./database/Retina_Vessel/before_organized/DRIVE/test/1st_manual"
    DRIVE_gif2png_rename(path_data)


    path_data = r"./database/Retina_Vessel/before_organized/DRIVE/training/images"
    # path_data = r"./database/Retina_Vessel/before_organized/DRIVE/test/images"
    DRIVE_tif2png_rename(path_data)


    path_raw_src = r"./database/Retina_Vessel/organized/48x48/whole/raw"
    path_mask_src = r"./database/Retina_Vessel/organized/48x48/whole/mask"
    path_raw_dst = r"./database/Retina_Vessel/organized/48x48/patch/raw"
    path_mask_dst = r"./database/Retina_Vessel/organized/48x48/patch/mask"
    whole2patch(path_raw_src, path_mask_src, path_raw_dst, path_mask_dst)


    # path_raw_src = r"./database/Retina_Vessel/organized/48x48/whole/raw"
    # path_mask_src = r"./database/Retina_Vessel/organized/48x48/whole/mask"
    # path_raw_dst = r"./database/Retina_Vessel/organized/48x48/patch/raw"
    # path_mask_dst = r"./database/Retina_Vessel/organized/48x48/patch/mask"
    # whole2patch_v2(path_raw_src, path_mask_src, path_raw_dst, path_mask_dst)


