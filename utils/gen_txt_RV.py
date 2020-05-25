 
__author__ = "Xiaowen"

import os
import cv2
import numpy as np
import shutil
import random



def gen_txt(path_data, path_txt_dst):

    list_files = os.listdir(path_data)
    list_patients = []
    for file in list_files:
        p = file.split("_")[0]
        if p not in list_patients:
            list_patients.append(p)

    random.shuffle(list_patients)

    list_fold = [0, 12, 24, 36, 48]

    n = 1
    for start_itm in list_fold:
        list_txt_train = []
        list_txt_val = []
        list_txt_test = []
        list_patient_val_test = list_patients[start_itm:(start_itm+12)]
        list_patient_train = []
        for p in list_patients:
            if p not in list_patient_val_test:
                list_patient_train.append(p)
        list_patient_val = list_patient_val_test[0:6]
        list_patient_test = list_patient_val_test[6:]
        # ---- train.txt ----
        for p in list_patient_train:
            for tif in list_files:
                if tif.split("_")[0] == p:
                    list_txt_train.append(tif)
        # ---- val.txt ----
        for p in list_patient_val:
            for tif in list_files:
                if tif.split("_")[0] == p:
                    list_txt_val.append(tif)
        # ---- test.txt ----
        for p in list_patient_test:
            for tif in list_files:
                if tif.split("_")[0] == p:
                    list_txt_test.append(tif)

        np.savetxt(r"{}/fold_{}/train.txt".format(path_txt_dst, n), np.reshape(list_txt_train, -1), delimiter=',', fmt='%5s')
        np.savetxt(r"{}/fold_{}/val.txt".format(path_txt_dst, n), np.reshape(list_txt_val, -1), delimiter=',', fmt='%5s')
        np.savetxt(r"{}/fold_{}/test.txt".format(path_txt_dst, n), np.reshape(list_txt_test, -1), delimiter=',', fmt='%5s')

        n += 1



if __name__ == "__main__":

    path_data = r"./database/Retina_Vessel/organized/48x48/whole/mask"
    path_txt_dst = r"./labels/Retina_Vessel"
    gen_txt(path_data, path_txt_dst)


