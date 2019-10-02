import os, tqdm
import cv2 as cv
import numpy as np
import nibabel as nib

def check_data(img):
    output = None
    num_use = img.shape[-1]//6*6
    num_trash = img.shape[-1]%6
    num_bot = np.ceil(num_trash/2).astype(np.int8)
    num_top = np.floor(num_trash/2).astype(np.int8)
    check = 2*num_bot+num_top
    if check == 0:
        output = img
    elif check == 2:
        output = img[..., num_bot:]
    else :    
        output = img[...,num_bot:-(num_top)]
    return output

def data_loader(path):
    output = []
    scan_list = sorted(os.listdir(path))[1:]
    for scan in scan_list:
        dante_path = os.path.join(path, scan, 'T1SPACE09mmISOPOSTwDANTE')
        img_name = [i for i in os.listdir(dante_path) if '.nii' in i and '_rsl' not in i][0]
        #print(img_name)
        img = nib.load(os.path.join(dante_path, img_name))
        img = img.get_data()
        output.append(check_data(img))
    return output