import os, argparse
import cv2 as cv
import numpy as np
import pandas as pd
import nibabel as nib

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", dest="input_dir", type=str, help="Path of Input")
parser.add_argument("--val_num", dest="val_num", type=int, help="Validation Data Number", default=0)

args = parser.parse_args()

dict_args = vars(args)

for i in dict_args.keys():
    assert dict_args[i]!=None, '"%s" key is None Value!'%i
print("\n================ Training Options ================")
print("Input dir : ", args.input_dir)
print("Validation Number : ", args.val_num)
print("====================================================\n")

data_dir = args.input_dir
data_lists = sorted(os.listdir(data_dir))
si15_lists = [i for i in data_lists if '_siemens_15' in i]

print(len(si15_lists))

val_idx =[0, 18, 36]
test_idx = 54

if args.val_num == 3:
    data = []
    label = []
    for name in test_lists:
        tmp_path = os.path.join(data_dir, name)
        head = nib.load(tmp_path)
        img = head.get_data().astype(np.uint16)[..., 2:-2]
        tmp = np.array(np.dsplit(img, img.shape[-1]//6))
        tmp = tmp.mean(axis=-1)
        tmp = np.transpose(tmp, [1,2,0])
        label.append(img)
        data.append(tmp)
    label = np.array(label)[..., np.newaxis]
    data = np.array(data)[..., np.newaxis]
    print(data.shape, label.shape)

    np.save('./data/vox_test_img', data)
    np.save('./data/vox_test_lab', label)

else :
    data = []
    label = []
    n_size = 48
    n_slice = 8
    cs_strides = 24
    a_strides = 4
    cnt=0
    start_idx = val_idx[args.val_num]

    for name in si15_lists[start_idx:start_idx+18]:
        tmp_path = os.path.join(data_dir, name)
        head = nib.load(tmp_path)
        img = head.get_data().astype(np.uint16)
        cor, sag, axi = img.shape
        means = [img[idx:idx+n_size, jdx:jdx+n_size, kdx:kdx+(n_slice*6)].mean() 
                for idx in range(0, cor-n_size, cs_strides) 
                for jdx in range(0, sag-n_size, cs_strides) 
                for kdx in range(0, axi-(n_slice*6), a_strides)]
        
        for idx in range(0, cor-n_size, cs_strides):
            for jdx in range(0, sag-n_size, cs_strides):
                for kdx in range(0, axi-(n_slice*6), a_strides*6):
                    tmp = img[idx:idx+n_size, jdx:jdx+n_size, kdx:kdx+(n_slice*6)]
                    if tmp.mean() > np.mean(means)+10:
                        cnt+=1
                        label.append(tmp)
                        tmp = np.array(np.dsplit(tmp, n_slice))
                        tmp = tmp.mean(axis=-1)
                        tmp = np.transpose(tmp, [1,2,0])
                        data.append(tmp)
                        
        label = np.array(label)[..., np.newaxis]
        data = np.array(data)[..., np.newaxis]
        print(data.shape, label.shape)

        np.save('./data/vox_part_%02d_img'%(args.val_num), data)
        np.save('./data/vox_part_%02d_lab'%(args.val_num), label)

