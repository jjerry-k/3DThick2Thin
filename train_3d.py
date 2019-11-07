import os, csv, tqdm, datetime, random, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", dest="input_dir", type=str, help="Path of Input")
parser.add_argument("--model", dest="model", type=str, help="Model Name")
parser.add_argument("--plain", dest="plain", type=str, help="Plain of MRI")
parser.add_argument("--num_channel", dest="num_channel", type=int, help="Number of channels")
parser.add_argument("--val_num", dest="val_num", type=int, help="Validation Data Number", default=0)
parser.add_argument("--loss", dest="loss", type=str, help="Type of Loss")
parser.add_argument("--batch", dest="batch", type=int, help="Size of Batch")
parser.add_argument("--epochs", dest="epochs", type=int, help="Number of Epochs")

args = parser.parse_args()

dict_args = vars(args)

for i in dict_args.keys():
    assert dict_args[i]!=None, '"%s" key is None Value!'%i
print("\n================ Training Options ================")
print("Input dir : ", args.input_dir)
print("Model name : ", args.model)
print("Plain : ", args.plain)
print("Number of channels : ", args.num_channel)
print("Validation Number : ", args.val_num)
print("Loss Function: ", args.loss.upper())
print("Batch Size : ", args.batch)
print("Epochs : ", args.epochs)
print("====================================================\n")


import cv2 as cv
import numpy as np
import pandas as pd
import nibabel as nib
import tensorflow as tf
from tensorflow.keras import models, layers, losses, optimizers
from tensorflow.keras.utils import Progbar

from loss import *
from utils import *
from metric import *
from network import *

random.seed(777)
tf.set_random_seed(777)
np.random.seed(777)
os.environ['PYTHONHASHSEED'] = '777'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

loss_dict = {'MSE': mse, 'GRAD': gradient_3d_loss, 'MSEGRAD': mse_grad_loss}


# ============================================================================================
# ======================================== CC 359 ===========================================
# ============================================================================================
print("================ Loading Data ================")

plain = args.plain
data_root = args.input_dir

npy_list = [npy for npy in sorted(os.listdir(data_root)) if '.npy' in npy and plain in npy]
#print(npy_list)

train_img_list = [npy for npy in npy_list if 'img' in npy]
train_lab_list = [npy for npy in npy_list if 'lab' in npy]

except_idx = args.val_num
val_img_list = None
val_lab_list = None

if except_idx is not None:
    val_img_list = train_img_list.pop(except_idx)
    val_lab_list = train_lab_list.pop(except_idx)
print("Training Lists")
print(train_img_list)
print(train_lab_list)

print("\nValidation Lists")
print(val_img_list)
print(val_lab_list)

train_img = []
train_lab = []
cnt = 0
for img_name, lab_name in zip(train_img_list, train_lab_list):
    
    tmp_img = np.load(os.path.join(data_root, img_name))
    tmp_lab = np.load(os.path.join(data_root, lab_name))
    
    if cnt == 0:
        train_img = tmp_img
        train_lab = tmp_lab
    else:
        train_img = np.concatenate((train_img, tmp_img), axis=0)
        train_lab = np.concatenate((train_lab, tmp_lab), axis=0)
    cnt += 1
    
    
val_img = np.load(os.path.join(data_root, img_name))
val_lab = np.load(os.path.join(data_root, lab_name))

print("\nTraining shape")
print(train_img.shape)
print(train_lab.shape)
print("\nValidation shape")
print(val_img.shape)
print(val_lab.shape)


print("=================================================\n")





print("================ Building Network ================")
residual_channels = args.num_channel
G = SR3D_res(residual_channel=residual_channels, 
             layer_activation='leaky_relu', 
             name='3D_SR_Gen')
D = dis3D_res(residual_channel=residual_channels, 
              name='3D_SR_Dis')

D.compile(optimizer=optimizers.Adam(lr=0.0001, epsilon=1e-8), loss=losses.binary_crossentropy)

D.trainable=False

A = models.Model(inputs=G.input, outputs = [G.output, D(G.input)], name='GAN')
A.compile(optimizer=optimizers.Adam(lr=0.0001, epsilon=1e-8), loss=[loss_dict[args.loss.upper()], losses.binary_crossentropy], 
          loss_weights=[10, 1], metrics={'3D_SR_Gen_output_act':[gradient_3d_loss, psnr]})
print("==================================================\n")




print("================ Making save point ================")
date = datetime.datetime.today()
date = "%04d_%02d_%02d"%(date.year, date.month, date.day)

common_path = '%s/%s/RBSRGAN%d_%s_%s_%d'%(date, args.model, residual_channels, args.loss, plain, except_idx)

ckpt_root = os.path.join('./checkpoint', common_path)
result_root = os.path.join('./result', common_path)
print(ckpt_root)
print(result_root)

try:
    os.makedirs(ckpt_root)
    print("\nMake Save Directory!\n")
except:
    print("\nDirectory Already Exist!\n")

try:
    os.makedirs(result_root)
    print("\nMake Save Directory!\n")
except:
    print("\nDirectory Already Exist!\n")
    

model_json = A.to_json()
with open(os.path.join(ckpt_root, "model.json"), "w") as json_file:
    json_file.write(model_json)
print("\nModel Saved!\n")
print("===================================================\n")




print("================ Training Start ! ================")
print("==================================================")
epochs=args.epochs
batch_size = args.batch
train_length = len(train_img)
val_length = len(val_img)
num_iter = int(np.ceil(train_length/batch_size))
num_val_iter = int(np.ceil(val_length/batch_size))

# train_loss = {"Generator_Total" : [], "Generator_Style" : [], "Generator_AD" : [], 
#               "mi": [], "mse": [], "grad": [], "psnr": [], "Discriminator_AD" : []}
# val_loss = {"Generator_Total" : [], "Generator_Style" : [], "Generator_AD" : [], 
#             "mi": [], "mse": [], "grad": [], "psnr": []}

train_dict_keys = ['Generator_Total', 'Generator_Style', 'Generator_AD', 
                   'mse', 'grad', 'psnr', 'Discriminator_AD']

val_dict_keys = ['Generator_Total', 'Generator_Style', 'Generator_AD', 
                 'mse', 'grad', 'psnr']

top_gen_loss = float('inf')
stop_patience = 5
stop_cnt = 0
save_patience = 2
save_cnt = 0
top_epoch = 0
prev_val_loss = 0
prev_val_mi = 0
prev_val_grad = 0
prev_val_psnr = 0

print(train_length, batch_size, num_iter)

total_progbar = Progbar(epochs)
for epoch in range(epochs):
    epoch_t_g_total = 0
    epoch_t_g_style = 0
    epoch_t_g_dis = 0
    epoch_t_d_dis = 0
    epoch_t_mse = 0
    epoch_t_mi = 0
    epoch_t_grad = 0
    epoch_t_psnr = 0
    
    shuffle_idx = np.random.choice(train_length, train_length, replace=False)
    
    epoch_progbar = Progbar(num_iter, width=15)
    for i, step in enumerate(range(0, train_length, batch_size)):

        #print(step, shuffle_idx[step:step+batch_size])

        # Generate fake images
        step_idx = shuffle_idx[step:step+batch_size]
        fake_imgs = G.predict(train_img[step_idx])

        # Train Discriminator
        dis_input = np.concatenate([fake_imgs, train_lab[step_idx]])
        dis_label =np.concatenate([np.zeros((len(step_idx), 1)),
                                 np.ones((len(step_idx), 1))])
        Dis_Loss = D.train_on_batch(dis_input, dis_label)

        # Train Generator
        Gan_Loss = A.train_on_batch(train_img[step_idx], [train_lab[step_idx], np.ones((len(step_idx), 1))])
        
        epoch_t_g_total += Gan_Loss[0]
        epoch_t_g_style += Gan_Loss[-4]
        epoch_t_g_dis += Gan_Loss[-3]
        epoch_t_mse += np.mean(np.square(fake_imgs-train_lab[step_idx]))
        epoch_t_grad += Gan_Loss[-2]
        epoch_t_psnr += Gan_Loss[-1]
        epoch_t_d_dis += Dis_Loss
        
        if i != num_iter-1:
            epoch_progbar.update(i+1, [("G_Style", Gan_Loss[-4]), ("G_MSE", np.mean(np.square(fake_imgs-train_lab[step_idx]))), ("G_Grad", Gan_Loss[-2])])
    
#     train_loss["Generator_Total"].append(epoch_t_g_total/num_iter)
#     train_loss["Generator_Style"].append(epoch_t_g_style/num_iter)
#     train_loss["Generator_AD"].append(epoch_t_g_dis/num_iter)
#     train_loss["Discriminator_AD"].append(epoch_t_d_dis/num_iter)
#     train_loss["mse"].append(epoch_t_mse/num_iter)
#     train_loss["mi"].append(epoch_t_mi/num_iter)
#     train_loss["grad"].append(epoch_t_grad/num_iter)
#     train_loss["psnr"].append(epoch_t_psnr/num_iter)
    
    tr_gen_tot = epoch_t_g_total/num_iter
    tr_gen_sty = epoch_t_g_style/num_iter
    tr_gen_ad = epoch_t_g_dis/num_iter
    tr_dis_ad = epoch_t_d_dis/num_iter
    tr_mse = epoch_t_mse/num_iter
    tr_grad = epoch_t_grad/num_iter
    tr_psnr = epoch_t_psnr/num_iter
    
    epoch_v_g_total = 0
    epoch_v_g_style = 0
    epoch_v_g_dis = 0
    epoch_v_mse = 0
    epoch_v_mi = 0
    epoch_v_grad = 0
    epoch_v_psnr = 0
    
    for j, val_idx in enumerate(range(0, val_length, batch_size)):
        val_y2 = np.ones([len(val_img[val_idx:val_idx+batch_size]), 1])
        V_loss = A.test_on_batch(val_img[val_idx:val_idx+batch_size], 
                                           [val_lab[val_idx:val_idx+batch_size], val_y2])
        V_output, _= A.predict(val_img[val_idx:val_idx+batch_size])
        
        epoch_v_g_total += V_loss[0]
        epoch_v_g_style += V_loss[-4]
        epoch_v_g_dis += V_loss[-3]
        epoch_v_mse += np.mean(np.square(V_output-val_lab[val_idx:val_idx+batch_size]))
        epoch_v_grad += V_loss[-2]
        epoch_v_psnr += V_loss[-1]
    
#     val_loss["Generator_Total"].append(epoch_v_g_total/num_val_iter)
#     val_loss["Generator_Style"].append(epoch_v_g_style/num_val_iter)
#     val_loss["Generator_AD"].append(epoch_v_g_dis/num_val_iter)
#     val_loss["mse"].append(epoch_v_mse/num_val_iter)
#     val_loss["mi"].append(epoch_v_mi/num_val_iter)
#     val_loss["grad"].append(epoch_v_grad/num_val_iter)
#     val_loss["psnr"].append(epoch_v_psnr/num_val_iter)
    val_gen_tot = epoch_t_g_total/num_val_iter
    val_gen_sty = epoch_t_g_style/num_val_iter
    val_gen_ad = epoch_t_g_dis/num_val_iter
    val_mse = epoch_t_mse/num_val_iter
    val_grad = epoch_t_grad/num_val_iter
    val_psnr = epoch_t_psnr/num_val_iter
    
    mean_val_loss = val_gen_sty
#     prev_val_loss = mean_val_loss
#     prev_val_mi = epoch_v_mi/num_val_iter
#     prev_val_grad = epoch_v_grad/num_val_iter
#     prev_val_psnr = epoch_v_psnr/num_val_iter
    
    # For csv every epoch
    train_csv =  open(os.path.join(result_root,'train_loss.csv'), 'a', newline='')
    val_csv =  open(os.path.join(result_root, 'val_loss.csv'), 'a', newline='')
    
    train_writer = csv.writer(train_csv)
    val_writer = csv.writer(val_csv)
    
    if epoch == 0:
        train_writer.writerow(train_dict_keys)
        val_writer.writerow(val_dict_keys)
        
    train_writer.writerow([tr_gen_tot, tr_gen_sty, tr_gen_ad, tr_mse, tr_grad, tr_psnr, tr_dis_ad])
    val_writer.writerow([val_gen_tot, val_gen_sty, val_gen_ad, val_mse, val_grad, val_psnr])
    
    train_csv.close()
    val_csv.close()
    
    
    epoch_progbar.update(i+1, [("Val_G_Style", val_gen_sty), ("Val_G_MSE", val_mse), ("Val_G_Grad", val_grad)])
    
#     total_progbar.update(epoch+1, [("G_Style", epoch_t_g_style/num_iter), ("G_MSE", epoch_t_mse/num_iter), ("G_Grad", epoch_t_grad/num_iter), 
#                                    ("Val_G_Style", epoch_v_g_style/num_val_iter), ("Val_G_MSE", epoch_v_mse/num_val_iter), ("Val_G_Grad", epoch_v_grad/num_iter)])
    # Saving Phase
    if mean_val_loss < top_gen_loss:
        stop_cnt = 0
        top_gen_loss = mean_val_loss
        if epoch == 0: 
            A.save_weights(os.path.join(ckpt_root, "%05d_%.4f_%.4f.h5"%(epoch+1, epoch_t_g_style, top_gen_loss)))
            top_epoch = epoch
        elif top_epoch + save_patience > epoch : pass
        else:
            A.save_weights(os.path.join(ckpt_root, "%05d_%.4f_%.4f.h5"%(epoch+1, epoch_t_g_style, top_gen_loss)))
            top_epoch = epoch
    else:
        stop_cnt+=1
    
    if stop_cnt == stop_patience : break

print("Training Done ! ")

# train_df = pd.DataFrame(train_loss)
# train_df.to_csv(os.path.join(result_root,'train_loss.csv'))

# val_df = pd.DataFrame(val_loss)
# val_df.to_csv(os.path.join(result_root, 'val_loss.csv'))