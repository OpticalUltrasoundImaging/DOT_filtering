# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 13:06:57 2024

@author: mingh
"""

import torch
import torch.nn as nn
import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
# from scipy import ndimage
import time
from tqdm import tqdm

from model_func import Maskdata2D, UNet2D_mask
from func import train_nn, eval_nn, finetune_nn, loading_data

# %%
# import VGG model for VGGloss
import torchvision.models as models
vgg = models.vgg16(pretrained=True).features[:6]
def perceptual_loss(x, y):
    features_x = vgg(x)
    features_y = vgg(y)
    return torch.nn.functional.mse_loss(features_x, features_y)


# %%
# loading the 
train_dataset, test_dataset, ft_dataset_train, ft_dataset_test, std_ground_ft, mean_ground_ft,  mean_ground, std_ground,  mean_mask_ft, std_mask_ft = loading_data()

# %%
# Create data loaders for training and testing datasets
batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
ft_dataloader_train = DataLoader(ft_dataset_train, batch_size=batch_size, shuffle=True)
ft_dataloader_test = DataLoader(ft_dataset_test, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the U-Net model
in_channels = 14
out_channels = 7
model = UNet2D_mask(in_channels, out_channels)

criterion = nn.MSELoss(reduction='none')
# criterion = pytorch_ssim.ssim()
criterion_pert = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=2e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=100,threshold=0.01, threshold_mode='abs',verbose =True)
vgg.to(device)

def depth_contrast():
    inputs_max = []
    grounds_max = []
    outputs_max = []
    inputs_contrast = []
    grounds_contrast = []
    outputs_contrast = []
    third_inputs_contrast = []
    third_grounds_contrast = []
    third_outputs_contrast = []
    for idx in range(len(ft_dataset_test)):
        inputs = ft_dataloader_test[idx][0][:7,:,:].detach().cpu().numpy()
        inputs = std_ground_ft * inputs + mean_ground_ft
        input_max = inputs.max()
        inputs_max.append(input_max)
        
        grounds = ft_dataloader_test[idx][1].detach().cpu().numpy()
        grounds = std_ground_ft * grounds + mean_ground_ft
        ground_max = grounds.max()
        grounds_max.append(ground_max)
        
        outputs = img_output_store[idx]
        outputs = std_ground_ft * outputs + mean_ground_ft
        output_max = outputs.max()
        outputs_max.append(output_max)
        
        masks = ft_dataset_train[idx][0][7:,:,:].detach().cpu().numpy()
        masks = std_mask_ft * masks + mean_mask_ft
        if np.count_nonzero(masks.sum(axis=(1,2))) > 1:
            first_idx = (masks.sum(axis=(1,2))!=0).argmax(axis=0)
            second_idx = first_idx + 1
            
            input_contrast = inputs[second_idx].mean()/inputs[first_idx].mean()
            inputs_contrast.append(input_contrast)
            ground_contrast = grounds[second_idx].mean()/grounds[first_idx].mean()
            grounds_contrast.append(ground_contrast)
            output_contrast = outputs[second_idx].mean()/outputs[first_idx].mean()
            outputs_contrast.append(output_contrast)
            if np.count_nonzero(masks.sum(axis=(1,2))) > 2:
                first_idx = (masks.sum(axis=(1,2))!=0).argmax(axis=0)
                second_idx = first_idx + 2
                
                input_contrast = inputs[second_idx].mean()/inputs[first_idx].mean()
                third_inputs_contrast.append(input_contrast)
                ground_contrast = grounds[second_idx].mean()/grounds[first_idx].mean()
                third_grounds_contrast.append(ground_contrast)
                output_contrast = outputs[second_idx].mean()/outputs[first_idx].mean()
                third_outputs_contrast.append(output_contrast)
        else:
            continue
    return inputs_max, grounds_max, outputs_max, inputs_contrast, grounds_contrast, outputs_contrast, third_inputs_contrast, third_grounds_contrast, third_outputs_contrast
            

# %%
trained_unet, epoch_loss_all, avg_loss_all = train_nn(model, train_dataloader, test_dataloader) 

pert_output_store, img_output_store = eval_nn(trained_unet, test_dataloader, std_ground_ft, mean_ground_ft)
plt.plot(epoch_loss_all,label='training loss')
plt.plot(avg_loss_all,label='testing loss')
plt.show()

finetune_unet, epoch_loss_all_finetune, avg_loss_all_finetune = finetune_nn(model, ft_dataloader_train, ft_dataloader_test) 
pert_output_store, img_output_store = eval_nn(finetune_unet, ft_dataloader_test, std_ground_ft, mean_ground_ft) 
inputs_max, grounds_max, outputs_max, inputs_contrast, grounds_contrast, outputs_contrast, third_inputs_contrast, third_grounds_contrast, third_outputs_contrast = depth_contrast()
input_acc = [inputs_max[i] / grounds_max[i] for i in range(len(inputs_max))]
output_acc = [outputs_max[i] / grounds_max[i] for i in range(len(outputs_max))]
# pert_output_store_ft, img_output_store_ft = eval_nn(trained_unet, val_dataloader)

test3data = scipy.io.loadmat('./dataset/threetars.mat')
recons_three = np.array(test3data['volume7all'])
mask_three = np.array(test3data['maskall'])
recons_three_test = np.tile(recons_three,(64,1,1,1))
recons_three = (recons_three - mean_ground)/std_ground
mask_three = (mask_three - mean_mask_ft)/std_mask_ft
pert_three = np.ones((len(recons_three), 2772))
test_dataset_three = Maskdata2D(recons_three, recons_three, recons_three, pert_three, mask_three)
test_dataloader_three = DataLoader(test_dataset_three, batch_size=batch_size, shuffle=False)
(inputs_three, targets_three,  weights_three, perts_three) = next(iter(test_dataloader_three))
inputs = inputs_three.to(device)
perts = perts_three.to(device)
perts = perts[:,2772:]
outputs, perts_out = model(inputs,perts)
