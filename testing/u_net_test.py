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
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
# from scipy import ndimage
import torch.nn.functional as F
import cv2 as cv
import time
from tqdm import tqdm
from skimage.transform import resize

from model_func import Maskdata2D, UNet2D_mask
from func import train_nn, eval_nn, finetune_nn, loading_data

 # %%               
train_dataset, test_dataset, val_dataset_train, val_dataset_test, std_ground_ft, mean_ground_ft,  mean_ground, std_ground,  mean_mask_ft, std_mask_ft = loading_data()

# %%
# import VGG model for VGGloss
import torchvision.models as models
vgg = models.vgg16(pretrained=True).features[:6]
def perceptual_loss(x, y):
    features_x = vgg(x)
    features_y = vgg(y)
    return torch.nn.functional.mse_loss(features_x, features_y)

augmentation_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=(-45,45)),
    transforms.GaussianBlur(7, sigma=(0.1, 2.0)),
    transforms.RandomAffine(degrees=(0, 50), translate=(0, 0.3), scale=(0.5, 1))
])
# %%
# Create augmented input and target datasets for fine tune
augmented_inputs_val = []
augmented_targets_val = []
augmented_weights_val = []
augmented_masks_val = []


# %%
# Create data loaders for training and testing datasets
batch_size = 16

val_dataloader_train = DataLoader(val_dataset_train, batch_size=batch_size, shuffle=True)
val_dataloader_test = DataLoader(val_dataset_test, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the U-Net model
in_channels = 14
out_channels = 7
model = UNet2D_mask(in_channels, out_channels)
model.load_state_dict(torch.load( "./models/DOT_filter_model.pth"))
model.to(device)
model.eval()

criterion = nn.MSELoss(reduction='none')
# criterion = pytorch_ssim.ssim()
criterion_pert = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=2e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=100,threshold=0.01, threshold_mode='abs',verbose =True)
vgg.to(device)

# %%
def eval_nn(model, data):
    model.eval()
    running_loss = 0.0
    pert_output_store = np.empty((0,2772))
    img_output_store = np.empty((0,7,64,64))
    
    with torch.no_grad():
        for batch_idx, (inputs, targets, weights, perts) in enumerate(data):
            inputs = inputs.to(device)
            targets = targets.to(device)
            weights = weights.to(device)
            perts = perts.to(device)
            actual_perts = perts[:,:2772]
            geom = perts[:,2772:]

            weight_matrix = torch.where(weights, 98, 2)
            
            # Forward pass
            outputs, pert_outputs = model(inputs,geom)
            loss_img = criterion(outputs, targets)
            pert_loss = 500 * criterion_pert(pert_outputs, actual_perts)
            # loss = torch.mean(loss_img * weight_matrix) + pert_loss
            
            pert_output_store = np.append(pert_output_store,np.array(pert_outputs.detach().cpu()), axis=0)
            img_output_store = np.append(img_output_store,np.array(outputs.detach().cpu()), axis=0)
            
            outputs_img = outputs.squeeze(1)[:,:,2:-2,2:-2].reshape((len(outputs),3,100,84))
            targets_img = targets.squeeze(1)[:,:,2:-2,2:-2].reshape((len(targets),3,100,84))
            loss_vgg = perceptual_loss(outputs_img, targets_img)
            loss = torch.mean(loss_img * weight_matrix) + loss_vgg + pert_loss
            
            # loss_img_ssim = 100 * pytorch_ssim.ssim(outputs.squeeze(), targets.squeeze(),weight_matrix.squeeze())
            # loss = torch.mean(loss_img * weight_matrix) + pert_loss + loss_img_ssim
            # loss = loss_img + criterion_pert(pert_outputs, actual_perts) #+  0.05*back_reduce_loss(outputs)
            
            running_loss += loss.item()
          
        # Print average loss over the validation dataset
        avg_loss = running_loss / len(data)
        print('Average Loss: {:.4f}'.format(avg_loss))
        
        return pert_output_store, img_output_store

pert_output_store, img_output_store = eval_nn(model, val_dataloader_test) 

test3data = scipy.io.loadmat('.dataset/threetars.mat')
recons_three = np.array(test3data['volume7all'])
mask_three = np.array(test3data['maskall'])
recons_three_test = np.tile(recons_three,(64,1,1,1))
recons_three = (recons_three - mean_ground_ft)/std_ground_ft
mask_three = (mask_three - mean_mask_ft)/std_mask_ft
pert_three = np.ones((len(recons_three), 2772))
test_dataset_three = Maskdata2D(recons_three, recons_three, recons_three, pert_three, mask_three)
test_dataloader_three = DataLoader(test_dataset_three, batch_size=batch_size, shuffle=False)
(inputs_three, targets_three,  weights_three, perts_three) = next(iter(test_dataloader_three))
inputs = inputs_three.to(device)
perts = perts_three.to(device)
perts = perts[:,2772:]
outputs, perts_out = model(inputs,perts)

