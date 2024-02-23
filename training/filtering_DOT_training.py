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
from func import train_nn, eval_nn, finetune_nn

data = scipy.io.loadmat('./dataset/simudata_dataset.mat')
recons = data['volume7all']
groundtruthes = data['real_reconall']
weight_matrix = data['weight_matrixall']
perturbations = data['data_forsaveall']
mua0all = data['mua0all']
mus0all = data['mus0all']
radius1all = data['radius1all']
radius2all = data['radius2all']
radius3all = data['radius3all']
rtg1all = data['rtg1all']
rtg2all = data['rtg2all']
rtg3all = data['rtg3all']
ua1all = data['ua1all']
ua2all = data['ua2all']
ua3all = data['ua3all']
# histall = data['hist_all']
maskall = data['maskall']
perts = np.concatenate((perturbations[:,:,0], perturbations[:,:,1], perturbations[:,:,2]), axis = 1).astype(np.float32)
weight_matrix = weight_matrix + 1

torch.manual_seed(21)
X_train, X_test, y_train, y_test, weight_train, weight_test, pert_train, pert_test, mua0train, mua0test, mus0train, mus0test, \
    radius1train, radius1test, radius2train, radius2test, radius3train, radius3test, rtg1train, rtg1test, rtg2train, rtg2test, \
        rtg3train, rtg3test, ua1train, ua1test, ua2train, ua2test, ua3train, ua3test, mask_train, mask_test \
            = train_test_split(recons, groundtruthes, weight_matrix, perts, mua0all, mus0all, radius1all, radius2all, radius3all, \
                                             rtg1all, rtg2all, rtg3all, ua1all, ua2all, ua3all, maskall, test_size=0.15, random_state=42) # usually 21
 # %%               
# load single scan dataset as validation
valdata = scipy.io.loadmat('./dataset/finetune_dataset.mat')
recons_val = valdata['volume7all']
groundtruthes_val = valdata['real_reconall']
weight_matrix_val = valdata['weight_matrixall']
perturbations_val = valdata['data_forsaveall']
mua0all_val = valdata['mua0all']
mus0all_val = valdata['mus0all']
radius1all_val = valdata['radius1all']
radius2all_val = valdata['radius2all']
radius3all_val = valdata['radius3all']
rtg1all_val = valdata['rtg1all']
rtg2all_val = valdata['rtg2all']
rtg3all_val = valdata['rtg3all']
ua1all_val = valdata['ua1all']
ua2all_val = valdata['ua2all']
ua3all_val = valdata['ua3all']
# hist_val = valdata['hist_all']
maskall_val = valdata['maskall']
perts_val = np.concatenate((perturbations_val[:,:,0], perturbations_val[:,:,1], perturbations_val[:,:,2]), axis = 1).astype(np.float32)
weight_matrix_val = weight_matrix_val + 1

X_train_val, X_test_val, y_train_val, y_test_val, weight_train_val, weight_test_val, pert_train_val, pert_test_val, mua0train_val, mua0test_val, mus0train_val, mus0test_val, \
    radius1train_val, radius1test_val, radius2train_val, radius2test_val, radius3train_val, radius3test_val, rtg1train_val, rtg1test_val, rtg2train_val, rtg2test_val, \
        rtg3train_val, rtg3test_val, ua1train_val, ua1test_val, ua2train_val, ua2test_val, ua3train_val, ua3test_val, mask_train_val, mask_test_val \
            = train_test_split(recons_val, groundtruthes_val, weight_matrix_val, perts_val, mua0all_val, mus0all_val, radius1all_val, radius2all_val, radius3all_val, \
                                             rtg1all_val, rtg2all_val, rtg3all_val, ua1all_val, ua2all_val, ua3all_val, maskall_val, test_size=0.15, random_state=42) # usually 21
# %%
# pre-processing for training
noise = np.random.normal(0,0.015, X_train.shape)
X_train = X_train + noise
mean_recon = np.mean(recons)
std_recon = np.std(recons)
mean_ground = np.mean(groundtruthes)
std_ground = np.std(groundtruthes)
# mean_hist = np.mean(histall)
# std_hist = np.std(histall)
mean_mask = np.mean(maskall)
std_mask = np.std(maskall)
X_train = (X_train - mean_ground)/std_ground
X_test = (X_test - mean_ground)/std_ground
y_train = (y_train - mean_ground)/std_ground
y_test = (y_test - mean_ground)/std_ground
# hist_train = (hist_train - mean_hist)/std_hist
# hist_test = (hist_test - mean_hist)/std_hist
mask_train = (mask_train - mean_mask)/std_mask
mask_test = (mask_test - mean_mask)/std_mask
threshold = (0.08 - mean_ground)/std_ground
# %%
# pre-processing for fine-tune
noise_val = np.random.normal(0,0.0015, X_train_val.shape)
X_train_val = X_train_val + noise_val
mean_recon_val = np.mean(recons_val)
std_recon_val = np.std(recons_val)
mean_ground_val = np.mean(groundtruthes_val)
std_ground_val = np.std(groundtruthes_val)
# mean_hist_val = np.mean(hist_val)
# std_hist_val = np.std(hist_val)
mean_mask_val = np.mean(maskall_val)
std_mask_val = np.std(maskall_val)
X_train_val = (X_train_val - mean_ground_val)/std_ground_val
X_test_val = (X_test_val - mean_ground_val)/std_ground_val
y_train_val = (y_train_val - mean_ground_val)/std_ground_val
y_test_val = (y_test_val - mean_ground_val)/std_ground_val
# hist_train_val = (hist_train_val - mean_hist_val)/std_hist_val
mask_train_val = (mask_train_val - mean_mask_val)/std_mask_val
mask_test_val = (mask_test_val - mean_mask_val)/std_mask_val
# hist_test_val = (hist_test_val - mean_hist_val)/std_hist_val
# %%
# import VGG model for VGGloss
import torchvision.models as models
vgg = models.vgg16(pretrained=True).features[:6]
def perceptual_loss(x, y):
    features_x = vgg(x)
    features_y = vgg(y)
    return torch.nn.functional.mse_loss(features_x, features_y)

    
max_absorp_recon = np.max(recons)
max_absorp_gt = np.max(groundtruthes)
augmentation_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=(-45,45)),
    transforms.GaussianBlur(7, sigma=(0.1, 2.0)),
    transforms.RandomAffine(degrees=(0, 50), translate=(0, 0.3), scale=(0.5, 1))
])
# %%
# Create augmented input and target datasets
augmented_inputs = []
augmented_targets = []
augmented_weights = []
augmented_masks = []

for i in range(len(X_train)):
    input_sample = X_train[i]
    target_sample = y_train[i]
    weight_sample = weight_train[i]
    weight_sample[weight_sample == 2] = 0
    mask_sample = mask_train[i]

    # augmented_input_sample = np.copy(input_sample)
    # augmented_target_sample = np.copy(target_sample)
    
    sample = np.dstack((input_sample, target_sample, weight_sample, mask_sample))
    augmented_sample = augmentation_transforms(sample)
    augmented_sample = np.transpose(augmented_sample, (1, 2, 0))

    # augmented_inputs.append(input_sample)
    augmented_inputs.append(augmented_sample[:,:,:7])
    # augmented_targets.append(target_sample)
    augmented_targets.append(augmented_sample[:,:,7:14])
    weight_auged = augmented_sample[:,:,14:21]
    weight_auged[weight_auged == 2] = 0
    augmented_weights.append(weight_auged)
    augmented_masks.append(augmented_sample[:,:,21:])

# %%
# Create augmented input and target datasets for fine tune
augmented_inputs_val = []
augmented_targets_val = []
augmented_weights_val = []
augmented_masks_val = []

for i in range(len(X_train_val)):
    input_sample_val = X_train_val[i]
    target_sample_val = y_train_val[i]
    weight_sample_val = weight_train_val[i]
    weight_sample_val[weight_sample_val == 2] = 0
    mask_sample_val = mask_train_val[i]

    # augmented_input_sample = np.copy(input_sample)
    # augmented_target_sample = np.copy(target_sample)
    
    sample_val = np.dstack((input_sample_val, target_sample_val, weight_sample_val, mask_sample_val))
    augmented_sample_val = augmentation_transforms(sample_val)
    augmented_sample_val = np.transpose(augmented_sample_val, (1, 2, 0))

    # augmented_inputs.append(input_sample)
    augmented_inputs_val.append(augmented_sample_val[:,:,:7])
    # augmented_targets.append(target_sample)
    augmented_targets_val.append(augmented_sample_val[:,:,7:14])
    weight_auged_val = augmented_sample_val[:,:,14:21]
    weight_auged_val[weight_auged_val == 2] = 0
    augmented_weights_val.append(weight_auged_val)
    augmented_masks_val.append(augmented_sample_val[:,:,21:])

weight_test_val[weight_test_val == 2] = 0  
weight_test_val = weight_test_val.astype(bool)
weight_test[weight_test == 2] = 0
weight_test = weight_test.astype(bool)
augmented_inputs = np.array(augmented_inputs)
augmented_targets = np.array(augmented_targets)
augmented_weights = np.array(augmented_weights).astype(bool)
augmented_masks = np.array(augmented_masks)
augmented_inputs_val = np.array(augmented_inputs_val)
augmented_targets_val = np.array(augmented_targets_val)
augmented_weights_val = np.array(augmented_weights_val).astype(bool)
augmented_masks_val = np.array(augmented_masks_val)
# %%

train_dataset = Maskdata2D(augmented_inputs, augmented_targets, augmented_weights, pert_train, augmented_masks)
test_dataset = Maskdata2D(X_test, y_test, weight_test, pert_test, mask_test)
val_dataset_train = Maskdata2D(augmented_inputs_val, augmented_targets_val, augmented_weights_val, pert_train_val, augmented_masks_val)
val_dataset_test = Maskdata2D(X_test_val, y_test_val, weight_test_val, pert_test_val, mask_test_val)

images = np.array(weight_auged)
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
plt.title('reconstruction')
min_value = np.min(images)
max_value = np.max(images)
for i, ax in enumerate(axes.flatten()):
    if i < 7:
        im = ax.imshow(np.transpose(images[:,:,i]),vmin=min_value, vmax=max_value) 
        ax.axis('off')  # Turn off axis labels
        
        cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
        cbar = fig.colorbar(im, cax=cax)

    else:
        ax.axis('off') 
plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.show()
# %%
# Create data loaders for training and testing datasets
batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_dataloader_train = DataLoader(val_dataset_train, batch_size=batch_size, shuffle=True)
val_dataloader_test = DataLoader(val_dataset_test, batch_size=batch_size, shuffle=False)

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
    for idx in range(len(val_dataset_test)):
        inputs = val_dataset_test[idx][0][:7,:,:].detach().cpu().numpy()
        inputs = std_ground_val * inputs + mean_ground_val
        input_max = inputs.max()
        inputs_max.append(input_max)
        
        grounds = val_dataset_test[idx][1].detach().cpu().numpy()
        grounds = std_ground_val * grounds + mean_ground_val
        ground_max = grounds.max()
        grounds_max.append(ground_max)
        
        outputs = img_output_store[idx]
        outputs = std_ground_val * outputs + mean_ground_val
        output_max = outputs.max()
        outputs_max.append(output_max)
        
        masks = val_dataset_test[idx][0][7:,:,:].detach().cpu().numpy()
        masks = std_mask_val * masks + mean_mask_val
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

pert_output_store, img_output_store = eval_nn(trained_unet, test_dataloader, std_ground_val, mean_ground_val)
plt.plot(epoch_loss_all,label='training loss')
plt.plot(avg_loss_all,label='testing loss')
plt.show()

finetune_unet, epoch_loss_all_finetune, avg_loss_all_finetune = finetune_nn(model, val_dataloader_train, val_dataloader_test) 
pert_output_store, img_output_store = eval_nn(finetune_unet, val_dataloader_test, std_ground_val, mean_ground_val) 
inputs_max, grounds_max, outputs_max, inputs_contrast, grounds_contrast, outputs_contrast, third_inputs_contrast, third_grounds_contrast, third_outputs_contrast = depth_contrast()
input_acc = [inputs_max[i] / grounds_max[i] for i in range(len(inputs_max))]
output_acc = [outputs_max[i] / grounds_max[i] for i in range(len(outputs_max))]
# pert_output_store_val, img_output_store_val = eval_nn(trained_unet, val_dataloader)

test3data = scipy.io.loadmat('./dataset/threetars.mat')
recons_three = np.array(test3data['volume7all'])
mask_three = np.array(test3data['maskall'])
recons_three_test = np.tile(recons_three,(64,1,1,1))
recons_three = (recons_three - mean_ground)/std_ground
mask_three = (mask_three - mean_mask)/std_mask
pert_three = np.ones((len(recons_three), 2772))
test_dataset_three = Maskdata2D(recons_three, recons_three, recons_three, pert_three, mask_three)
test_dataloader_three = DataLoader(test_dataset_three, batch_size=batch_size, shuffle=False)
(inputs_three, targets_three,  weights_three, perts_three) = next(iter(test_dataloader_three))
inputs = inputs_three.to(device)
perts = perts_three.to(device)
perts = perts[:,2772:]
outputs, perts_out = model(inputs,perts)

# # calculate and save clinical result
# def calculate_clinical():
#     test_patient = scipy.io.loadmat('./clinical_shallow_example/hemo/data_diagnostic_0213.mat')
    
#     recons_patient = np.array(test_patient['volume_all'])
#     mask_patient = np.array(test_patient['mask_all'])
    
#     recons_patient = (recons_patient - mean_ground)/std_ground
#     mask_patient = (mask_patient - mean_mask)/std_mask
#     pert_patient = np.ones((len(recons_patient), 2772))
#     test_dataset_patient = Maskdata2D(recons_patient, recons_patient, recons_patient, pert_patient, mask_patient)
#     test_dataloader_patient = DataLoader(test_dataset_patient, batch_size=batch_size, shuffle=False)
    
    
#     inputs_contrast = []
#     outputs_contrast = []
#     third_inputs_contrast = []
#     third_outputs_contrast = []
#     inputs_all = np.empty((0,14,64,64))
#     outputs_all = np.empty((0,7,64,64))
#     model.eval()
#     with torch.no_grad():
#         for batch_idx, (inputs, targets, weights, perts) in enumerate(test_dataloader_patient):
#             inputs = inputs.to(device)
#             perts = perts.to(device)
#             perts = perts[:,2772:]
#             outputs, perts_out = model(inputs,perts)
            
#             inputs = inputs.detach().cpu().numpy()
#             inputs = std_ground * inputs + mean_ground
#             outputs = outputs.detach().cpu().numpy()
#             outputs = std_ground * outputs + mean_ground
            
#             inputs_all = np.append(inputs_all,inputs, axis=0)
#             outputs_all = np.append(outputs_all,outputs, axis=0)
#             for index_input in range(len(inputs)):
#                 masks = inputs[index_input][7:,:,:]
#                 masks = std_mask * masks + mean_mask
                
#                 if np.count_nonzero(masks.sum(axis=(1,2))) > 1:
#                     first_idx = (masks.sum(axis=(1,2))!=0).argmax(axis=0)
#                     second_idx = first_idx + 1
                    
#                     inputs_p = inputs[index_input][:7,:,:]
#                     outputs_p = outputs[index_input]
#                     input_contrast = inputs_p[second_idx].max()/inputs_p[first_idx].max()
#                     inputs_contrast.append(input_contrast)
#                     output_contrast = outputs_p[second_idx].max()/outputs_p[first_idx].max()
#                     outputs_contrast.append(output_contrast)
#                     if np.count_nonzero(masks.sum(axis=(1,2))) > 2:
#                         first_idx = (masks.sum(axis=(1,2))!=0).argmax(axis=0)
#                         second_idx = first_idx + 2
                        
#                         input_contrast = inputs_p[second_idx].max()/inputs_p[first_idx].max()
#                         third_inputs_contrast.append(input_contrast)
#                         output_contrast = outputs_p[second_idx].max()/outputs_p[first_idx].max()
#                         third_outputs_contrast.append(output_contrast)
#                 else:
#                     continue
#     return  inputs_contrast,outputs_contrast,third_inputs_contrast,third_outputs_contrast,inputs_all,outputs_all

# # clinical example    
# test_patient = scipy.io.loadmat('C:/Users/mingh/Box/3D DOT/clinical_shallow_example/saved_data/p41.mat')
# recons_patient = np.array(test_patient['volume78'])
# mask_patient = np.array(test_patient['maskall'])
# recons_patient_test = np.tile(recons_patient,(64,1,1,1))
# recons_patient = (recons_patient - mean_ground)/std_ground
# mask_patient = (mask_patient - mean_mask)/std_mask
# pert_patient = np.ones((len(recons_patient), 2772))
# test_dataset_patient = Maskdata2D(recons_patient, recons_patient, recons_patient, pert_patient, mask_patient)
# test_dataloader_patient = DataLoader(test_dataset_patient, batch_size=batch_size, shuffle=False)
# (inputs_patient, targets_patient,  weights_patient, perts_patient) = next(iter(test_dataloader_patient))
# inputs = inputs_patient.to(device)
# perts = perts_patient.to(device)
# perts = perts[:,2772:]
# outputs, perts_out = model(inputs,perts)

import scipy.io as io

# inputs_contrast,outputs_contrast,third_inputs_contrast,third_outputs_contrast,inputs_all,outputs_all = calculate_clinical()
