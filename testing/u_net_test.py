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

from models_backup import MyAugDataset, UNet3D, Histdata, Maskdata, CBAM, Maskdata2D, UNet2D, UNet2D_mask

 # %%               
# load single scan dataset as validation
valdata = scipy.io.loadmat('C:/Users/mingh/Box/3D DOT/simudata_wmeasurement/finetune_wshallow_wmc_wyun_twohalfball_oval_star_cube_hist.mat')
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

# for i in range(100):
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
augmented_inputs_val = np.array(augmented_inputs_val)
augmented_targets_val = np.array(augmented_targets_val)
augmented_weights_val = np.array(augmented_weights_val).astype(bool)
augmented_masks_val = np.array(augmented_masks_val)

val_dataset_train = Maskdata2D(augmented_inputs_val, augmented_targets_val, augmented_weights_val, pert_train_val, augmented_masks_val)
val_dataset_test = Maskdata2D(X_test_val, y_test_val, weight_test_val, pert_test_val, mask_test_val)

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
model.load_state_dict(torch.load("C:/Users/mingh/Box/3D DOT/models/DOT_filter_model.pth"))
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
            
            
        

def saveModel():
    path = "C:/Users/mingh/Box/3D DOT/models/DOT_filter_model.pth"
    torch.save(model.state_dict(), path)

pert_output_store, img_output_store = eval_nn(model, val_dataloader_test) 
inputs_max, grounds_max, outputs_max, inputs_contrast, grounds_contrast, outputs_contrast, third_inputs_contrast, third_grounds_contrast, third_outputs_contrast = depth_contrast()
input_acc = [inputs_max[i] / grounds_max[i] for i in range(len(inputs_max))]
output_acc = [outputs_max[i] / grounds_max[i] for i in range(len(outputs_max))]
# pert_output_store_val, img_output_store_val = eval_nn(trained_unet, val_dataloader)

test3data = scipy.io.loadmat('C:/Users/mingh/Box/3D DOT/simudata_wmeasurement/threetars.mat')
recons_three = np.array(test3data['volume7all'])
mask_three = np.array(test3data['maskall'])
recons_three_test = np.tile(recons_three,(64,1,1,1))
recons_three = (recons_three - mean_ground_val)/std_ground_val
mask_three = (mask_three - mean_mask_val)/std_mask_val
pert_three = np.ones((len(recons_three), 2772))
test_dataset_three = Maskdata2D(recons_three, recons_three, recons_three, pert_three, mask_three)
test_dataloader_three = DataLoader(test_dataset_three, batch_size=batch_size, shuffle=False)
(inputs_three, targets_three,  weights_three, perts_three) = next(iter(test_dataloader_three))
inputs = inputs_three.to(device)
perts = perts_three.to(device)
perts = perts[:,2772:]
outputs, perts_out = model(inputs,perts)

def calculate_clinical():
    test_patient = scipy.io.loadmat('C:/Users/mingh/Box/3D DOT/clinical_shallow_example/hemo/data_diagnostic_0221.mat')
    
    recons_patient = np.array(test_patient['volume_all'])
    mask_patient = np.array(test_patient['mask_all'])
    
    recons_patient = (recons_patient - mean_ground_val)/std_ground_val
    mask_patient = (mask_patient - mean_mask_val)/std_mask_val
    pert_patient = np.ones((len(recons_patient), 2772))
    test_dataset_patient = Maskdata2D(recons_patient, recons_patient, recons_patient, pert_patient, mask_patient)
    test_dataloader_patient = DataLoader(test_dataset_patient, batch_size=batch_size, shuffle=False)
    
    
    inputs_contrast = []
    outputs_contrast = []
    third_inputs_contrast = []
    third_outputs_contrast = []
    inputs_all = np.empty((0,14,64,64))
    outputs_all = np.empty((0,7,64,64))
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets, weights, perts) in enumerate(test_dataloader_patient):
            inputs = inputs.to(device)
            perts = perts.to(device)
            perts = perts[:,2772:]
            outputs, perts_out = model(inputs,perts)
            
            inputs = inputs.detach().cpu().numpy()
            inputs = std_ground_val * inputs + mean_ground_val
            outputs = outputs.detach().cpu().numpy()
            outputs = std_ground_val * outputs + mean_ground_val
            
            inputs_all = np.append(inputs_all,inputs, axis=0)
            outputs_all = np.append(outputs_all,outputs, axis=0)
            
            for index_input in range(len(inputs)):
                input_img = inputs[index_input][:7,:,:]
                output_img = outputs[index_input]
                
                images = input_img
                fig, axes = plt.subplots(3, 3, figsize=(12, 12))
                plt.title('reconstruction')
                min_value = 0
                max_value = 0.2
                for i, ax in enumerate(axes.flatten()):
                    if i < 7:
                        im = ax.imshow(np.transpose(images[i]),vmin=min_value, vmax=max_value,origin='lower',cmap='jet')  # Display grayscale images
                        ax.axis('off')  # Turn off axis labels
                        
                        cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
                        cbar = fig.colorbar(im, cax=cax)

                    else:
                        ax.axis('off')  # Turn off axis labels for the last subplot
                plt.subplots_adjust(hspace=0.4, wspace=0.4)  
                plt.show()
                
                images = output_img
                fig, axes = plt.subplots(3, 3, figsize=(12, 12))
                plt.title('Outputs')
                # min_value = np.min(images)
                # max_value = np.max(images)
                for i, ax in enumerate(axes.flatten()):
                    if i < 7:
                        im = ax.imshow(np.transpose(images[i]),vmin=min_value, vmax=max_value,origin='lower',cmap='jet')  # Display grayscale images
                        ax.axis('off')  # Turn off axis labels
                        
                        cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
                        cbar = fig.colorbar(im, cax=cax)

                    else:
                        ax.axis('off')  # Turn off axis labels for the last subplot
                plt.subplots_adjust(hspace=0.4, wspace=0.4) 
                plt.show()
            
            for index_input in range(len(inputs)):
                masks = inputs[index_input][7:,:,:]
                masks = std_mask_val * masks + mean_mask_val
                
                if np.count_nonzero(masks.sum(axis=(1,2))) > 1:
                    first_idx = (masks.sum(axis=(1,2))!=0).argmax(axis=0)
                    second_idx = first_idx + 1
                    
                    inputs_p = inputs[index_input][:7,:,:]
                    outputs_p = outputs[index_input]
                    input_contrast = inputs_p[second_idx].max()/inputs_p[first_idx].max()
                    inputs_contrast.append(input_contrast)
                    output_contrast = outputs_p[second_idx].max()/outputs_p[first_idx].max()
                    outputs_contrast.append(output_contrast)
                    if np.count_nonzero(masks.sum(axis=(1,2))) > 2:
                        first_idx = (masks.sum(axis=(1,2))!=0).argmax(axis=0)
                        second_idx = first_idx + 2
                        
                        input_contrast = inputs_p[second_idx].max()/inputs_p[first_idx].max()
                        third_inputs_contrast.append(input_contrast)
                        output_contrast = outputs_p[second_idx].max()/outputs_p[first_idx].max()
                        third_outputs_contrast.append(output_contrast)
                else:
                    continue
    return  inputs_contrast,outputs_contrast,third_inputs_contrast,third_outputs_contrast,inputs_all,outputs_all
    
test_patient = scipy.io.loadmat('C:/Users/mingh/Box/3D DOT/clinical_shallow_example/saved_data/p41.mat')
recons_patient = np.array(test_patient['volume78'])
mask_patient = np.array(test_patient['maskall'])
recons_patient_test = np.tile(recons_patient,(64,1,1,1))
recons_patient = (recons_patient - mean_ground_val)/std_ground_val
mask_patient = (mask_patient - mean_mask_val)/std_mask_val
pert_patient = np.ones((len(recons_patient), 2772))
test_dataset_patient = Maskdata2D(recons_patient, recons_patient, recons_patient, pert_patient, mask_patient)
test_dataloader_patient = DataLoader(test_dataset_patient, batch_size=batch_size, shuffle=False)
(inputs_patient, targets_patient,  weights_patient, perts_patient) = next(iter(test_dataloader_patient))
inputs = inputs_patient.to(device)
perts = perts_patient.to(device)
perts = perts[:,2772:]
outputs, perts_out = model(inputs,perts)
# outputs_img = np.squeeze(outputs.detach().cpu().numpy())
# perts_out = perts_out.detach().cpu().numpy()
import scipy.io as io

inputs_contrast,outputs_contrast,third_inputs_contrast,third_outputs_contrast,inputs_all,outputs_all = calculate_clinical()
