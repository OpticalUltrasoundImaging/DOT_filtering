# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 18:28:57 2024

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
from model_func import UNet2D_mask
from model_func import Maskdata2D, UNet2D_mask
# import VGG model for VGGloss
import torchvision.models as models
vgg = models.vgg16(pretrained=True).features[:6]
def perceptual_loss(x, y):
    features_x = vgg(x)
    features_y = vgg(y)
    return torch.nn.functional.mse_loss(features_x, features_y)


# Create the U-Net model
in_channels = 14
out_channels = 7
model = UNet2D_mask(in_channels, out_channels)

criterion = nn.MSELoss(reduction='none')
# criterion = pytorch_ssim.ssim()
criterion_pert = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=2e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=100,threshold=0.01, threshold_mode='abs',verbose =True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg.to(device)

# %% initial training
def train_nn(model, train_data, test_data):
    epoch_loss_all = []
    avg_loss_all = []
    num_epochs = 200
    
    model.to(device)
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, targets, weights, perts) in enumerate(train_data):
            inputs = inputs.to(device)
            targets = targets.to(device)
            weights = weights.to(device)
            perts = perts.to(device)
            actual_perts = perts[:,:2772]
            geom = perts[:,2772:]
            
            weight_matrix = torch.where(weights, 98, 2)
            # Forward pass
            outputs, pert_outputs = model(inputs, geom)

            loss_img = criterion(outputs, targets)
            pert_loss = 500 * criterion_pert(pert_outputs, actual_perts)
            # loss = torch.mean(loss_img * weight_matrix) + pert_loss

            outputs_img = outputs.squeeze(1)[:,:,2:-2,2:-2].reshape((len(targets),3,100,84))
            targets_img = targets.squeeze(1)[:,:,2:-2,2:-2].reshape((len(targets),3,100,84))
            loss_vgg = perceptual_loss(outputs_img, targets_img)
            loss = torch.mean(loss_img * weight_matrix) + loss_vgg + pert_loss
            
            # loss_img_ssim = 100 * pytorch_ssim.ssim(outputs.squeeze(), targets.squeeze(),weight_matrix.squeeze())
            # loss = torch.mean(loss_img * weight_matrix) + pert_loss + loss_img_ssim
            # loss = loss_img + criterion_pert(pert_outputs, actual_perts) 

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Print batch loss
            if (batch_idx+1) % 40 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch+1, num_epochs, batch_idx+1, len(train_data), loss.item()))
                print('Epoch [{}/{}], Step [{}/{}], Pert Loss: {:.4f}'
                      .format(epoch+1, num_epochs, batch_idx+1, len(train_data), pert_loss.item()))
                scheduler.step(loss)
                
            torch.cuda.empty_cache()
            
        
        # Print epoch loss
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Elapsed time: ", elapsed_time)
        epoch_loss = running_loss / len(train_data)
        epoch_loss_all.append(epoch_loss)
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, epoch_loss))
        
        model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets, weights, perts) in enumerate(test_data):
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

                outputs_img = outputs.squeeze(1)[:,:,2:-2,2:-2].reshape((len(outputs),3,100,84))
                targets_img = targets.squeeze(1)[:,:,2:-2,2:-2].reshape((len(targets),3,100,84))
                loss_vgg = perceptual_loss(outputs_img, targets_img)
                loss = torch.mean(loss_img * weight_matrix) + loss_vgg + pert_loss
                
                # loss_img_ssim = 100 * pytorch_ssim.ssim(outputs.squeeze(), targets.squeeze(),weight_matrix.squeeze())
                # loss = torch.mean(loss_img * weight_matrix) + pert_loss + loss_img_ssim
                # loss = loss_img + criterion_pert(pert_outputs, actual_perts)#+  0.05*back_reduce_loss(outputs)
                
                running_loss += loss.item()
                
                        
            # Print average loss over the validation dataset
            avg_loss = running_loss / len(test_data)
            avg_loss_all.append(avg_loss)
            print('Lowest testing Loss: {:.4f}'.format(avg_loss))
    return model, epoch_loss_all, avg_loss_all

# %% evaluate the model
def eval_nn(model, data, std_ground_val, mean_ground_val):
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
            
            images = inputs[0].detach().cpu().numpy()
            images = std_ground_val * images + mean_ground_val
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
            
            fig, axes = plt.subplots(3, 3, figsize=(12, 12))
            plt.title('mask')
            min_value = 0
            max_value = 0.2
            for i, ax in enumerate(axes.flatten()):
                if i < 7:
                    im = ax.imshow(np.transpose(images[i+7]),vmin=min_value, vmax=max_value,origin='lower',cmap='jet')  # Display grayscale images
                    ax.axis('off')  # Turn off axis labels
                    
                    cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
                    cbar = fig.colorbar(im, cax=cax)

                else:
                    ax.axis('off')  # Turn off axis labels for the last subplot
            plt.subplots_adjust(hspace=0.4, wspace=0.4)  
            plt.show()
            
            images = targets[0].detach().cpu().numpy()
            images = std_ground_val * images + mean_ground_val
            fig, axes = plt.subplots(3, 3, figsize=(12, 12))
            plt.title('Ground truth')
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
            
            images = outputs[0].detach().cpu().numpy()
            images = std_ground_val * images + mean_ground_val
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
            
            fig, axes = plt.subplots(3, 3, figsize=(12, 12))
            plt.title('Weight mask')
            images = weight_matrix[0].detach().cpu().numpy()
            min_value = np.min(images)
            max_value = np.max(images)
            for i, ax in enumerate(axes.flatten()):
                if i < 7:
                    im = ax.imshow(np.transpose(images[i]),vmin=min_value, vmax=max_value,origin='lower')  # Display grayscale images
                    ax.axis('off')  # Turn off axis labels
                    
                    cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
                    cbar = fig.colorbar(im, cax=cax)

                else:
                    ax.axis('off')  # Turn off axis labels for the last subplot
            plt.subplots_adjust(hspace=0.4, wspace=0.4)     
            plt.show()
                    
        # Print average loss over the validation dataset
        avg_loss = running_loss / len(data)
        print('Average Loss: {:.4f}'.format(avg_loss))
        
        return pert_output_store, img_output_store
    
# %% fine tune the model
def finetune_nn(model, train_data, test_data):
    epoch_loss_all = []
    avg_loss_all = []
    num_epochs = 200
    
    # count = 0
    # for param in model.parameters():
    #     count = count+1
    #     param.requires_grad = False
    #     if count == 6:
    #         break 

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5, weight_decay=2e-5)
    model.to(device)
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, targets, weights, perts) in enumerate(train_data):
            inputs = inputs.to(device)
            targets = targets.to(device)
            weights = weights.to(device)
            perts = perts.to(device)
            actual_perts = perts[:,:2772]
            geom = perts[:,2772:]
            
            weight_matrix = torch.where(weights, 98, 2)
            # Forward pass
            outputs, pert_outputs = model(inputs, geom)

            loss_img = criterion(outputs, targets)
            pert_loss = 500 * criterion_pert(pert_outputs, actual_perts)
            # loss = torch.mean(loss_img * weight_matrix) + pert_loss

            outputs_img = outputs.squeeze(1)[:,:,2:-2,2:-2].reshape((len(targets),3,100,84))
            targets_img = targets.squeeze(1)[:,:,2:-2,2:-2].reshape((len(targets),3,100,84))
            loss_vgg = perceptual_loss(outputs_img, targets_img)
            loss = torch.mean(loss_img * weight_matrix) + loss_vgg + pert_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Print batch loss
            if (batch_idx+1) % 40 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Fine tune Loss: {:.4f}'
                      .format(epoch+1, num_epochs, batch_idx+1, len(train_data), loss.item()))
                print('Epoch [{}/{}], Step [{}/{}], Fine tune Pert Loss: {:.4f}'
                      .format(epoch+1, num_epochs, batch_idx+1, len(train_data), pert_loss.item()))
                scheduler.step(loss)
            
            torch.cuda.empty_cache()

        # Print epoch loss
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Elapsed time: ", elapsed_time)
        epoch_loss = running_loss / len(train_data)
        epoch_loss_all.append(epoch_loss)
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, epoch_loss))
        
        model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets, weights, perts) in enumerate(test_data):
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

                outputs_img = outputs.squeeze(1)[:,:,2:-2,2:-2].reshape((len(outputs),3,100,84))
                targets_img = targets.squeeze(1)[:,:,2:-2,2:-2].reshape((len(targets),3,100,84))
                loss_vgg = perceptual_loss(outputs_img, targets_img)
                loss = torch.mean(loss_img * weight_matrix) + loss_vgg + pert_loss
                
                # loss_img_ssim = 100 * pytorch_ssim.ssim(outputs.squeeze(), targets.squeeze(),weight_matrix.squeeze())
                # loss = torch.mean(loss_img * weight_matrix) + pert_loss + loss_img_ssim
                # loss = loss_img + criterion_pert(pert_outputs, actual_perts)#+  0.05*back_reduce_loss(outputs)
                
                running_loss += loss.item()
                
                        
            # Print average loss over the validation dataset
            avg_loss = running_loss / len(test_data)
            avg_loss_all.append(avg_loss)
            print('Average testing Loss: {:.4f}'.format(avg_loss))
        if epoch == 0:
            lowest_loss = avg_loss
        else:
            if avg_loss < lowest_loss:
                saveModel()
                lowest_loss = avg_loss
        print('Average testing Loss: {:.4f}'.format(lowest_loss))
                
    return model, epoch_loss_all, avg_loss_all

def saveModel():
    path = "./models/DOT_filter_model.pth"
    torch.save(model.state_dict(), path)
    
    
def loading_data():
    # %%
    # load multiple target scan (can be removed)
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
    recons_ft = valdata['volume7all']
    groundtruthes_ft = valdata['real_reconall']
    weight_matrix_ft = valdata['weight_matrixall']
    perturbations_ft = valdata['data_forsaveall']
    mua0all_ft = valdata['mua0all']
    mus0all_ft = valdata['mus0all']
    radius1all_ft = valdata['radius1all']
    radius2all_ft = valdata['radius2all']
    radius3all_ft = valdata['radius3all']
    rtg1all_ft = valdata['rtg1all']
    rtg2all_ft = valdata['rtg2all']
    rtg3all_ft = valdata['rtg3all']
    ua1all_ft = valdata['ua1all']
    ua2all_ft = valdata['ua2all']
    ua3all_ft = valdata['ua3all']
    # hist_ft = valdata['hist_all']
    maskall_ft = valdata['maskall']
    perts_ft = np.concatenate((perturbations_ft[:,:,0], perturbations_ft[:,:,1], perturbations_ft[:,:,2]), axis = 1).astype(np.float32)
    weight_matrix_ft = weight_matrix_ft + 1

    X_train_ft, X_test_ft, y_train_ft, y_test_ft, weight_train_ft, weight_test_ft, pert_train_ft, pert_test_ft, mua0train_ft, mua0test_ft, mus0train_ft, mus0test_ft, \
        radius1train_ft, radius1test_ft, radius2train_ft, radius2test_ft, radius3train_ft, radius3test_ft, rtg1train_ft, rtg1test_ft, rtg2train_ft, rtg2test_ft, \
            rtg3train_ft, rtg3test_ft, ua1train_ft, ua1test_ft, ua2train_ft, ua2test_ft, ua3train_ft, ua3test_ft, mask_train_ft, mask_test_ft \
                = train_test_split(recons_ft, groundtruthes_ft, weight_matrix_ft, perts_ft, mua0all_ft, mus0all_ft, radius1all_ft, radius2all_ft, radius3all_ft, \
                                                 rtg1all_ft, rtg2all_ft, rtg3all_ft, ua1all_ft, ua2all_ft, ua3all_ft, maskall_ft, test_size=0.15, random_state=42) # usually 21
    # %%
    # pre-processing for training
    noise = np.random.normal(0,0.015, X_train.shape)
    X_train = X_train + noise
    mean_recon = np.mean(recons)
    std_recon = np.std(recons)
    mean_ground = np.mean(groundtruthes)
    std_ground = np.std(groundtruthes)

    mean_mask = np.mean(maskall)
    std_mask = np.std(maskall)
    X_train = (X_train - mean_ground)/std_ground
    X_test = (X_test - mean_ground)/std_ground
    y_train = (y_train - mean_ground)/std_ground
    y_test = (y_test - mean_ground)/std_ground

    mask_train = (mask_train - mean_mask)/std_mask
    mask_test = (mask_test - mean_mask)/std_mask
    threshold = (0.08 - mean_ground)/std_ground
    # %%
    # pre-processing for fine-tune
    noise_ft = np.random.normal(0,0.0015, X_train_ft.shape)
    X_train_ft = X_train_ft + noise_ft
    mean_recon_ft = np.mean(recons_ft)
    std_recon_ft = np.std(recons_ft)
    mean_ground_ft = np.mean(groundtruthes_ft)
    std_ground_ft = np.std(groundtruthes_ft)

    mean_mask_ft = np.mean(maskall_ft)
    std_mask_ft = np.std(maskall_ft)
    X_train_ft = (X_train_ft - mean_ground_ft)/std_ground_ft
    X_test_ft = (X_test_ft - mean_ground_ft)/std_ground_ft
    y_train_ft = (y_train_ft - mean_ground_ft)/std_ground_ft
    y_test_ft = (y_test_ft - mean_ground_ft)/std_ground_ft

    mask_train_ft = (mask_train_ft - mean_mask_ft)/std_mask_ft
    mask_test_ft = (mask_test_ft - mean_mask_ft)/std_mask_ft
    # %%
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
    augmented_inputs_ft = []
    augmented_targets_ft = []
    augmented_weights_ft = []
    augmented_masks_ft = []

    for i in range(len(X_train_ft)):
        input_sample_ft = X_train_ft[i]
        target_sample_ft = y_train_ft[i]
        weight_sample_ft = weight_train_ft[i]
        weight_sample_ft[weight_sample_ft == 2] = 0
        mask_sample_ft = mask_train_ft[i]

        # augmented_input_sample = np.copy(input_sample)
        # augmented_target_sample = np.copy(target_sample)
        
        sample_ft = np.dstack((input_sample_ft, target_sample_ft, weight_sample_ft, mask_sample_ft))
        augmented_sample_ft = augmentation_transforms(sample_ft)
        augmented_sample_ft = np.transpose(augmented_sample_ft, (1, 2, 0))

        # augmented_inputs.append(input_sample)
        augmented_inputs_ft.append(augmented_sample_ft[:,:,:7])
        # augmented_targets.append(target_sample)
        augmented_targets_ft.append(augmented_sample_ft[:,:,7:14])
        weight_auged_ft = augmented_sample_ft[:,:,14:21]
        weight_auged_ft[weight_auged_ft == 2] = 0
        augmented_weights_ft.append(weight_auged_ft)
        augmented_masks_ft.append(augmented_sample_ft[:,:,21:])

    weight_test_ft[weight_test_ft == 2] = 0  
    weight_test_ft = weight_test_ft.astype(bool)
    weight_test[weight_test == 2] = 0
    weight_test = weight_test.astype(bool)
    augmented_inputs = np.array(augmented_inputs)
    augmented_targets = np.array(augmented_targets)
    augmented_weights = np.array(augmented_weights).astype(bool)
    augmented_masks = np.array(augmented_masks)
    augmented_inputs_ft = np.array(augmented_inputs_ft)
    augmented_targets_ft = np.array(augmented_targets_ft)
    augmented_weights_ft = np.array(augmented_weights_ft).astype(bool)
    augmented_masks_ft = np.array(augmented_masks_ft)
    # %%

    train_dataset = Maskdata2D(augmented_inputs, augmented_targets, augmented_weights, pert_train, augmented_masks)
    test_dataset = Maskdata2D(X_test, y_test, weight_test, pert_test, mask_test)
    val_dataset_train = Maskdata2D(augmented_inputs_ft, augmented_targets_ft, augmented_weights_ft, pert_train_ft, augmented_masks_ft)
    val_dataset_test = Maskdata2D(X_test_ft, y_test_ft, weight_test_ft, pert_test_ft, mask_test_ft)
    
    return train_dataset, test_dataset, val_dataset_train, val_dataset_test, std_ground_ft, mean_ground_ft, mean_ground, std_ground, mean_mask_ft, std_mask_ft