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
    path = "C:/Users/mingh/Box/3D DOT/models/DOT_filter_model.pth"
    torch.save(model.state_dict(), path)
    