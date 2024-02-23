# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 14:32:39 2023

@author: mingh
"""
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
# from scipy import ndimage
import torch.nn.functional as F
import cv2 as cv
from tqdm import tqdm
from skimage.transform import resize


# %% attention module
class CoTAttention(nn.Module):

    def __init__(self, dim=512,kernel_size=3):
        super().__init__()
        self.dim=dim
        self.kernel_size=kernel_size

        self.key_embed=nn.Sequential(
            nn.Conv2d(dim,dim,kernel_size=kernel_size,padding=kernel_size//2,groups=4,bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.value_embed=nn.Sequential(
            nn.Conv2d(dim,dim,1,bias=False),
            nn.BatchNorm2d(dim)
        )

        factor=4
        self.attention_embed=nn.Sequential(
            nn.Conv2d(2*dim,2*dim//factor,1,bias=False),
            nn.BatchNorm2d(2*dim//factor),
            nn.ReLU(),
            nn.Conv2d(2*dim//factor,kernel_size*kernel_size*dim,1)
        )


    def forward(self, x):
        bs,c,h,w=x.shape
        k1=self.key_embed(x) #bs,c,h,w
        v=self.value_embed(x).view(bs,c,-1) #bs,c,h,w

        y=torch.cat([k1,x],dim=1) #bs,2c,h,w
        att=self.attention_embed(y) #bs,c*k*k,h,w
        att=att.reshape(bs,c,self.kernel_size*self.kernel_size,h,w)
        att=att.mean(2,keepdim=False).view(bs,c,-1) #bs,c,h*w
        k2=F.softmax(att,dim=-1)*v
        k2=k2.view(bs,c,h,w)

        return k1+k2
# %% dataset with fine mesh    
class Maskdata2D(Dataset):
    def __init__(self, inputs, targets, weights, perts, masks):
        self.inputs = inputs
        self.targets = targets
        self.weights = weights
        self.perts = perts
        self.masks = masks
        
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        # Precompute or cache transformations here
        self.preprocessed_inputs = [self._preprocess_input(input_data) for input_data in tqdm(self.inputs)]
        self.preprocessed_targets = [self._preprocess_target(target_data) for target_data in tqdm(self.targets)]
        self.preprocessed_weights = [self._preprocess_weight(weight_data) for weight_data in tqdm(self.weights)]
        self.preprocessed_masks = [self._preprocess_mask(mask_data) for mask_data in tqdm(self.masks)]

    def _preprocess_input(self, input_data):
        # Add your preprocessing code here
        input_data = resize(input_data, (64, 64, 7))
        # input_data = ndimage.zoom(input_data, (64/33, 64/65, 1))
        return self.transform(input_data.astype(np.float32))

    def _preprocess_target(self, target_data):
        # Add your preprocessing code here
        target_data = cv.blur(target_data, (5, 5))
        # target_data = cv.GaussianBlur(target_data, (5, 5), 0)
        target_data = resize(target_data, (64, 64, 7))
        # target_data = ndimage.zoom(target_data, (64/33, 64/65, 1))
        return self.transform(target_data.astype(np.float32))
    
    def _preprocess_weight(self, weight_data):
        # Add your preprocessing code here
        weight_data = resize(weight_data, (64, 64, 7))
        return self.transform(weight_data.astype(bool))
    
    def _preprocess_mask(self, mask_data):
        # Add your preprocessing code here
        mask_data = resize(mask_data, (64, 64, 7))
        # input_data = ndimage.zoom(input_data, (64/33, 64/65, 1))
        return self.transform(mask_data.astype(np.float32))

    def __getitem__(self, index):
        input_img = self.preprocessed_inputs[index]
        target_data = self.preprocessed_targets[index]
        weight_data = self.preprocessed_weights[index]
        pert_data = self.perts[index]
        mask_data = self.preprocessed_masks[index]
        input_data = torch.cat((input_img, mask_data), 0)

        return input_data, target_data, weight_data, pert_data

    def __len__(self):
        return len(self.inputs)
    
#%% model structure
class UNet2D_mask(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet2D_mask, self).__init__()

        # Encoder
        self.encoder_conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.encoder_att1 = CoTAttention(dim=32,kernel_size=3) # CBAM(32) 
        self.encoder_relu1 = nn.ELU()
        self.encoder_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.encoder_att2 = CoTAttention(dim=64,kernel_size=3)
        self.encoder_relu2 = nn.ELU()
        self.encoder_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.encoder_att3 = CoTAttention(dim=128,kernel_size=3)
        self.encoder_relu3 = nn.ELU()
        self.encoder_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.encoder_att4 = CoTAttention(dim=256,kernel_size=3)
        self.encoder_relu4 = nn.ELU()
        self.encoder_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.encoder_att5 = CoTAttention(dim=512,kernel_size=3)
        self.encoder_relu5 = nn.ELU()
        self.encoder_pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck_conv0 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bottleneck_relu0 = nn.ELU()
        self.bottleneck_conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bottleneck_relu1 = nn.ELU()
        self.bottleneck_conv2 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bottleneck_relu2 = nn.ELU()

        # Decoder
        self.decoder_upsample0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.decoder_conv0 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.decoder_att0 = CoTAttention(dim=512,kernel_size=3)
        self.decoder_relu0 = nn.ELU()
        self.decoder_upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.decoder_conv1 = nn.Conv2d(768, 256, kernel_size=3, padding=1)
        self.decoder_att1 = CoTAttention(dim=256,kernel_size=3)
        self.decoder_relu1 = nn.ELU()
        self.decoder_upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.decoder_conv2 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.decoder_att2 = CoTAttention(dim=256,kernel_size=3)
        self.decoder_relu2 = nn.ELU()
        self.decoder_upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.decoder_conv3 = nn.Conv2d(320, 256, kernel_size=3, padding=1)
        self.decoder_att3 = CoTAttention(dim=256,kernel_size=3)
        self.decoder_relu3 = nn.ELU()
        self.decoder_conv3_1 = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        self.decoder_att3_1 = CoTAttention(dim=64,kernel_size=3)
        self.decoder_relu3_1 = nn.ELU()
        self.decoder_conv4 = nn.Conv2d(64, 16, kernel_size=3, padding=1)
        self.decoder_att4 = CoTAttention(dim=16,kernel_size=3)
        self.decoder_relu4 = nn.ELU()
        self.decoder_conv5 = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)
        self.decoder_relu5 = nn.ELU()
        self.decoder_conv6 = nn.Conv2d(14, out_channels, kernel_size=3, padding=1)
        self.decoder_relu7 = nn.ELU()
        self.decoder_conv7 = nn.Conv2d(14, out_channels, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(4096, 2772)
        self.fc_relu1 = nn.ELU()
        self.fc2 = nn.Linear(2772, 2772)
        self.fc_relu2 = nn.ELU()
        self.fc3 = nn.Linear(2772, 4096)
        self.fc_relu3 = nn.ELU()
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.25)
        
        self.end_relu = nn.ELU()

    def forward(self, x, geom):
        # Encoder
        encoder1 = self.encoder_conv1(x)
        encoder1 = self.encoder_att1(encoder1)
        encoder1 = self.encoder_relu1(encoder1)
        encoder2 = self.encoder_conv2(encoder1)
        encoder2 = self.encoder_att2(encoder2)
        encoder2 = self.encoder_relu2(encoder2)
        encoder3 = self.encoder_pool2(encoder2)
        encoder4 = self.encoder_conv3(encoder3)
        encoder4 = self.encoder_att3(encoder4)
        encoder4 = self.encoder_relu3(encoder4)
        encoder5 = self.encoder_pool3(encoder4)
        encoder6 = self.encoder_conv4(encoder5)
        encoder6 = self.encoder_att4(encoder6)
        encoder6 = self.encoder_relu4(encoder6)
        encoder7 = self.encoder_pool4(encoder6)
        encoder8 = self.encoder_conv5(encoder7)
        encoder8 = self.encoder_att5(encoder8)
        encoder8 = self.encoder_relu5(encoder8)
        encoder9 = self.encoder_pool5(encoder8)

        # Output perturbation as pert_output2
        bottleneck = self.bottleneck_conv0(encoder9)
        bottleneck = self.bottleneck_relu0(bottleneck)
        pert_output = bottleneck.view(len(bottleneck),-1)
        pert_output1 = self.fc1(pert_output)
        pert_output1 = self.fc_relu1(pert_output1)
        pert_output1 = self.dropout1(pert_output1)
        pert_output2 = self.fc2(pert_output1)
        pert_output2 = self.fc_relu2(pert_output2)
        pert_output3 = self.fc3(pert_output2)
        pert_output3 = self.fc_relu3(pert_output3)
        pert_output3 = self.dropout2(pert_output3)
        bottleneck1 = pert_output3.view(bottleneck.shape)
        
        # middle layer
        bottleneck2 = self.bottleneck_conv1(bottleneck1)
        bottleneck2 = self.bottleneck_relu1(bottleneck2)
        bottleneck3 = self.bottleneck_conv2(bottleneck2)
        bottleneck3 = self.bottleneck_relu2(bottleneck3)

        # Decoder
        decoder0 = self.decoder_upsample0(bottleneck3)
        decoder0 = torch.cat([decoder0, encoder8[:, :, :decoder0.shape[2], :decoder0.shape[3]]], dim=1)
        decoder0 = self.decoder_conv0(decoder0)
        decoder0 = self.decoder_att0(decoder0)
        decoder0 = self.decoder_relu0(decoder0)

        decoder1 = self.decoder_upsample1(decoder0)
        decoder1 = torch.cat([decoder1, encoder6[:, :, :decoder1.shape[2], :decoder1.shape[3]]], dim=1)
        decoder1 = self.decoder_conv1(decoder1)
        decoder1 = self.decoder_att1(decoder1)
        decoder1 = self.decoder_relu1(decoder1)
        decoder2 = self.decoder_upsample2(decoder1)
        decoder2 = torch.cat([decoder2, encoder4[:, :, :decoder2.shape[2], :decoder2.shape[3]]], dim=1)
        decoder2 = self.decoder_conv2(decoder2)
        decoder2 = self.decoder_att2(decoder2)
        decoder2 = self.decoder_relu2(decoder2)
        decoder3 = self.decoder_upsample3(decoder2)
        decoder3 = torch.cat([decoder3, encoder2[:, :, :decoder3.shape[2], :decoder3.shape[3]]], dim=1)
        decoder4 = self.decoder_conv3(decoder3)
        decoder4 = self.decoder_att3(decoder4)
        decoder4 = self.decoder_relu3(decoder4)
        decoder4 = self.decoder_conv3_1(decoder4)
        decoder4 = self.decoder_att3_1(decoder4)
        decoder4 = self.decoder_relu3_1(decoder4)
        decoder5 = self.decoder_conv4(decoder4)
        decoder5 = self.decoder_att4(decoder5)
        decoder5 = self.decoder_relu4(decoder5)
        decoder6 = self.decoder_conv5(decoder5)
        decoder6 = self.decoder_relu5(decoder6)
        decoder6 = torch.cat([decoder6, x[:,7:,:,:]], dim=1)
        output = self.decoder_conv6(decoder6)
        output = self.decoder_relu7(output)
        output = torch.cat([output, x[:,7:,:,:]], dim=1)
        output = self.decoder_conv7(output)
        output = self.end_relu(output)
        return output, pert_output2

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()
        
        # Encoder
        self.encoder_conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, padding=1)
        self.encoder_CBAM1 = CBAM(32)
        # self.encoder_acmix1 = ACmix(in_planes=32, out_planes=32)
        # self.encoder_relu1 = nn.ReLU(inplace=True)
        self.encoder_relu1 = nn.ELU()
        self.encoder_conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.encoder_CBAM2 = CBAM(64)
        # self.encoder_relu2 = nn.ReLU(inplace=True)
        self.encoder_relu2 = nn.ELU()
        self.encoder_pool2 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2),padding=(0,0,0))
        self.encoder_conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.encoder_CBAM3 = CBAM(128)
        # self.encoder_relu3 = nn.ReLU(inplace=True)
        self.encoder_relu3 = nn.ELU()
        self.encoder_pool3 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2),padding=(0,0,0))
        self.encoder_conv4 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.encoder_CBAM4 = CBAM(256)
        # self.encoder_relu4 = nn.ReLU(inplace=True)
        self.encoder_relu4 = nn.ELU()
        self.encoder_pool4 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2),padding=(0,0,0))
        self.encoder_conv5 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.encoder_CBAM5 = CBAM(512)
        # self.encoder_relu5 = nn.ReLU(inplace=True)
        self.encoder_relu5 = nn.ELU()
        self.encoder_pool5 = nn.MaxPool3d(kernel_size=(3,2,2), stride=(1,2,2),padding=(1,0,0))
        
        # Bottleneck
        self.bottleneck_conv0 = nn.Conv3d(512, 256, kernel_size=3, padding=1)
        # self.bottleneck_relu0 = nn.ReLU(inplace=True)
        self.bottleneck_relu0 = nn.ELU()
        self.bottleneck_conv1 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        # self.bottleneck_relu1 = nn.ReLU(inplace=True)
        self.bottleneck_relu1 = nn.ELU()
        self.bottleneck_conv2 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        # self.bottleneck_relu2 = nn.ReLU(inplace=True)
        self.bottleneck_relu2 = nn.ELU()
        
        # Decoder
        self.decoder_upsample0 = nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=True)
        self.decoder_conv0 = nn.Conv3d(1024, 512, kernel_size=3, padding=1)  # Concatenation channel: 128 + 64 = 192
        self.decoder_CBAM0 = CBAM(512)
        # self.decoder_relu0 = nn.ReLU(inplace=True)
        self.decoder_relu0 = nn.ELU()
        self.decoder_upsample1 = nn.Upsample(scale_factor=(2,2,2), mode='trilinear', align_corners=True)
        self.decoder_conv1 = nn.Conv3d(768, 256, kernel_size=3, padding=1)  # Concatenation channel: 128 + 64 = 192
        self.decoder_CBAM1 = CBAM(256)
        # self.decoder_relu1 = nn.ReLU(inplace=True)
        self.decoder_relu1 = nn.ELU()
        # self.decoder_conv1_1 = nn.Conv3d(256, 256, kernel_size=3, padding=1)  # Concatenation channel: 128 + 64 = 192
        # self.decoder_relu1_1 = nn.ReLU(inplace=True)
        # self.decoder_relu1_1 = nn.LeakyReLU()
        self.decoder_upsample2 = nn.Upsample(scale_factor=(2,2,2), mode='trilinear', align_corners=True)
        self.decoder_conv2 = nn.Conv3d(384, 256, kernel_size=3, padding=1)  # Concatenation channel: 128 + 64 = 192
        self.decoder_CBAM2 = CBAM(256)
        # self.decoder_relu2 = nn.ReLU(inplace=True)
        self.decoder_relu2 = nn.ELU()
        # self.decoder_conv2_1 = nn.Conv3d(256, 256, kernel_size=3, padding=1)  # Concatenation channel: 128 + 64 = 192
        # self.decoder_relu2_1 = nn.ReLU(inplace=True)
        # self.decoder_relu2_1 = nn.LeakyReLU()
        self.decoder_upsample3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.decoder_conv3 = nn.Conv3d(320, 256, kernel_size=3, padding=1)  # Concatenation channel: 128 + 64 = 192
        self.decoder_CBAM3 = CBAM(256)
        # self.decoder_relu3 = nn.ReLU(inplace=True)
        self.decoder_relu3 = nn.ELU()
        self.decoder_conv3_1 = nn.Conv3d(256, 64, kernel_size=3, padding=1)  # Concatenation channel: 128 + 64 = 192
        self.decoder_CBAM3_1 = CBAM(64)
        # self.decoder_relu3_1 = nn.ReLU(inplace=True)
        self.decoder_relu3_1 = nn.ELU()
        self.decoder_conv4 = nn.Conv3d(64, 16, kernel_size=3, padding=1)
        self.decoder_CBAM4 = CBAM(16)
        # self.decoder_relu4 = nn.ReLU(inplace=True)
        self.decoder_relu4 = nn.ELU()
        self.decoder_conv5 = nn.Conv3d(16, out_channels, kernel_size=3, padding=1)
        
        # self.fc0 = nn.Linear(5482, 4096) # 4096 + 1386 = 5482
        # self.fc_relu0 = nn.LeakyReLU()
        self.fc1 = nn.Linear(4096, 2772) # 4096 + 1386 = 5482
        self.fc_relu1 = nn.ELU()
        self.fc2 = nn.Linear(2772, 2772)
        self.fc_relu2 = nn.ELU()
        self.fc3 = nn.Linear(2772, 4096)
        self.fc_relu3 = nn.ELU()
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.25)
        
        self.end_relu = nn.ELU()

    def forward(self, x, geom):
        # Encoder
        encoder1 = self.encoder_conv1(x)
        encoder1 = self.encoder_CBAM1(encoder1)
        # encoder1 = self.encoder_acmix1(encoder1)
        encoder1 = self.encoder_relu1(encoder1)
        encoder2 = self.encoder_conv2(encoder1)
        encoder2 = self.encoder_CBAM2(encoder2)
        encoder2 = self.encoder_relu2(encoder2)
        encoder2 = F.pad(encoder2, (0,0,0,0,0,1), mode='replicate')
        encoder3 = self.encoder_pool2(encoder2)
        encoder4 = self.encoder_conv3(encoder3)
        encoder4 = self.encoder_CBAM3(encoder4)
        encoder4 = self.encoder_relu3(encoder4)
        # encoder4 = self.encoder_conv3_1(encoder4)
        # encoder4 = self.encoder_relu3_1(encoder4)
        encoder5 = self.encoder_pool3(encoder4)
        encoder6 = self.encoder_conv4(encoder5)
        encoder6 = self.encoder_CBAM4(encoder6)
        encoder6 = self.encoder_relu4(encoder6)
        encoder7 = self.encoder_pool4(encoder6)
        encoder8 = self.encoder_conv5(encoder7)
        encoder8 = self.encoder_CBAM5(encoder8)
        encoder8 = self.encoder_relu5(encoder8)
        encoder9 = self.encoder_pool5(encoder8)

        # Output perturbation as pert_output2
        bottleneck = self.bottleneck_conv0(encoder9)
        bottleneck = self.bottleneck_relu0(bottleneck)
        pert_output = bottleneck.view(len(bottleneck),-1)
        # pert_output = torch.cat([pert_output, geom], dim=1)
        # pert_output1 = self.fc0(pert_output)
        # pert_output1 = self.fc_relu0(pert_output1)
        pert_output1 = self.fc1(pert_output)
        pert_output1 = self.fc_relu1(pert_output1)
        pert_output1 = self.dropout1(pert_output1)
        pert_output2 = self.fc2(pert_output1)
        pert_output2 = self.fc_relu2(pert_output2)
        pert_output3 = self.fc3(pert_output2)
        pert_output3 = self.fc_relu3(pert_output3)
        pert_output3 = self.dropout2(pert_output3)
        bottleneck1 = pert_output3.view(bottleneck.shape)
        
        # middle layer
        bottleneck2 = self.bottleneck_conv1(bottleneck1)
        bottleneck2 = self.bottleneck_relu1(bottleneck2)
        bottleneck3 = self.bottleneck_conv2(bottleneck2)
        bottleneck3 = self.bottleneck_relu2(bottleneck3)

        # Decoder
        decoder0 = self.decoder_upsample0(bottleneck3)
        decoder0 = torch.cat([decoder0, encoder8[:, :, :decoder0.shape[2], :decoder0.shape[3], :decoder0.shape[4]]], dim=1)
        decoder0 = self.decoder_conv0(decoder0)
        decoder0 = self.decoder_CBAM0(decoder0)
        decoder0 = self.decoder_relu0(decoder0)
        
        decoder1 = self.decoder_upsample1(decoder0)
        decoder1 = torch.cat([decoder1, encoder6[:, :, :decoder1.shape[2], :decoder1.shape[3], :decoder1.shape[4]]], dim=1)
        decoder1 = self.decoder_conv1(decoder1)
        decoder1 = self.decoder_CBAM1(decoder1)
        decoder1 = self.decoder_relu1(decoder1)
        # decoder1 = self.decoder_conv1_1(decoder1)
        # decoder1 = self.decoder_relu1_1(decoder1)
        decoder2 = self.decoder_upsample2(decoder1)
        decoder2 = torch.cat([decoder2, encoder4[:, :, :decoder2.shape[2], :decoder2.shape[3], :decoder2.shape[4]]], dim=1)
        # decoder1 = decoder1[:,:,:7,:,:]
        decoder2 = self.decoder_conv2(decoder2)
        decoder2 = self.decoder_CBAM2(decoder2)
        decoder2 = self.decoder_relu2(decoder2)
        # decoder2 = self.decoder_conv2_1(decoder2)
        # decoder2 = self.decoder_relu2_1(decoder2)
        decoder3 = self.decoder_upsample3(decoder2)
        decoder3 = torch.cat([decoder3, encoder2[:, :, :decoder3.shape[2], :decoder3.shape[3], :decoder3.shape[4]]], dim=1)
        decoder3 = decoder3[:,:,:7,:,:]
        decoder4 = self.decoder_conv3(decoder3)
        decoder4 = self.decoder_CBAM3(decoder4)
        decoder4 = self.decoder_relu3(decoder4)
        decoder4 = self.decoder_conv3_1(decoder4)
        decoder4 = self.decoder_CBAM3_1(decoder4)
        decoder4 = self.decoder_relu3_1(decoder4)
        decoder5 = self.decoder_conv4(decoder4)
        decoder5 = self.decoder_CBAM4(decoder5)
        decoder5 = self.decoder_relu4(decoder5)
        output = self.decoder_conv5(decoder5)
        output = self.end_relu(output)
        
        return output, pert_output2

    
#%% CBAM attention block for 2d
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=4, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out