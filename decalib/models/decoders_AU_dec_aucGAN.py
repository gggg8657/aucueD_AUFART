# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import torch
import torch.nn as nn

# class AUIntegratedDecoder(nn.Module):
#     def __init__(self, deca, au_dim):
#         super(AUIntegratedDecoder, self).__init__()
#         self.deca = deca
#         self.au_dim = au_dim

#     def forward(self, images, au_param):
#         # DECA 인코더를 사용하여 기본 파라미터 추출
#         codedict = self.deca.encode(images, use_detail=True)
        
#         # AU 파라미터와 디테일 코드 결합
#         detail_code = codedict['detail']
#         combined_code = torch.cat((detail_code, au_params), dim=1)
        
#         # DECA 디코더를 사용하여 주름 생성
#         displacement_map = self.deca.D_detail(combined_code)

#         return displacement_map

class Generator(nn.Module):
    def __init__(self, latent_dim=100, out_channels=1, out_scale=0.01, sample_mode = 'bilinear'):
        super(Generator, self).__init__()
        self.out_scale = out_scale
        
        self.init_size = 32 // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2, mode=sample_mode), #16
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode=sample_mode), #32
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode=sample_mode), #64
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode=sample_mode), #128
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode=sample_mode), #256
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, out_channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img*self.out_scale


class Generator2(nn.Module):
    def __init__(self, latent_dim=100, au_dim=41, out_channels=1, out_scale=0.01, sample_mode='bilinear'):
        super(Generator2, self).__init__()
        self.out_scale = out_scale
        self.latent_dim = latent_dim
        self.au_dim = au_dim
        
        self.init_size = 32 // 4  # Initial size before upsampling
        # latent_dim + au_dim으로 입력 차원 변경
        self.l1 = nn.Sequential(nn.Linear(latent_dim + au_dim, 128 * self.init_size ** 2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2, mode=sample_mode), # 16
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode=sample_mode), # 32
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode=sample_mode), # 64
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode=sample_mode), # 128
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode=sample_mode), # 256
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, out_channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise):
        # noise = torch.cat((noise, au_params), dim=1)
        out = self.l1(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img*self.out_scale
    
class ConditionalGenerator(nn.Module):
    def __init__(self, condition_dim):
        super(ConditionalGenerator, self).__init__()
        self.encoder = Encoder()
        self.decoder = GANDecoder()
        self.fc = nn.Linear(condition_dim, 128)

    def forward(self, x, condition):
        condition_feat = self.fc(condition).view(-1, 128, 1, 1)
        encoded_feat = self.encoder(x)
        combined_feat = torch.cat([encoded_feat, condition_feat], dim=1)
        output = self.decoder(combined_feat)
        return output

class GANDecoder(nn.Module):#conditional Decoder
    def __init__(self):
        super(GANDecoder, self).__init__()
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(181, 256, kernel_size=3, padding=1),  # 256 x 2 x 2
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # 512 x 4 x 4
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),  # 256 x 8 x 8
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # 128 x 16 x 16
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 64 x 32 x 32
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # 32 x 64 x 64
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  # 16 x 128 x 128
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),  # 8 x 256 x 256
            nn.InstanceNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),  # 1 x 256 x 256
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)
    
class ConditionalDiscriminator(nn.Module):
    def __init__(self):
        super(ConditionalDiscriminator, self).__init__()
        self.l1 = nn.Conv2d(27*512, 41, kernel_size=1)
        self.model = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, stride=2, padding=1),  # 64 x 112 x 112
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 128 x 56 x 56
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 256 x 28 x 28
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # 512 x 28 x 28
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)  # 1 x 28 x 28
            ,nn.Sigmoid()
        )
    
    def forward(self, img, condition):
        condition = condition.squeeze(0)
        # print(condition.shape)
        condition = condition.view(condition.size(0),3,9,512)
        condition = F.interpolate(condition, size=(224,224), mode='bilinear', align_corners=False)
        
        # print(condition.shape)
        # condition = condition.view(condition.size(0), 1, 1, 1).repeat(1, 1, img.size(2), img.size(3))
        d_in = torch.cat((img, condition), 1)
        return self.model(d_in)
    
import torch.nn.functional as F
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)  # 64 x 128 x 128
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 128 x 64 x 64
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # 256 x 32 x 32
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # 512 x 16 x 16
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)  # 1024 x 8 x 8
        self.conv6 = nn.Conv2d(1024, 512, kernel_size=3, stride=2, padding=1)  # 512 x 4 x 4
        self.conv7 = nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1)  # 256 x 2 x 2
        self.conv8 = nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1)  # 128 x 1 x 1
        # self.conv9 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)  # 64 x 1 x 1
        # self.conv8 = nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1)  # 64 x 1 x 1
        # self.conv9 = nn.Conv2d(64, 181, kernel_size=1, stride=1, padding=0)  # 181 x 1 x 1
        
        # self.norm1 = nn.InstanceNorm2d(64)
        self.norm2 = nn.InstanceNorm2d(128)
        self.norm3 = nn.InstanceNorm2d(256)
        self.norm4 = nn.InstanceNorm2d(512)
        self.norm5 = nn.InstanceNorm2d(1024)
        self.norm6 = nn.InstanceNorm2d(512)
        self.norm7 = nn.InstanceNorm2d(256)
        # self.norm8 = nn.InstanceNorm2d(128)
        
    def forward(self, x):
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)  # Resize 256x256
        x = F.relu(self.conv1(x))
        x = F.leaky_relu(self.norm2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.norm3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.norm4(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.norm5(self.conv5(x)), 0.2)
        x = F.leaky_relu(self.norm6(self.conv6(x)), 0.2)
        x = F.leaky_relu(self.norm7(self.conv7(x)), 0.2)
        x = F.leaky_relu(self.conv8(x), 0.2)
        # x = F.leaky_relu(self.norm9(self.conv9(x)), 0.2)
        return x
class Generator3(nn.Module):
    def __init__(self, latent_dim=100, out_channels=1, out_scale=0.01, sample_mode = 'bilinear'):
        super(Generator3, self).__init__()
        self.out_scale = out_scale
        # self.encoder = Encoder()
        # self.init_size = 32 // 4  # Initial size before upsampling
        self.lin = nn.Conv2d(27*512, 41, kernel_size=1)
        # self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(222, 256, kernel_size=3, padding=1),  # 256 x 2 x 2
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # 512 x 4 x 4
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),  # 256 x 8 x 8
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # 128 x 16 x 16
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 64 x 32 x 32
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # 32 x 64 x 64
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  # 16 x 128 x 128
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),  # 8 x 256 x 256
            nn.InstanceNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),  # 1 x 256 x 256
            nn.Tanh()
        )

    def forward(self, noise, cond):
        cond = cond.view(cond.size(0), -1, 1, 1)
        cond = self.lin(cond)
        out = torch.cat([noise, cond], dim=1)
        # out = out.view(out.shape[0], 222, 1, 1)
        img = self.model(out)
        return img