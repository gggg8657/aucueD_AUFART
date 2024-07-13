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
    def __init__(self, latent_dim=100, au_dim=41, out_channels=1, out_scale=0.01, sample_mode='bilinear'):
        super(Generator, self).__init__()
        self.out_scale = out_scale
        self.latent_dim = latent_dim
        self.au_dim = au_dim
        
        self.init_size = 32 // 4  # Initial size before upsampling
        # latent_dim + au_dim으로 입력 차원 변경
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))
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
    
        

class AGenerator(nn.Module):
    def __init__(self, latent_dim=100, au_dim=41, out_channels=1, out_scale=0.01, sample_mode='bilinear'):
        super(AGenerator, self).__init__()
        self.out_scale = out_scale
        self.latent_dim = latent_dim
        self.au_dim = au_dim
        
        self.init_size = 32 // 4  # Initial size before upsampling
        # latent_dim + au_dim으로 입력 차원 변경
        self.l1 = nn.Sequential(nn.Linear(27 * 512, 128 * self.init_size ** 2))
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
        # noise shape: (batch, 27, 512)
        # flatten noise to (batch, 27 * 512)
        noise_flat = noise.view(noise.size(0), -1)
        out = self.l1(noise_flat)
        
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img * self.out_scale