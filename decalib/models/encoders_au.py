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

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

class demo_AUEncoder(nn.Module):
    def __init__(self):
        super(demo_AUEncoder,self).__init__()

        self.flatten=nn.Flatten(start_dim=1, end_dim=-1)
        self.fc = nn.Linear(27*512, 75)

    def forward(self, x):
        x=self.flatten(x)
        x=self.fc(x)
        return x
class AUEncoder(nn.Module):
    def __init__(self):
        super(AUEncoder,self).__init__()

        self.flatten=nn.Flatten(start_dim=1, end_dim=-1)
        self.fc1 = nn.Linear(27*512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 41)
        self.relu = nn.ReLU()

    def forward(self, x):
        x=self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
class Enc_deca_AU(nn.Module):
    def __init__(self):
        super(Enc_deca_AU,self).__init__()

        self.flatten=nn.Flatten(start_dim=1, end_dim=-1)
        self.fc1 = nn.Linear(27*512+181, 256)
        self.fc2 = nn.Linear(256, 181)
        # self.fc3 = nn.Linear(128, 64)
        # self.fc4 = nn.Linear(64, 41)
        # self.relu = nn.ReLU()

    def forward(self, x):
        x=self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
class AUEncoder_Old(nn.Module):
    def __init__(self):
        super(AUEncoder, self).__init__()
        ### regressor
        self.layers = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
        )
        # self.last_op = last_op

    def forward(self, features):
        x = self.layers(features)
        return x

class AUDetector(nn.Module):
    def __init__(self):
        super(AUDetector, self).__init__()
        ### regressor
        self.layers = nn.Sequential(
            nn.Linear(53, 41),
            nn.Sigmoid()
        )
        # self.last_op = last_op

    def forward(self, features):
        x = self.layers(features)
        return x
