import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # 64 x 128 x 128
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 128 x 64 x 64
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 256 x 32 x 32
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 512 x 16 x 16
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),  # 1024 x 8 x 8
            nn.InstanceNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, stride=2, padding=1),  # 512 x 4 x 4
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1),  # 256 x 2 x 2
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),  # 128 x 1 x 1
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        return self.model(x)
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
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

    def forward(self, x):
        return self.model(x)
    
class ConditionalGenerator(nn.Module):
    def __init__(self, condition_dim):
        super(ConditionalGenerator, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.fc = nn.Linear(condition_dim, 128)

    def forward(self, x, condition):
        condition_feat = self.fc(condition).view(-1, 128, 1, 1)
        encoded_feat = self.encoder(x)
        combined_feat = torch.cat([encoded_feat, condition_feat], dim=1)
        output = self.decoder(combined_feat)
        return output
    
class Discriminator(nn.Module):
    def __init__(self, condition_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=2, padding=1),  # 64 x 112 x 112
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
        )
    
    def forward(self, img, condition):
        condition = condition.view(condition.size(0), 1, 1, 1).repeat(1, 1, img.size(2), img.size(3))
        d_in = torch.cat((img, condition), 1)
        return self.model(d_in)