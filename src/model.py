import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        return self.relu(out)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False)
        self.bn = nn.BatchNorm2d(out_channels * 4)
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        out = torch.cat((x1, x2, x3, x4), dim=1)
        out = self.bn(out)
        return self.project(F.relu(out))

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=64):
        super(Generator, self).__init__()

        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )

        self.enc1 = ResidualBlock(features, features, stride=2)
        self.enc2 = ResidualBlock(features, features * 2, stride=2)
        self.enc3 = ResidualBlock(features * 2, features * 4, stride=2)
        self.enc4 = ResidualBlock(features * 4, features * 8, stride=2)

        self.bottleneck = ASPP(features * 8, features * 8)

        self.dec1 = self.expand_block(features * 8, features * 4)
        self.dec2 = self.expand_block(features * 8, features * 2)
        self.dec3 = self.expand_block(features * 4, features)
        self.dec4 = self.expand_block(features * 2, features)

        # Final Output Layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features // 2, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_init = self.init_conv(x)

        e1 = self.enc1(x_init)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b = self.bottleneck(e4)

        d1 = self.dec1(b)
        d2 = self.dec2(torch.cat([d1, e3], dim=1))
        d3 = self.dec3(torch.cat([d2, e2], dim=1))
        d4 = self.dec4(torch.cat([d3, e1], dim=1))

        out = self.final_conv(d4)
        return out

    @staticmethod
    def expand_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class Discriminator(nn.Module):
    def __init__(self, in_channels=4, features=[64, 128, 256, 512]):
        super(Discriminator, self).__init__()
        layers = []

        for idx, feature in enumerate(features):
            if idx == 0:
                layers.append(spectral_norm(nn.Conv2d(in_channels, feature, 4, 2, 1)))
            else:
                layers.append(spectral_norm(nn.Conv2d(features[idx - 1], feature, 4, 2, 1)))
                layers.append(nn.BatchNorm2d(feature))

            layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Final layer
        layers.append(nn.Conv2d(features[-1], 1, 4, 1, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)