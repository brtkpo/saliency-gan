import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=64):
        super(Generator, self).__init__()

        self.enc1 = self.contract_block(in_channels, features, 4, 2, 1)
        self.enc2 = self.contract_block(features, features*2, 4, 2, 1)
        self.enc3 = self.contract_block(features*2, features*4, 4, 2, 1)
        self.enc4 = self.contract_block(features*4, features*8, 4, 2, 1)

        self.dec1 = self.expand_block(features*8, features*4, 4, 2, 1)
        self.dec2 = self.expand_block(features*8, features*2, 4, 2, 1)
        self.dec3 = self.expand_block(features*4, features, 4, 2, 1)
        self.dec4 = nn.ConvTranspose2d(features*2, out_channels, kernel_size=4, stride=2, padding=1)

        self.out_act = nn.Sigmoid()

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d1 = self.dec1(e4)
        d2 = self.dec2(torch.cat([d1, e3], dim=1))
        d3 = self.dec3(torch.cat([d2, e2], dim=1))
        d4 = self.dec4(torch.cat([d3, e1], dim=1))
        return self.out_act(d4)

    @staticmethod
    def contract_block(in_channels, out_channels, kernel_size, stride, padding):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        return block

    @staticmethod
    def expand_block(in_channels, out_channels, kernel_size, stride, padding):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        return block

class Discriminator(nn.Module):
    def __init__(self, in_channels=4, features=[64, 128, 256, 512]):
        super(Discriminator, self).__init__()
        layers = []

        for idx, feature in enumerate(features):
            if idx == 0:
                layers.append(nn.Conv2d(in_channels, feature, 4, 2, 1))
            else:
                layers.append(nn.Conv2d(features[idx-1], feature, 4, 2, 1))
                layers.append(nn.BatchNorm2d(feature))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Conv2d(features[-1], 1, 4, 1, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
    net.apply(init_func)