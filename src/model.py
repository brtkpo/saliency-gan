import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class ResidualBlock(nn.Module):
    """
    Residual convolutional block used in the encoder part of the generator.

    This block consists of two convolutional layers with batch normalization
    and a residual skip connection. If the input and output channel sizes
    differ, a projection is applied to the skip path.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int, optional
        Stride applied in the first convolution layer. Default is 1.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass through the residual block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor after applying residual connection.
        """
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        return self.relu(out)


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling module.

    This module captures multi-scale contextual information using parallel
    dilated convolutions with different dilation rates.

    Parameters
    ----------
    in_channels : int
        Number of input feature channels.
    out_channels : int
        Number of output feature channels.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(
            in_channels, out_channels, 3, padding=6, dilation=6, bias=False
        )
        self.conv3 = nn.Conv2d(
            in_channels, out_channels, 3, padding=12, dilation=12, bias=False
        )
        self.conv4 = nn.Conv2d(
            in_channels, out_channels, 3, padding=18, dilation=18, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels * 4)
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ASPP module.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map (B, C, H, W)

        Returns
        -------
        torch.Tensor
            Output feature map (B, out_channels, H, W)
        """
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        out = torch.cat((x1, x2, x3, x4), dim=1)
        out = self.bn(out)
        return self.project(F.relu(out))


class Generator(nn.Module):
    """
    Generator network for saliency map prediction.

    The model follows an encoder–decoder architecture with residual blocks,
    skip connections, and an ASPP bottleneck to capture multi-scale features.

    Parameters
    ----------
    in_channels : int, optional
        Number of input image channels. Default is 3.
    out_channels : int, optional
        Number of output channels for the saliency map. Default is 1.
    features : int, optional
        Base number of convolutional filters used in the encoder. Default is 64.
    """

    def __init__(
        self, in_channels: int = 3, out_channels: int = 1, features: int = 64
    ) -> None:
        super(Generator, self).__init__()

        self.init_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, features, kernel_size=7, stride=1, padding=3, bias=False
            ),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
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
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the generator.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor (B, 3, H, W)

        Returns
        -------
        torch.Tensor:
            Predicted saliency map (B, 1, H, W)
        """
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
    def expand_block(in_channels: int, out_channels: int) -> nn.Sequential:
        """
        Create a decoder upsampling block.

        The block performs bilinear upsampling followed by convolution,
        batch normalization, and ReLU activation.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.

        Returns
        -------
        nn.Sequential
            Sequential module implementing the upsampling block.
        """
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class Discriminator(nn.Module):
    """
    PatchGAN discriminator used for adversarial training.

    The discriminator operates on the concatenation of the input image and
    the predicted or ground-truth saliency map.

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels (image + saliency map). Default is 4.
    features : list[int], optional
        List specifying the number of feature channels in each layer.
        Default is [64, 128, 256, 512].
    """

    def __init__(
        self, in_channels: int = 4, features: list[int] = [64, 128, 256, 512]
    ) -> None:
        super(Discriminator, self).__init__()
        layers = []

        for idx, feature in enumerate(features):
            if idx == 0:
                layers.append(spectral_norm(nn.Conv2d(in_channels, feature, 4, 2, 1)))
            else:
                layers.append(
                    spectral_norm(nn.Conv2d(features[idx - 1], feature, 4, 2, 1))
                )
                layers.append(nn.BatchNorm2d(feature))

            layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Final layer
        layers.append(nn.Conv2d(features[-1], 1, 4, 1, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (B, 4, H, W)

        Returns
        -------
        torch.Tensor:
            PatchGAN prediction map.
        """
        return self.model(x)
