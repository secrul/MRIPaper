"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch import nn
from torch.nn import functional as F
import sys
sys.path.append('/home/data/ljh/fastmri/MRI/util')
from fftc import ifft2c_old as ifft2c
from fftc import fft2c_old as fft2c

def DC(img, kspace, mask, inv_mask):
    #mask batch, 1, 320, 1;mask为1表示gt代替，为0使用预测值
    
    pre_kspace = fft2c(img.permute(0, 2, 3, 1))
    # print(pre_kspace.shape, kspace.shape,mask.shape)
    ans_kspace = kspace * mask + pre_kspace * inv_mask

    return ifft2c(ans_kspace).permute(0, 3, 1, 2)



class Unet_dc(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        normtype: str = 'in',
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.normtype = normtype

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, normtype)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, normtype))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, normtype)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, normtype))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, normtype),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )

    def forward(self,  image, kspace, mask, inv_mask):
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        # print(image.shape, kspace.shape, mask.shape)
        # res = 1 - mask
        # print(res.shape)
        # print(mask[0,...])
        # print(res[0,...])
        # assert 1 > 4
        stack = []
        output = image

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)
            # print(output.shape)
        return DC(output, kspace, mask, inv_mask)


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, normtype = 'in'):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.normtype = normtype

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        if self.normtype == 'bn':
            self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image):
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_chans, out_chans, kernel_size=2, stride=2, bias=False
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image):
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        return self.layers(image)

if __name__ == '__main__':
    x = torch.rand(1,2,256,256)
    kspace = torch.rand(1,256,256,2)
    mask = []
    for i in range(128):
        mask.append(0)
    for i in range(128):
        mask.append(1)
    mask = torch.tensor(mask)
    mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    # print(mask.shape)
    # assert 1 > 3
    model = Unet_dc(2,2, 32, 4, 'in')
    num_params = sum(param.nelement() for param in model.parameters())
    print(num_params / 1e6)
    out = model(x, kspace, mask)
    print(out.shape)
