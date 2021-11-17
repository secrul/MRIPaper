"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch import nn
from torch.nn import functional as F


class ComplexConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size = 3, stride=1, padding=1, act = None, dilation=1, groups=1, bias=True):
        super(ComplexConv,self).__init__()
        self.padding = padding
        self.bn_re = nn.BatchNorm2d(out_channel)
        self.bn_im = nn.BatchNorm2d(out_channel)
        self.act = act
        ## Model components
        self.conv_re = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.conv_im = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x): # shpae of x : [batch,2,channel,axis1,axis2]
        # print(x.shape,x[...,0].shape)
        # assert 1 > 4
        real = self.conv_re(x[...,0]) - self.conv_im(x[...,1])
        imaginary = self.conv_re(x[...,1]) + self.conv_im(x[...,0])
        # print(real.shape, imaginary.shape)
        if self.act:
            real = self.bn_re(real)
            imaginary = self.bn_im(imaginary)
        output = torch.stack((real,imaginary),dim=-1)
        # print(output.shape)
        # assert 1 > 5

        return output

class ComplexTransConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size = 2, stride=2, padding=0, act = None, dilation=1, groups=1, bias=False):
        super(ComplexTransConv,self).__init__()
        self.padding = padding
        self.bn_re = nn.BatchNorm2d(out_channel)
        self.bn_im = nn.BatchNorm2d(out_channel)
        self.act = act
        ## Model components
        self.Tconv_re = nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.Tconv_im = nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x): # shpae of x : [batch,2,channel,axis1,axis2]
        real = self.Tconv_re(x[...,0]) - self.Tconv_im(x[...,1])
        imaginary = self.Tconv_re(x[...,1]) + self.Tconv_im(x[...,0])
        if self.act:
            real = self.bn_re(real)
            imaginary = self.bn_im(imaginary)
        output = torch.stack((real,imaginary),dim=-1)
        
        return output

class Unet(nn.Module):
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
                ComplexConv(ch, self.out_chans, kernel_size=1, stride=1, padding=0),
            )
        )

    def _avg_complex(self, x):
        real = F.avg_pool2d(x[...,0], kernel_size=2, stride=2, padding=0)
        imaginary = F.avg_pool2d(x[...,1], kernel_size=2, stride=2, padding=0)
        output = torch.stack((real,imaginary),dim=-1)
        
        return output


    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        stack = []
        output = image

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            # print(output.shape)
            output = layer(output)
            # print(output.shape)
            stack.append(output)
            output = self._avg_complex(output)

        output = self.conv(output)

        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)


            output = torch.cat([output, downsample_layer], dim=1)
            # print(output.shape, downsample_layer.shape)
            # assert 3> 8
            output = conv(output)

        return output


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, normtype = 'bn'):
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
        ComplexConv(in_chans, out_chans, kernel_size=3, padding=1, act='bn', bias=False),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        ComplexConv(out_chans, out_chans, kernel_size=3, padding=1, act='bn',bias=False),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        # print('img',image.shape)
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
            ComplexTransConv(
                in_chans, out_chans, kernel_size=2,act='bn', stride=2, bias=False
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        return self.layers(image)

if __name__ == '__main__':
    x = torch.rand(2,1,256,256,2)
    model = Unet(1,1, 32, 4, 'bn')
    num_params = sum(param.nelement() for param in model.parameters())
    print(num_params / 1e6)
    out = model(x)
    print(out.shape)
