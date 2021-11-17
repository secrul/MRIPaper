import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F


class CReLu(nn.Module):
    """
    input: [b,c,h,w,2]
    out: [b,c,h,w,2]
    """
    def __init__(self):
        super(CReLu, self).__init__()

    def forward(self,x):
        real = F.relu(x[...,0])
        imaginary = F.relu(x[...,1])
    
        output = torch.stack((real,imaginary),dim=-1)
        return output

class ComplexUpsample(nn.Module):
    """
    input: [b,c,h,w,2]
    out: [b,c,h,w,2]
    """
    def __init__(self, scale=2, mode = 'nearest'):
        super(ComplexUpsample, self).__init__()
        self.scale = scale
        self.mode = mode

    def forward(self,x):
        real = F.interpolate(x[...,0], scale_factor=self.scale, mode = self.mode)
        imaginary = F.interpolate(x[...,1], scale_factor=self.scale, mode = self.mode)
    
        output = torch.stack((real,imaginary),dim=-1)
        return output



class ComplexPool(nn.Module):
    """
    input: [b,c,h,w,2]
    out: [b,c,h/2,w/2,2]
    """
    def __init__(self, in_c, out_c):
        super(ComplexPool, self).__init__()
        self.conv = ComplexConv(in_channel=in_c, out_channel=out_c, kernel_size=2, stride=2, padding=0, act=False)

    def forward(self,x):

        output = self.conv(x)
        return output



class ComplexConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size = 3, stride=1, padding=1, act = True, norm = False, dilation=1, groups=1, bias=False):
        super(ComplexConv,self).__init__()
        self.padding = padding
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.act = act
        self.norm = norm
        if self.act:
            self.relu = CReLu()
        if self.norm:
            self.bn = RadialBN(out_channel)
        
        ## Model components
        self.conv_re = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.conv_im = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x): #
        real = self.conv_re(x[...,0]) - self.conv_im(x[...,1])
        imaginary = self.conv_re(x[...,1]) + self.conv_im(x[...,0])
        output = torch.stack((real,imaginary),dim=-1)
        if self.norm:
            output = self.bn(output)
        if self.act:
            return self.relu(output)
        else:
            return output



class ComplexTransConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size = 2, stride=2, padding=0, act = True, norm = False, dilation=1, groups=1, bias=False):
        super(ComplexTransConv,self).__init__()
        self.padding = padding
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.norm = norm
        self.act = act
        if self.norm:
            self.bn = RadialBN(out_channel)
        if self.act:
            self.relu = CReLu()
        ## Model components
        self.Tconv_re = nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.Tconv_im = nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x): # shpae of x : [batch,2,channel,axis1,axis2]
        real = self.Tconv_re(x[...,0]) - self.Tconv_im(x[...,1])
        imaginary = self.Tconv_re(x[...,1]) + self.Tconv_im(x[...,0])
        output = torch.stack((real,imaginary),dim=-1)
        if self.norm:
            output = self.bn(output)
        if self.act:
            output = self.relu(output)
        return output


class RadialBN(nn.Module):
    def __init__(self, num_features, t=5, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True,
        ):
        super(RadialBN, self).__init__()
        self.num_features = num_features
        self.t = t
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.bn_func = nn.BatchNorm2d(num_features=num_features,
                                                      eps=eps,
                                                      momentum=momentum,
                                                      affine=affine,
                                                      track_running_stats=track_running_stats)

    def convert_polar_to_cylindrical(self, x1, x2):
        '''
        converts the polar representation (i.e. magnitude and phase) of the complex tensor x1 ( or tensors x1 and x2)
        to cylindrical representation (i.e. real and imaginary)
        :param:
            x1: is a tensor contains both magnitude and phase channels in the last dims if x2=None;
            or contains only magnitude part if x2 contains phase component.
            x2: is a tensor similar to x2 or None
        '''

        real = x1 * torch.cos(x2)
        imag = x1 * torch.sin(x2)
        return real, imag


    def convert_cylindrical_to_polar(self, x1, x2):
        '''
        converts the cylindrical representation (i.e. real and imaginary) of the complex tensor x1 ( or tensors x1 and x2)
        to polar representation (i.e. magnitude and phase)
        :param:
            x1: is a tensor contains both real and imaginary channels in the last dims if x2=None;
            or contains only real part if x2 contains imaginary component.
            x2: is a tensor similar to x2 or None
        '''

        mag = (x1 ** 2 + x2 ** 2) ** (0.5)
        phase = torch.atan2(x2, x1)

        phase[phase.ne(phase)] = 0.0  # remove NANs
        return mag, phase


    def forward(self, x):
        input_real = x[...,0]
        input_imag = x[...,1]

        mag, phase = self.convert_cylindrical_to_polar(input_real, input_imag)

        # normalize the magnitude (see paper: El-Rewaidy et al. "Deep complex convolutional network for fast reconstruction of 3D late gadolinium enhancement cardiac MRI", NMR in Biomedicne, 2020)
        output_mag_norm = self.bn_func(mag) + self.t  # Normalize the radius to be around self.t (i.e. 5 std) (1 also works fine)

        output_real, output_imag = self.convert_polar_to_cylindrical(output_mag_norm, phase)

        output = torch.stack((output_real, output_imag), dim= - 1)

        return output   

#%%
if __name__ == "__main__":
    ## Random Tensor for Input
    ## shape : [batchsize,2,channel,axis1_size,axis2_size]
    ## Below dimensions are totally random
    x = torch.randn((1,2,4,4,2))
    print(x)
    # relu = CReLu()
    # print(relu(x))
    # bn = RadialBN(2)
    # print(bn(x))
    # pool1 = ComplexPool()
    # pool2 = ComplexPool(flag='avg')
    # print(pool1(x))
    # print(pool2(x))
    # up = ComplexUpsample()
    # print(up(x))
    # 1. Make ComplexConv Object
    ## (in_channel, out_channel, kernel_size) parameter is required
    # complexConv = ComplexConv(3,1,3, stride = 1, padding = 1, act = 'bn')
    # complexConv = ComplexTransConv(in_channel = 3,out_channel = 1, kernel_size = 2, stride = 2, padding = 0, act = 'bn')
    
    # 2. compute
    # y = complexConv(x)
    # print(y.shape)