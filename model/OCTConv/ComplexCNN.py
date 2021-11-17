import torch
import torch.nn as nn
import numpy as np

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
        real = self.conv_re(x[:,0]) - self.conv_im(x[:,1])
        imaginary = self.conv_re(x[:,1]) + self.conv_im(x[:,0])
        if self.act:
            real = self.bn_re(real)
            imaginary = self.bn_im(imaginary)
        output = torch.stack((real,imaginary),dim=1)
        
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
        real = self.Tconv_re(x[:,0]) - self.Tconv_im(x[:,1])
        imaginary = self.Tconv_re(x[:,1]) + self.Tconv_im(x[:,0])
        if self.act:
            real = self.bn_re(real)
            imaginary = self.bn_im(imaginary)
        output = torch.stack((real,imaginary),dim=1)
        
        return output
        
#%%
if __name__ == "__main__":
    ## Random Tensor for Input
    ## shape : [batchsize,2,channel,axis1_size,axis2_size]
    ## Below dimensions are totally random
    x = torch.randn((10,2,3,100,100))
    
    # 1. Make ComplexConv Object
    ## (in_channel, out_channel, kernel_size) parameter is required
    # complexConv = ComplexConv(3,1,3, stride = 1, padding = 1, act = 'bn')
    complexConv = ComplexTransConv(in_channel = 3,out_channel = 1, kernel_size = 2, stride = 2, padding = 0, act = 'bn')
    
    # 2. compute
    y = complexConv(x)
    print(y.shape)