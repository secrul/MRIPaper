"""
Dual-Octave Convolution for Accelerated Parallel MR Image Reconstruction
"""

"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import sys
sys.path.append('/home/westdata/ljh/projects/Brain_MRI/Brain_MRI/model/OCTConv')
import torch
from torch import nn
from torch.nn import functional as F

from ComplexCNN import ComplexTransConv, ComplexConv

sys.path.append('/home/westdata/ljh/projects/Brain_MRI/Brain_MRI/util')
from fftc import ifft2c_old as ifft2c
from fftc import fft2c_old as fft2c
import copy
def DC(img, kspace, mask):
    #mask batch, 1, 320, 1;mask为1表示gt代替，为0使用预测值
    
    pre_kspace = fft2c(img.permute(0, 2, 3, 1))
    # print(pre_kspace.shape, kspace.shape,mask.shape)
    ans_kspace = kspace * mask + pre_kspace *(1 - mask)

    return ifft2c(ans_kspace).permute(0, 3, 1, 2)



class OCT2d(nn.Module):

    def __init__(self, low_c, high_c, kernel_size=3, strides=1,padding=1, types = 'first'):
       
        super().__init__()

        # self.alpha = alpha_out
        self.low_c = low_c
        self.high_c = high_c
        self.stride = strides
        self.padding = padding
        self.kernel_size = kernel_size
        self.types = types

        self.avgpool =  nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2)

        if self.types == 'first':
            self.high_re2low = nn.Conv2d(1, self.low_c, kernel_size=self.kernel_size,stride = self.stride,  padding=self.padding, bias=True)
            self.high_re2high = nn.Conv2d(1,self.high_c, kernel_size=self.kernel_size,stride = self.stride, padding=self.padding, bias=True)
            self.high_im2low = nn.Conv2d(1, self.low_c, kernel_size=self.kernel_size,stride = self.stride,  padding=self.padding, bias=True)
            self.high_im2high = nn.Conv2d(1,self.high_c, kernel_size=self.kernel_size,stride = self.stride, padding=self.padding, bias=True)
        
        if self.types == 'oct':

            self.low_re2low = nn.Conv2d(self.low_c, self.low_c, kernel_size=self.kernel_size, stride = self.stride,  padding=self.padding, bias=True)
            self.low_re2high = nn.Conv2d(self.low_c, self.high_c, kernel_size=self.kernel_size,stride =self.stride, padding=self.padding, bias=True)
            self.low_im2low = nn.Conv2d(self.low_c, self.low_c, kernel_size=self.kernel_size,stride = self.stride,   padding=self.padding, bias=True)
            self.low_im2high = nn.Conv2d(self.low_c, self.high_c, kernel_size=self.kernel_size,stride = self.stride, padding=self.padding, bias=True)

            self.high_re2low = nn.Conv2d(self.high_c, self.low_c, kernel_size=self.kernel_size,stride = self.stride,  padding=self.padding, bias=True)
            self.high_re2high = nn.Conv2d(self.high_c,self.high_c, kernel_size=self.kernel_size,stride = self.stride, padding=self.padding, bias=True)
            self.high_im2low = nn.Conv2d(self.high_c, self.low_c, kernel_size=self.kernel_size,stride = self.stride,  padding=self.padding, bias=True)
            self.high_im2high = nn.Conv2d(self.high_c,self.high_c, kernel_size=self.kernel_size,stride = self.stride, padding=self.padding, bias=True)
        if self.types == 'last':
            self.high_re2high = nn.Conv2d(self.high_c + self.low_c,1, kernel_size=self.kernel_size,stride = self.stride, padding=self.padding, bias=True)
            self.high_im2high = nn.Conv2d(self.high_c + self.low_c,1, kernel_size=self.kernel_size,stride = self.stride, padding=self.padding, bias=True)
        


    def forward(self, low_input_re = None, low_input_im = None, high_input_re = None, high_input_im = None):
        #low -> high 先卷积再上采样
        #high -> low先下采样再卷积
        if self.types == 'first':
            high_2_high_re = self.high_re2high(high_input_re) - self.high_im2high(high_input_im)
            high_2_high_im = self.high_im2high(high_input_re) + self.high_re2high(high_input_im)

            high_2_low_re = self.high_re2low(self.avgpool(high_input_re)) - self.high_im2low(self.avgpool(high_input_im))
            high_2_low_im = self.high_im2low(self.avgpool(high_input_re)) + self.high_re2low(self.avgpool(high_input_im))      

            low_ans_re = high_2_low_re 
            low_ans_im = high_2_low_im
            high_ans_re = high_2_high_re
            high_ans_im = high_2_high_im

            return low_ans_re, low_ans_im, high_ans_re, high_ans_im

        if self.types == 'oct':
            low_2_low_re = self.low_re2low(low_input_re) - self.low_im2low(low_input_im)
            low_2_low_im = self.low_im2low(low_input_re) + self.low_re2low(low_input_im)
            high_2_high_re = self.high_re2high(high_input_re) - self.high_im2high(high_input_im)
            high_2_high_im = self.high_im2high(high_input_re) + self.high_re2high(high_input_im)

            low_2_high_re = self.up(self.low_re2high(low_input_re) - self.low_im2high(low_input_im))
            low_2_high_im = self.up(self.low_im2high(low_input_re) + self.low_re2high(low_input_im))
            high_2_low_re = self.high_re2low(self.avgpool(high_input_re)) - self.high_im2low(self.avgpool(high_input_im))
            high_2_low_im = self.high_im2low(self.avgpool(high_input_re)) + self.high_re2low(self.avgpool(high_input_im))      

            low_ans_re = low_2_low_re + high_2_low_re 
            low_ans_im = low_2_low_im + high_2_low_im
            high_ans_re = high_2_high_re + low_2_high_re
            high_ans_im = high_2_high_im + low_2_high_im

            return low_ans_re, low_ans_im, high_ans_re, high_ans_im
        if self.types == 'last':

            high_2_high_re = self.high_re2high(high_input_re) - self.high_im2high(high_input_im)
            high_2_high_im = self.high_im2high(high_input_re) + self.high_re2high(high_input_im)
  

            return high_2_high_re, high_2_high_im

        

class firstOct_complex_conv2d(nn.Module):
    def __init__(self, low_c = 4, high_c = 28, kernel_size=3, strides=1,padding=1):
       
        super().__init__()
        self.low_c = low_c
        self.high_c = high_c
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.relu =nn.ReLU(inplace=True)
        self.oct = OCT2d(self.low_c, self.high_c, types = 'first')
    def forward(self, img):
        img_r = img[:,0,:,:].unsqueeze(1)
        img_i = img[:,1,:,:].unsqueeze(1)
        low_ans_re, low_ans_im, high_ans_re, high_ans_im = self.oct(high_input_re=img_r,high_input_im=img_i)
        return self.relu(low_ans_re), self.relu(low_ans_im), self.relu(high_ans_re), self.relu(high_ans_im)


class Oct_complex_conv2d(nn.Module):
    def __init__(
        self, low_c = 4, high_c = 28, kernel_size=3, strides=1,padding=1):

        super().__init__()
        self.low_c = low_c
        self.high_c = high_c
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

        self.relu =nn.ReLU(inplace=True)
        self.oct = OCT2d(self.low_c, self.high_c, types = 'oct')
    def forward(self, low_ans_re, low_ans_im, high_ans_re, high_ans_im):

        low_ans_re, low_ans_im, high_ans_re, high_ans_im = self.oct(low_ans_re, low_ans_im, high_ans_re, high_ans_im)
        return self.relu(low_ans_re), self.relu(low_ans_im), self.relu(high_ans_re), self.relu(high_ans_im)


class lastOct_complex_conv2d(nn.Module):
    def __init__(
        self, low_c = 4, high_c = 28, kernel_size=3, strides=1,padding=1):
    
        super().__init__()
        self.low_c = low_c
        self.high_c = high_c
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

        self.relu =nn.ReLU(inplace=True)
        self.oct = OCT2d(self.low_c, self.high_c, types = 'last')
    def forward(self, high_ans_re, high_ans_im):
        high_ans_re, high_ans_im = self.oct(high_input_re=high_ans_re,high_input_im=high_ans_im)
        return self.relu(high_ans_re), self.relu(high_ans_im)

class DualOct(nn.Module):

    def __init__(self, alpha, channel, n_block):
    
        super().__init__()
        self.alpha = alpha
        self.channel = channel
        self.n_blocks = n_block
        templayerList = []
        self.avgpool =  nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2)
        for i in range(self.n_blocks):
            templayerList.append(firstOct_complex_conv2d()) 
            templayerList.append(Oct_complex_conv2d()) 
            templayerList.append(Oct_complex_conv2d()) 
            templayerList.append(Oct_complex_conv2d()) 
            templayerList.append(lastOct_complex_conv2d()) 

        self.layerList = nn.ModuleList(templayerList)

    def forward(self, img, mask_k, mask):
        tmp = copy.deepcopy(img)
        for i in range(self.n_blocks * 5):
            i += 1
            if i % 5 == 1:
                low_re, low_im, high_re, high_im = self.layerList[i-1](tmp)
            if 2 <= i % 5 <= 4:
                low_re, low_im, high_re, high_im = self.layerList[i-1](low_re, low_im, high_re, high_im)
            if i % 5 == 0:
                low_re = self.up(low_re)
                low_im = self.up(low_im)
                high_re = torch.cat([high_re, low_re], dim = 1)
                high_im = torch.cat([high_im, low_im], dim = 1)
                # print(high_re.shape)
                # print(high_im.shape)
                # assert 1 > 4
                high_re, high_im = self.layerList[i-1](high_re, high_im)
                out = torch.cat([high_re, high_im],dim=1)
                # print(out.shape, img.shape)
                # assert 1 > 4
                out += img
                tmp = DC(out, mask_k, mask)

        return tmp
if __name__ == '__main__':
    x = torch.rand(2,2,256,256)
    mask_k = torch.rand(2,256,256,2)
    mask = torch.rand(2,1,256,1)
    # y = torch.rand(2,56,512,512)
    model = DualOct(0.125, 32, 10)
    num_params = sum(param.nelement() for param in model.parameters())
    print(num_params / 1e6)
    out = model(x,mask_k, mask)
    print(out.shape)
