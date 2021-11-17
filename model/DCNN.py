# -*- encoding: utf-8 -*-
'''
@File    :   DCNN.py
@Time    :   2021/11/17 10:45:16
@Author  :   secrul 
@Version :   1.0
@Contact :   secrul@163.com
'''

# here put the import lib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

"""
A Deep Cascade of Convolutional Neural Networks for Dynamic MR Image Reconstruction
"""
"""
author
"""


def abs4complex(x):
    y = torch.zeros_like(x)
    y[:,0:1] = torch.sqrt(x[:,0:1]*x[:,0:1]+x[:,1:2]*x[:,1:2])
    y[:,1:2] = 0

    return y

def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)

def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)

def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)

def fft2(data):
    """
    Apply centered 2 dimensional Fast Fourier Transform.

    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.

    Returns:
        torch.Tensor: The FFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = torch.fft(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    return data

def ifft2(data):
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.

    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.

    Returns:
        torch.Tensor: The IFFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = torch.ifft(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    return data

class dataConsistencyLayer_static(nn.Module):
    def __init__(self, initLamda = 1, trick = 0, dynamic = False, conv = None):
        super(dataConsistencyLayer_static, self).__init__()
        self.normalized = True 
        self.trick = trick
        tempConvList = []
        if(self.trick in [3,4]):
            if(conv is None):
                if(dynamic):
                    conv = nn.Conv3d(4,2,1,padding=0)
                else:
                    conv = nn.Conv2d(4,2,1,padding=0)
            tempConvList.append(conv)
        self.trickConvList = nn.ModuleList(tempConvList)

    def dc_operate(self, xin, y, mask):
        iScale = 1
        if(len(xin.shape)==4):
            if(xin.shape[1]==1):
                emptyImag = torch.zeros_like(xin)
                xin_c = torch.cat([xin,emptyImag],1).permute(0,2,3,1)
            else:
                xin_c = xin.permute(0,2,3,1)
                y = y.permute(0,2,3,1)
            mask = mask.reshape(mask.shape[0],mask.shape[1],mask.shape[2],1)
        elif(len(xin.shape)==5):
            if(xin.shape[1]==1):
                emptyImag = torch.zeros_like(xin)
                xin_c = torch.cat([xin,emptyImag],1).permute(0,2,3,4,1)
            else:
                xin_c = xin.permute(0,2,3,4,1)
            mask = mask.reshape(mask.shape[0],mask.shape[1],mask.shape[2],mask.shape[3],1)
        else:
            assert False, "xin shape length has to be 4(2d) or 5(3d)"
        
        xin_f = fft2(xin_c)
        xGT_f = y
        # print(mask.shape, xin_f.shape, xGT_f.shape) #torch.Size([1, 320, 320, 2]) torch.Size([1, 320, 320, 2]) torch.Size([1, 320, 320, 2])
        # assert 1>4
        xout_f = xin_f + (- xin_f + xGT_f) * iScale * mask

        xout = ifft2(xout_f)
        if(len(xin.shape)==4):
            xout = xout.permute(0,3,1,2)
        else:
            xout = xout.permute(0,4,1,2,3)
        if(xin.shape[1]==1):
            xout = torch.sqrt(xout[:,0:1]*xout[:,0:1]+xout[:,1:2]*xout[:,1:2])
        
        return xout
    
    def forward(self, xin, y, mask):
        xt = xin
        if(self.trick == 1):
            xt = abs4complex(xt)
            xt = self.dc_operate(xt, y, mask)
        elif(self.trick == 2):
            xt = self.dc_operate(xt, y, mask)
            xt = abs4complex(xt)
            xt = self.dc_operate(xt, y, mask)
        elif(self.trick == 3):
            xdc1 = self.dc_operate(xt, y, mask)
            xt = abs4complex(xt)
            xdc2 = self.dc_operate(xt, y, mask)
            xdc = torch.cat([xdc1,xdc2],1)
            xt = self.trickConvList[0](xdc)
        elif(self.trick == 4):
            xdc1 = self.dc_operate(xt, y, mask)
            xabs = abs4complex(xdc1)
            xdc2 = self.dc_operate(xabs, y, mask)
            xdc = torch.cat([xdc1,xdc2],1)
            xt = self.trickConvList[0](xdc)
        else:
            xt = self.dc_operate(xt, y, mask)

        return xt



class convBlock(nn.Module):
    def __init__(self, iConvNum = 5, f=64):
        super(convBlock, self).__init__()
        self.Relu = nn.ReLU()
        self.conv1 = nn.Conv2d(2,f,3,padding = 1)
        convList = []
        for i in range(1, iConvNum-1):
            tmpConv = nn.Conv2d(f,f,3,padding = 1)
            convList.append(tmpConv)
        self.layerList = nn.ModuleList(convList)
        self.conv2 = nn.Conv2d(f,2,3,padding = 1)
    
    def forward(self, x1):
        x2 = self.conv1(x1)
        x2 = self.Relu(x2)
        for conv in self.layerList:
            x2 = conv(x2)
            x2 = self.Relu(x2)
        x3 = self.conv2(x2)
        
        x4 = x3 + x1
        
        return x4

class DCNN(nn.Module):
    def __init__(self, d = 5, c = 5, fNum = 64):
        super(DCNN, self).__init__()
        templayerList = []
        for i in range(c):
            tmpConv = convBlock(d, fNum)
            tmpDF = dataConsistencyLayer_static()
            templayerList.append(tmpConv)
            templayerList.append(tmpDF)
        self.layerList = nn.ModuleList(templayerList)
        
    def forward(self, x1, y, mask):
        xt = x1
        flag = True
        for layer in self.layerList:
            if(flag):
                xt = layer(xt)
                flag = False
            else:
                xt = layer(xt, y, mask)
                flag = True
            # xt = layer(xt)
        
        return xt


if __name__ == '__main__':

    x = torch.rand(2,2,320,320).cuda()
    k = torch.rand(2,2,320,320).cuda()
    m = torch.rand(2,1,320,1).cuda()
    model = DCNN(5, 5).cuda()
    num_params = sum(param.nelement() for param in model.parameters())
    print(num_params / 1e6)
    out = model(x,k,m)
    print(out.shape)