import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np

import sys
sys.path.append('/home/data/ljh/fastmri/MRI/util')
from fftc import ifft2c_old as ifft2c
from fftc import fft2c_old as fft2c
# from util import fftc
"""
Model Learning: Primal Dual Networks for Fast MR imaging
"""

class FFT_Mask_ForBack(torch.nn.Module):
    def __init__(self):
        super(FFT_Mask_ForBack, self).__init__()

    def forward(self, x, mask):
        x = x.permute(0,2,3,1)
        fftz = fft2c(x)
        z_hat = fftz * mask
        x = z_hat.permute(0,3,1,2)
        return x


class IFFT_Mask_ForBack(torch.nn.Module):
    def __init__(self):
        super(IFFT_Mask_ForBack, self).__init__()

    def forward(self, x, mask):
        x = x.permute(0,2,3,1)
        z_hat = x * mask
        z_hat = ifft2c(z_hat)
        x = z_hat.permute(0,3,1,2)
        return x


class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()

        self.conv1_prime = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 6, 3, 3)))
        self.conv2_prime = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv3_prime = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv4_prime = nn.Parameter(init.xavier_normal_(torch.Tensor(2, 32, 3, 3)))

        self.conv1_dual = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 4, 3, 3)))
        self.conv2_dual = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv3_dual = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv4_dual = nn.Parameter(init.xavier_normal_(torch.Tensor(2, 32, 3, 3)))

    def forward(self, d, mask, m, f, fft_forback, ifft_forback):
        m2k = fft_forback(m, mask)
        # print(d.shape)
        # print(m2k.shape)
        # print(f.shape)
        # assert 1 > 3
        input_to_prime = torch.cat((d, m2k, f), 1)

        d_hat = F.conv2d(input_to_prime, self.conv1_prime, padding=1)
        d_hat = F.relu(d_hat)
        d_hat = F.conv2d(d_hat, self.conv2_prime, padding=1)
        d_hat = F.relu(d_hat)
        d_hat = F.conv2d(d_hat, self.conv3_prime, padding=1)
        d_hat = F.conv2d(d_hat, self.conv4_prime, padding=1)

        d_hat = d_hat + d  # 网络只学习残差
        d_hat_i = ifft_forback(d_hat, mask)
        input_to_dual = torch.cat((m, d_hat_i), 1)
        p_hat = F.conv2d(input_to_dual, self.conv1_dual, padding=1)
        p_hat = F.relu(p_hat)
        p_hat = F.conv2d(p_hat, self.conv2_dual, padding=1)
        p_hat = F.relu(p_hat)
        p_hat = F.conv2d(p_hat, self.conv3_dual, padding=1)
        p_hat = F.conv2d(p_hat, self.conv4_dual, padding=1)

        p_hat = p_hat + m

        return d_hat, p_hat


class PDNET(torch.nn.Module):
    def __init__(self, LayerNo = 10):
        super(PDNET, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo
        self.fft_forback = FFT_Mask_ForBack()
        self.ifft_forback = IFFT_Mask_ForBack()

        for i in range(LayerNo):
            onelayer.append(BasicBlock())

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, x, m, mask):
        d = torch.zeros_like(x)

        for i in range(self.LayerNo):
            [d, m] = self.fcs[i](d, mask, m, x, self.fft_forback, self.ifft_forback)
        
        x_final = m
        return x_final

if __name__ == '__main__':
    x = torch.rand(2,2,256,256)
    mask_k = torch.rand(2,2,256,256)
    mask = torch.rand(2,1,256,1)
    model = PDNET()
    # print(model)
    num_params = sum(param.nelement() for param in model.parameters())
    print(num_params / 1e6)
    # out = model(x, mask_k, mask)
    # print(out.shape)