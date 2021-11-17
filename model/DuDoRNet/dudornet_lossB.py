import os
import torch
import torch.nn as nn
import sys
sys.path.append('/home/data/ljh/Brain_MRI/Brain_MRI/model/DuDoRNet')

from networks import gaussian_weights_init, DRDN
from utils import  DataConsistencyInKspace_I, DataConsistencyInKspace_K, fft2


"""

DataConsistencyInKspace_I:输入:图像域，先fft转换到k空间，然后dc，然后ifft转回到图像域

DataConsistencyInKspace_K：输入：k空间，dc，然后ifft转到图像域
"""
import pdb


class RecurrentModel(nn.Module):
    def __init__(self, n_recurrent = 5):
        super(RecurrentModel, self).__init__()

        self.loss_names = []
        self.networks = []
        self.optimizers = []

        self.n_recurrent = n_recurrent


        self.net_G_I = network = DRDN(n_channels=2, G0=32, kSize=3, D=3, C=4, G=32, dilateSet=[1,2,3,3])
        self.net_G_K = network = DRDN(n_channels=2, G0=32, kSize=3, D=3, C=4, G=32, dilateSet=[1,2,3,3])
        self.networks.append(self.net_G_I)
        self.networks.append(self.net_G_K)

        # data consistency layers in image space & k-space
        dcs_I = []
        for i in range(self.n_recurrent):
            dcs_I.append(DataConsistencyInKspace_I(noise_lvl=None))
        self.dcs_I = dcs_I

        dcs_K = []
        for i in range(self.n_recurrent):
            dcs_K.append(DataConsistencyInKspace_K(noise_lvl=None))
        self.dcs_K = dcs_K


    def initialize(self):
        [net.apply(gaussian_weights_init) for net in self.networks]


    def forward(self,x, k, mask):
        I = x
        net = {}
        ans_I = [0 for _ in range(self.n_recurrent)]
        ans_K = [0 for _ in range(self.n_recurrent)]
        for i in range(1, self.n_recurrent + 1):
            '''Image Space'''

            x_I = I

            net['r%d_img_pred' % i] = self.net_G_I(x_I)  # output recon image
            ans_I[i-1], _ = self.dcs_I[i - 1](net['r%d_img_pred' % i], k, mask)
            
            '''K Space'''
            net['r%d_kspc_img_dc_pred' % i] = fft2(ans_I[i-1].permute(0, 2, 3, 1))  # output data consistency image's kspace

            x_K = net['r%d_kspc_img_dc_pred' % i].permute(0, 3, 1, 2)
            #源代码里面是重建结束求loss，但是我认为应该dc后求loss吧;这个版本是先求loss
            ans_K[i-1] = self.net_G_K(x_K)  # output recon kspace
            I, _ = self.dcs_K[i - 1](ans_K[i-1], k, mask)  # output data consistency images

        return ans_I, ans_K, I

if __name__ == '__main__':
    x = torch.rand((2,2,256,256))
    k = torch.rand((2,256,256,2))
    mask = torch.rand((2,1,256,1))
    model = RecurrentModel()
    params = sum(param.nelement() for param in model.parameters())
    print(params / 1e6)
    ans_I, ans_K, out = model(x, k, mask)
    print(len(ans_I), len(ans_K))
    print(out.shape)