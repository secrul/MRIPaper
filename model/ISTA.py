import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" #为什么调用plt.imshow()时jupyter notebook崩溃? - 知乎
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import sys
sys.path.append('/home/data/ljh/fastmri/MRI/util')
from fftc import ifft2c_old as ifft2c
from fftc import fft2c_old as fft2c

"""
ISTA-Net: Interpretable optimization-inspired deep network for image compressive sensing
"""


class FFT_Mask_ForBack(torch.nn.Module):
    def __init__(self):
        super(FFT_Mask_ForBack, self).__init__()

    def forward(self, x, mask):
        x = x.permute(0,2,3,1)
        fftz = fft2c(x)
        z_hat = ifft2c(fftz * mask)
        # x = z_hat[:, :, :, 0:1]
        x = x.permute(0,3,1,2)
        return x


# Define ISTA-Net-plus Block
class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))


        self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 2, 3, 3)))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))

        self.conv_G = nn.Parameter(init.xavier_normal_(torch.Tensor(2, 32, 3, 3)))

    def forward(self, x, fft_forback, PhiTb, mask):
        x = x - self.lambda_step * fft_forback(x, mask)
        x = x + self.lambda_step * PhiTb
        x_input = x

        x_D = F.conv2d(x_input, self.conv_D, padding=1)

        x = F.conv2d(x_D, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)

        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))

        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)

        x_G = F.conv2d(x_backward, self.conv_G, padding=1)

        x_pred = x_input + x_G

        x = F.conv2d(x_forward, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_D_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_D_est - x_D

        return [x_pred, symloss]


# Define ISTA-Net-plus
class ISTANetplus(torch.nn.Module):
    def __init__(self, LayerNo = 10):
        super(ISTANetplus, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo
        self.fft_forback = FFT_Mask_ForBack()

        for i in range(LayerNo):
            onelayer.append(BasicBlock())

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, PhiTb, mask):

        x = PhiTb

        layers_sym = []   # for computing symmetric loss

        for i in range(self.LayerNo):
            [x, layer_sym] = self.fcs[i](x, self.fft_forback, PhiTb, mask)
            layers_sym.append(layer_sym)

        x_final = x

        return [x_final, layers_sym]
        
if __name__ == '__main__':
    x = torch.rand(2,2,256,256)
    mask = torch.rand(2,1,256,1)
    model = ISTANetplus()
    num_params = sum(param.nelement() for param in model.parameters())
    print(num_params / 1e6)
    [x_output, loss_layers_sym] = model(x,mask)
    print(x_output.shape)
    print(loss_layers_sym)