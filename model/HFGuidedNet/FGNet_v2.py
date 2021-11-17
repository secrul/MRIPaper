import torch
import torch.nn as nn
import sys
sys.path.append('/home/westdata/ljh/projects/Brain_MRI/Brain_MRI/util')
sys.path.append('/home/westdata/ljh/projects/Brain_MRI/Brain_MRI/model/HFGuidedNet')
from ComplexCNN import ComplexConv, CReLu, RadialBN, ComplexTransConv, ComplexPool

from fftc import ifft2c_old as ifft2c
from fftc import fft2c_old as fft2c
import copy
# from thop import profile

def DC(img, kspace, mask):
    #mask batch, 1, 320, 1;mask为1表示gt代替，为0使用预测值
    img = img.squeeze(1)
    kspace = kspace.squeeze(1) 
    pre_kspace = fft2c(img)
    ans_kspace = kspace * mask + pre_kspace *(1 - mask)

    return ifft2c(ans_kspace)


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Linear(channel, channel // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel, bias=False),
                nn.Sigmoid()
        )
        # self.conv = ComplexConv(2 * channel, channel, kernel_size=3, padding=1,act='bn')

    def forward(self, x):
        b,c,h,w,d = x.size()
        x_abs =  (x ** 2).sum(dim=-1)
        y = self.avg_pool(x_abs).view(b,c)
        y = self.conv_du(y).view(b,c,1,1)
        y = y.unsqueeze(-1)
        tmp = x * y
        return tmp


## spatial Attention (SA) Layer
class SALayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SALayer, self).__init__()

    def forward(self, x, error):

        tmp = x * error
        return tmp

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, num_feat=64, kernel_size=3, reduction=4,flag = 'HF'):
    
        super(RCAB, self).__init__()
        self.flag = flag
        self.conv1 = ComplexConv(num_feat, num_feat, kernel_size=kernel_size, padding=1)
        self.conv2 = ComplexConv(num_feat, num_feat, kernel_size=kernel_size, padding=1, act=None)
        self.conv1_k = ComplexConv(num_feat, num_feat, kernel_size=1, padding=0)
        self.conv2_k = ComplexConv(num_feat, num_feat, kernel_size=1, padding=0, act=None)
        self.fuse = ComplexConv(num_feat * 3, num_feat, kernel_size=1, padding=0, act=None)
        modules_body = []
        modules_body.append(self.conv1)
        modules_body.append(self.conv2)
        modules_body.append(CALayer(num_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        if self.flag != 'HF':
            self.sa = SALayer(num_feat)
            self.sa_k = SALayer(num_feat)
        modules_body_k = []
        modules_body_k.append(self.conv1_k)
        modules_body_k.append(self.conv2_k)
        modules_body_k.append(CALayer(num_feat, reduction))
        self.body_k = nn.Sequential(*modules_body_k)
        

    def forward(self, x, error_i=None, error_k=None):
        x_k = fft2c(x)
        res_k = self.body_k(x_k)
        res = self.body(x)
        if self.flag != 'HF':
            res_k = self.sa_k(res_k, error_k)
            res = self.sa(res, error_i)
        return self.fuse(torch.cat([res, x, ifft2c(res_k)],dim=1))
        


class FGNetV2(nn.Module):
    def __init__(
        self, in_c = 1, out_c = 1, num_feat=64, kernel_size=3, reduction=4,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(FGNetV2, self).__init__()
        self.feature_ext = nn.Sequential(
            ComplexConv(in_c * 2, num_feat // 2, kernel_size = 3, stride=1, padding = 1),
            ComplexConv(num_feat // 2, num_feat, kernel_size = 3, stride=1, padding = 1)
        )

        self.last = nn.Sequential(
            ComplexConv(num_feat, num_feat // 2, kernel_size = 3, stride=1, padding = 1),
            #1卡是act为True，3卡对比为False
            ComplexConv(num_feat // 2, 1, kernel_size = 3, stride=1, padding = 1)
        )
        self.b1 = RCAB(flag='Recon')
        self.b2 = RCAB(flag='Recon')
        self.b3 = RCAB(flag='Recon')
        self.b4 = RCAB(flag='Recon')
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
        self.HF = nn.Sequential(
            ComplexConv(in_c, num_feat // 2, kernel_size = 3, stride=1, padding = 1),
            ComplexConv(num_feat // 2, num_feat, kernel_size = 3, stride=1, padding = 1),
            RCAB(flag='HF'),
            RCAB(flag='HF')
        )
        self.afterHF = nn.Sequential(
            ComplexConv(num_feat, num_feat // 2, kernel_size = 3, stride=1, padding = 1),
            ComplexConv(num_feat // 2, 1, kernel_size = 3, stride=1, padding = 1)
        )

        self.convf = ComplexConv(num_feat * 2, num_feat, kernel_size = 3, stride=1, padding = 1)
        self.sig = nn.Sigmoid()


    def cal_sa_mask(self, error):
        error = self.sig((error ** 2).sum(dim=-1))
        # avgout = torch.mean(error, dim=1, keepdim=True)
        maxout, _ = torch.max(error, dim=1, keepdim=True)
        # y = torch.cat([avgout, maxout], dim=1)
        y = maxout
        y = y.unsqueeze(-1)
        # print(y.shape)
        # assert 1 > 4
        return y

    def forward(self, x, k, mask):
        # x = torch.cat([x, error], dim=1)
        x_HF = self.HF(x)
        out1 = self.afterHF(x_HF)
        # print(tmp.shape)
        x0 = self.feature_ext(torch.cat([x, out1], dim=1))
        x0 = torch.cat([x0, x_HF], dim=1)
        x0 = self.convf(x0)
        out1_k = fft2c(out1)
        out1_s_i = self.cal_sa_mask(out1)
        out1_s_k = self.cal_sa_mask(out1_k)
        
        x1 = x0 + self.b1(x0,out1_s_i, out1_s_k)
        x2 = x1 + self.b2(x1,out1_s_i, out1_s_k)
        x3 = x2 + self.b3(x2,out1_s_i, out1_s_k)
        x4 = x3 + self.b4(x3,out1_s_i, out1_s_k)
        output = self.last(x4)
        return DC(output, k, mask), output, out1


if __name__ == '__main__':
    model = FGNetV2(in_c=1, out_c=1)
    num_params = sum(param.nelement() for param in model.parameters())
    print(num_params / 1e6)
    x = torch.rand((2,1,256,256,2))
    k = torch.rand((2,1,256,256,2))
    error = torch.rand((2,1,256,256,2))
    mask = torch.rand((2,1,256,1))
    [dc,out,out1] = model(x, k, mask)
    print(out.shape)