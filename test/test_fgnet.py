# -*- encoding: utf-8 -*-
'''
@File    :   test_fgnet.py
@Time    :   2021/11/02 16:19:28
@Author  :   secrul 
@Version :   1.0
@Contact :   secrul@163.com
'''

# here put the import lib

from matplotlib import image
import sys
sys.path.append('/home/data/ljh/Brain_MRI/Brain_MRI/dataset')
sys.path.append('/home/data/ljh/Brain_MRI/Brain_MRI/util')
sys.path.append('/home/data/ljh/Brain_MRI/Brain_MRI/model')
sys.path.append('/home/data/ljh/Brain_MRI/Brain_MRI')
from Brain_single_dc_fgnet import dataset_Brain
import torch
from torch.nn import functional as F
import cv2
from unet import Unet
from DCNN import DCNN
import time
from fftc import ifft2c_old as ifft2c
from fftc import fft2c_old as fft2c
from loss import FocalFrequencyLoss
from HFGuidedNet.FGNet import FGNet
from HFGuidedNet.FGNet_less import FGNetL
from HFGuidedNet.FGNet_OC import FGNetOC
from HFGuidedNet.FGNet_v2 import FGNetV2

import numpy as np
from skimage import measure
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from fftc import ifft2c_old as ifft2c
from math_2 import complex_abs

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import matplotlib as mpl
mpl.use('Agg')

def mse(gt, pred):
    """ Compute Mean Squared Error (MSE) """
    return np.mean((gt - pred) ** 2)


def nmse(pred, gt):
    """ Compute Normalized Mean Squared Error (NMSE) """
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(pred,gt, data_range):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    return measure.compare_psnr(gt, pred, data_range = data_range)

def msssimLoss(pred,gt):
    """
    计算MSSSIM
    """
    return piq.ms_ssim.MultiScaleSSIMLoss(gt, pred)

def ssim(pred,gt,data_range):
    """ Compute Structural Similarity Index Metric (SSIM). """
    return measure.compare_ssim(gt[0], pred[0], data_range = data_range )

def min_maxNorm(img):
    #img:tonsor,[h,w]
    minn = img.min()
    maxx = img.max()

    # return img
    return (img - minn) / (maxx - minn)


def normalize(data, mean, stddev, eps=1e-11):

    return (data - mean) / (stddev + eps)



def test(premodel, dataloader, model, epoch, writer):
    total_loss = []
    total_psnr = []
    total_ssim = []
    slic = 0
    epoch_loss = 0
    with torch.no_grad():
        for i,data in enumerate(dataloader): # 每个batch更新一次参数
            #data:kspace_data, mask_kspace, mask_image_2c, gt_img_2c, file_name, slice_num, mask, std

            std = data[7]
            std = std.float().cuda(non_blocking=True)
            gt_kspace = data[0]
            mask = data[6]
            mask_image = data[2]
            gt_image = data[3]
            mask_image = mask_image.float().cuda(non_blocking=True)
            gt_image = gt_image.float().cuda(non_blocking=True)
            gt_kspace = gt_kspace.float().cuda(non_blocking=True)
            mask = mask.float().cuda(non_blocking=True)
            # pre_out = premodel(mask_image.permute(0, 3, 1, 2), gt_kspace.permute(0, 3, 1, 2), mask).permute(0, 2, 3, 1)
            pre_out = premodel(mask_image.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            error = gt_image - pre_out
            [out, out1] = model(mask_image.unsqueeze(1), gt_kspace.unsqueeze(1), mask)
            out = complex_abs(out) * std
            gt_image = complex_abs(gt_image) * std

            out = out.detach().cpu().numpy()
            gt_image = gt_image.detach().cpu().numpy()
            # print(out.shape)
            # plt.imshow(out[0,...], cmap = 'gray')
            # plt.savefig('out1.jpg')
            # assert 2 > 8
            data_max = np.max(gt_image)
            total_loss.append(nmse(out, gt_image))
            total_psnr.append(psnr(out, gt_image, data_max))
            total_ssim.append(ssim(out, gt_image, data_max))
    writer.add_scalar('psnr', np.mean(total_psnr), epoch)
    writer.add_scalar('ssim', np.mean(total_ssim), epoch)
    return np.mean(total_loss), np.mean(total_psnr), np.mean(total_ssim)

if __name__ == '__main__':
    seed = 0
    epoch = 200

    batchsize = 3
    learnrate = 1e-4
    continue_train = False
    global start_epoch
    start_epoch = 0
    global end_epoch
    end_epoch = 0
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    vla_root = '/home/data/ljh/Brain_MRI/Val'
    train_root = '/home/data/ljh/Brain_MRI/Train'
    test_dataset = dataset_Brain(vla_root,acc=4, seed=seed)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=1,num_workers=4,pin_memory=True)
    train_dataset = dataset_Brain(train_root, seed=seed, acc=4)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batchsize,num_workers=8,pin_memory=True)
    print(len(test_dataloader))


    # criterion = FocalFrequencyLoss()
    # criterion = torch.nn.MSELoss()
    criterion = torch.nn.L1Loss()
    dataset_model_lossfunc = 'Braindc_fgnetv2_unet_image_2c_l1x4_gamma_L1_31'
    # dataset_model_lossfunc = 'Braindc_fgnetoc_image_2c_l1x4_gamma_freloss'
    log_name = '../train_log'
    log_dirname = os.path.join(log_name, dataset_model_lossfunc) # loss_log
    if not os.path.exists(log_dirname):
        os.makedirs(log_dirname)

    writer = SummaryWriter(log_dirname)
    premodel = Unet(2, 2, 32, 4, 'in').to(device)
    checkpoint_path = '/home/data/ljh/Brain_MRI/Brain_MRI/checkpoints/Braindc_unet_image_2c_l1x4_1/epoch_24.pth'
    # checkpoint_path = '/home/data/ljh/Brain_MRI/Brain_MRI/checkpoints/Braindc_DCNN_image_2c_l1X4/epoch_88.pth'
    state_dict = torch.load(checkpoint_path)
    premodel.load_state_dict(state_dict)
    premodel.eval()
    model = FGNetV2().to(device)
    # model = FGNetL().to(device)
    # model = FGNetOC().to(device)
    model.eval()
    for i in range(start_epoch, end_epoch+1):
        checkpoint_path = '/home/data/ljh/Brain_MRI/Brain_MRI/checkpoints/Braindc_fgnetv2_unet_image_2c_l1x4_gamma_L1_31/epoch_{}.pth'.format(i)
        
        state_dict = torch.load(checkpoint_path)
        model.load_state_dict(state_dict)
        val(premodel, val_dataloader, model, i, writer)
