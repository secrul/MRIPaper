# -*- encoding: utf-8 -*-
'''
@File    :   test_dcnn.py
@Time    :   2021/11/02 16:18:57
@Author  :   secrul 
@Version :   1.0
@Contact :   secrul@163.com
'''

# here put the import lib

from matplotlib import image
import sys
sys.path.append('/home/data/ljh/Brain_MRI/Brain_MRI/dataset')
sys.path.append('/home/data/ljh/Brain_MRI/Brain_MRII/util')
sys.path.append('/home/data/ljh/Brain_MRI/Brain_MRI/model')
sys.path.append('/home/data/ljh/Brain_MRI/Brain_MRI/loss')
from Brain_single_dc import dataset_Brain
# from test_one_ljh import dataset_Brain
import torch
from torch.nn import functional as F
import time
from DCNN import DCNN

import numpy as np
from skimage import measure
import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from fftc import ifft2c_old as ifft2c
from math_2 import complex_abs
import matplotlib as mpl
mpl.use('Agg')
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
import matplotlib
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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


def test(model, test_dataloader, epochs,dataset_model_lossfunc):

    save_img = '/home/data/ljh/Brain_MRI/Brain_MRI/test_all_imgs'
    save_img = os.path.join(save_img, dataset_model_lossfunc)
    if not os.path.exists(save_img):
        os.mkdir(save_img)
    save_gt = '/home/data/ljh/Brain_MRI/Brain_MRI/test_all_imgs/gt'
    save_zero = '/home/data/ljh/Brain_MRI/Brain_MRI/test_all_imgs/zero-fillx8'
    print('starting training ---------')
    print(epochs)
    total_loss = []
    total_psnr = []
    total_ssim = []
    slic = 0
    epoch_loss = 0
    print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    print('-' * 10)

    epoch_loss = 0
    step = 0
    for i,data in enumerate(test_dataloader): # 每个batch更新一次参数
        #data:kspace_data, mask_kspace, mask_image_2c, gt_img_2c, file_name, slice_num, mask, std
        #data:kspace_data, mask_kspace, mask_image_2c, gt_img_2c, file_name, slice_num, mask, std
        file_name = data[4][0].split('/')[-1]
        slice_num = data[5].item()
        gt_kspace = data[0].permute(0, 3, 1, 2)
        mask = data[6]
        std = data[7]
        mask_image = data[2].permute(0, 3, 1, 2)
        gt_image = data[3]
        mask_image = mask_image.float().cuda(non_blocking=True)
        mask = mask.float().cuda(non_blocking=True)
        gt_kspace = gt_kspace.float().cuda(non_blocking=True)
        gt_image = gt_image.float().cuda(non_blocking=True)
        std = std.float().cuda(non_blocking=True)
        out = model(mask_image, gt_kspace, mask).permute(0, 2, 3, 1)
        # # out = mask_image.permute(0, 2, 3, 1)
        out = complex_abs(out) * std
        gt_image = complex_abs(gt_image) * std
        mask_image = complex_abs(mask_image.permute(0, 2, 3, 1)) * std
        out = out.detach().cpu().numpy()
        gt_image = gt_image.detach().cpu().numpy()
        mask_image = mask_image.detach().cpu().numpy()
        # print(out.shape, gt_image.shape)
        # assert 1 > 8
        # matplotlib.image.imsave(os.path.join(save_img,file_name+str(slice_num)+'result.png'), out[0,...],cmap='gray')
        # matplotlib.image.imsave(os.path.join(save_gt,file_name+str(slice_num)+'.png'), gt_image[0,...],cmap='gray')
        # matplotlib.image.imsave(os.path.join(save_img,file_name+str(slice_num)+'error.png'), out[0,...]-gt_image[0,...],cmap='bwr')
        # matplotlib.image.imsave(os.path.join(save_zero,file_name+str(slice_num)+'.png'), mask_image[0,...],cmap='gray')
 
        total_loss.append(nmse(out, gt_image))
        data_max = gt_image.max()
        tmp_psnr = psnr(out, gt_image, data_max)
        tmp_ssim = ssim(out, gt_image, data_max)
        print(file_name, slice_num,tmp_psnr, tmp_ssim)
        total_psnr.append(tmp_psnr)
        total_ssim.append(tmp_ssim)
    #     # slic += gt_image.shape[0]
    writer.add_scalar('psnr', np.mean(total_psnr), epoch)
    writer.add_scalar('ssim', np.mean(total_ssim), epoch)
    print(np.mean(total_loss), np.mean(total_psnr), np.mean(total_ssim))



if __name__ == '__main__':
    seed = 0
    epoch = 200

    batchsize = 1#unet_l1
    continue_train = False
    global start_epoch
    start_epoch = 87
    global end_epoch
    end_epoch = 87

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    test_root = '/home/data/ljh/Brain_MRI/Test/Test'
    test_dataset = dataset_Brain(test_root,acc=8,  seed = seed)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=1,num_workers=4,pin_memory=True)
    print(len(test_dataloader))

    layer_num = 10
    criterion = torch.nn.MSELoss()
    # dataset_model_lossfunc = 'zero-fill_l1x8'
    dataset_model_lossfunc = 'Braindc_dcnn_image_2c_l1x8'
    log_name = '../test_log'
    log_dirname = os.path.join(log_name, dataset_model_lossfunc) # loss_log
    if not os.path.exists(log_dirname):
        os.makedirs(log_dirname)
    writer = SummaryWriter(log_dirname) 
    model = DCNN(5, 5).to(device)
    for i in range(start_epoch, end_epoch+1):
        checkpoint_path = '/home/data/ljh/Brain_MRI/Brain_MRI/checkpoints/Braindc_DCNN_image_2c_l1X8/epoch_{}.pth'.format(i)
        state_dict = torch.load(checkpoint_path)
        model.load_state_dict(state_dict)
        model.eval()

        test(model, test_dataloader,i,dataset_model_lossfunc)
        assert 1 > 8
