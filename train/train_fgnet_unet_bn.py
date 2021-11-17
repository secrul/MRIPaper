from matplotlib import image
import sys
import math
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
from loss import FocalFrequencyLoss, gaussWeightLoss
from HFGuidedNet.FGNet import FGNet
from HFGuidedNet.FGNet_less import FGNetL
from HFGuidedNet.FGNet_OC import FGNetOC
from HFGuidedNet.FGNet_v2_bn import FGNetV2

import numpy as np
from skimage import measure
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
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


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels,kernel_size=kernel_size, groups=channels, bias=False, padding=kernel_size//2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    
    return gaussian_filter


def normalize(data, mean, stddev, eps=1e-11):

    return (data - mean) / (stddev + eps)


def train(model, train_dataloader, optimizer,lr_scheduler, dataset_model_lossfunc,criterion, val_dataloader, epochs, log_name = '../train_log',
                    checkpoint_name = '../checkpoints'):
    log_dirname = os.path.join(log_name, dataset_model_lossfunc) # loss_log
    if not os.path.exists(log_dirname):
        os.makedirs(log_dirname)
    save_dirname =  os.path.join(checkpoint_name, dataset_model_lossfunc)
    if not os.path.exists(save_dirname):
        os.makedirs(save_dirname)
    writer = SummaryWriter(log_dirname)
    print('starting training ---------')
    print(epochs)
    
    gamma = torch.tensor(0.1, requires_grad = True)
    for epoch in range(start_epoch, epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
        print('-' * 10)
        batch_size = train_dataloader.batch_size # 需要设置
        dt_size = len(train_dataloader.dataset)
        epoch_loss = 0
        step = 0
        for i,data in enumerate(train_dataloader): # 每个batch更新一次参数
            #data:preModel,kspace_data, mask_kspace, mask_image_2c, gt_img_2c, file_name, slice_num, mask, std
            step += 1
            pre_out = data[0]
            gt_kspace = data[1]
            mask = data[7]
            mask_image = data[3]
            gt_image = data[4]
            # print(gt_image.shape, gt_kspace.shape, mask.shape)
            #[24, 1, 320, 320]  [24, 320, 320, 2]     [24, 1, 320, 1]

            mask_image = mask_image.float().cuda(non_blocking=True)
            gt_image = gt_image.float().cuda(non_blocking=True)
            gt_kspace = gt_kspace.float().cuda(non_blocking=True)
            pre_out = pre_out.float().cuda(non_blocking=True)
            mask = mask.float().cuda(non_blocking=True)

            error = gt_image - pre_out
            [dc_out,out, out1] = model(mask_image.unsqueeze(1), gt_kspace.unsqueeze(1), mask)
            out_k = fft2c(out)
            out1_k = fft2c(out1)
            error_k = fft2c(error)
            # out = complex_abs(out) 
            # out1 = complex_abs(out1) 
            # error = complex_abs(error)
            # gt_image = complex_abs(gt_image)
            # print(error.shape)

            # error = gauss_blur(error.unsqueeze(1)).squeeze(1)
            # gamma = 0.5 - epoch * 0.02
            # gamma = max(gamma, 0)
            loss1 = criterion(out, gt_image) + criterion(out_k, gt_kspace)
            loss2 = criterion(out1, error) + criterion(out1_k, error_k)
            loss = (1 - gamma) * loss1 + gamma * loss2
            # loss = loss1
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print("{}/{},train_loss:{:.6}".format(step, (dt_size - 1) // batch_size + 1, loss.item()))
            
            
            writer.add_scalar('trian/loss:', loss.item(), epoch*dt_size+batch_size*i)
            # writer.add_scalar('trian/gamma:', gamma.item(), epoch*dt_size+batch_size*i)
        # lr_scheduler.step()
        # print(gamma)
        print("epoch %d loss:%0.6f" % (epoch, epoch_loss))
        torch.save(model.state_dict(), os.path.join(save_dirname,'epoch_{}.pth'.format(epoch)))
        torch.save(optimizer.state_dict(), os.path.join(save_dirname,'optimizer.pth'))
        model.eval()
        # torch.set_grad_enabled(False)
        print('VAL')
        nmse, psnr, ssim = val(val_dataloader, model, epoch, writer)
        print(nmse, psnr, ssim)
        # torch.set_grad_enabled(True)
        model.train()



def val(dataloader, model, epoch, writer):
    total_loss = []
    total_psnr = []
    total_ssim = []
    slic = 0
    epoch_loss = 0
    with torch.no_grad():
        for i,data in enumerate(dataloader): # 每个batch更新一次参数
            #data:preModel, kspace_data, mask_kspace, mask_image_2c, gt_img_2c, file_name, slice_num, mask, std
            pre_out = data[0]
            std = data[8]
            std = std.float().cuda(non_blocking=True)
            gt_kspace = data[1]
            mask = data[7]
            mask_image = data[3]
            gt_image = data[4]
            mask_image = mask_image.float().cuda(non_blocking=True)
            gt_image = gt_image.float().cuda(non_blocking=True)
            gt_kspace = gt_kspace.float().cuda(non_blocking=True)
            pre_out = pre_out.float().cuda(non_blocking=True)
            mask = mask.float().cuda(non_blocking=True)
            error = gt_image - pre_out
            [dcout, out, out1] = model(mask_image.unsqueeze(1), gt_kspace.unsqueeze(1), mask)
            out = complex_abs(dcout) * std
            gt_image = complex_abs(gt_image) * std

            out = out.detach().cpu().numpy()
            gt_image = gt_image.detach().cpu().numpy()

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

    batchsize = 1
    learnrate = 1e-4
    continue_train = False
    global start_epoch
    start_epoch = 0
    #k空间的损失添加一个高斯加权mask，参数sigma学习,gamma为固定值0.1

    # print("config")
    # print("""
    #     模型：在error添加高斯模糊，在dc前面做loss，k空间也是loss，max_pool,几处加改为concat
    #     最后指标计算dc之后的结果；

    # """)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    global gauss_blur
    gauss_blur = get_gaussian_kernel().cuda()
    vla_root = '/home/data/ljh/Brain_MRI/Val'
    train_root = '/home/data/ljh/Brain_MRI/Train'
    test_dataset = dataset_Brain(vla_root,premodel='unet', ty='val', acc=4, seed=seed)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=1,num_workers=4,pin_memory=True)
    train_dataset = dataset_Brain(train_root,premodel='unet', ty='train', acc=4, seed=seed)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batchsize,num_workers=8,pin_memory=True)
    print(len(test_dataloader))


    # criterion1 = gaussWeightLoss()
    # criterion = torch.nn.MSELoss()
    criterion = torch.nn.L1Loss()
    # dataset_model_lossfunc = 'Braindc_fgnetv2_unet_image_2c_l1x4_oneLoss_nobn_bn1'
    dataset_model_lossfunc = 'Braindc_fgnetv2_unet_image_2c_l1x4_gamma01_noblur_v3_bn=False'
    # dataset_model_lossfunc = 'testMemory'
    # dataset_model_lossfunc = 'Braindc_fgnetoc_image_2c_l1x4_gamma_freloss'
    
    # model = FGNet().to(device)
    model = FGNetV2().to(device)
    # model = FGNetL().to(device)
    # model = FGNetOC().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learnrate)
    if continue_train:
        # checkpoint_path = '/home/westdata/ljh/checkpoints/knee_single_dc_ffa_perceptual/epoch_{}.pth'.format(start_epoch-1)
        checkpoint_path = '/home/data/ljh/Brain_MRI/Brain_MRI/checkpoints/Braindc_fgnetv2_unet_image_2c_l1x4_gamma01_noblur_v3_nobn/epoch_10.pth'
        # checkpoint_path = '/home/westdata/ljh/checkpoints/knee_single_dc_unet_perceptual/epoch_{}.pth'.format(start_epoch-1)
        # checkpoint_path = '/home/westdata/ljh/checkpoints/knee_single_dc_unet_perceptual_bn/epoch_{}.pth'.format(start_epoch-1)
        state_dict = torch.load(checkpoint_path)
        model.load_state_dict(state_dict)
        # checkpoint_path_op = '/home/data/ljh/Brain_MRI/Brain_MRI/checkpoints/Braindc_fgnetv2_unet_image_2c_l1x4_gamma01_noblur_v3_nobn/optimizer.pth'
        # optimizer.load_state_dict(torch.load(checkpoint_path_op))
    num_params = sum(param.nelement() for param in model.parameters())
    print(num_params / 1e6)
    model.train()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=0, last_epoch=-1)
    print('vvvv')

    
    train(model, train_dataloader, optimizer, scheduler, dataset_model_lossfunc, criterion, test_dataloader,epoch)
