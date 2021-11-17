from matplotlib import image
import sys
sys.path.append('/home/data/ljh/fastmri/MRI/dataset')
sys.path.append('/home/data/ljh/fastmri/MRI/util')
sys.path.append('/home/data/ljh/fastmri/MRI/model')
sys.path.append('/home/data/ljh/fastmri/MRI/loss')
from Perceptual_Loss import Vgg16Loss
from knee_single import dataset_fastMRI
import torch
import torch.nn as nn
from torch.nn import functional as F
import cv2
from unet import Unet
from unet_dc import Unet_dc
from FFA import FFA
import numpy as np
from skimage import metrics
import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
import matplotlib.pyplot as plt
from fftc import ifft2c_old as ifft2c
from math_2 import complex_abs
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
# from swinir import SwinIR

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def mse(gt, pred):
    """ Compute Mean Squared Error (MSE) """
    return np.mean((gt - pred) ** 2)


def nmse(pred, gt):
    """ Compute Normalized Mean Squared Error (NMSE) """
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(pred,gt,data_range):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    return metrics.peak_signal_noise_ratio(gt, pred, data_range = data_range)

def msssimLoss(pred,gt):
    """
    计算MSSSIM
    """
    return piq.ms_ssim.MultiScaleSSIMLoss(gt, pred)

def ssim(pred,gt,data_range):
    """ Compute Structural Similarity Index Metric (SSIM). """
    return metrics.structural_similarity(gt[0], pred[0], data_range = data_range)

def min_maxNorm(img):
    #img:tonsor,[h,w]
    minn = img.min()
    maxx = img.max()

    # return img
    return (img - minn) / (maxx - minn)

def train(model, train_dataloader, optimizer,lr_scheduler, dataset_model_lossfunc,criterion, val_dataloader, epochs, log_name = '../train_log',
                    checkpoint_name = '../checkpoints'):
    log_dirname = os.path.join(log_name, dataset_model_lossfunc) # loss_log
    if not os.path.exists(log_dirname):
        os.makedirs(log_dirname)
    save_dirname =  os.path.join(checkpoint_name, dataset_model_lossfunc)
    if not os.path.exists(save_dirname):
        print('T*******')
        os.makedirs(save_dirname)
    writer_dir = os.path.join('../train_log/', dataset_model_lossfunc)
    writer = SummaryWriter(writer_dir)
    print('starting training ---------')
    print(epochs)
    for epoch in range(start_epoch, epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)
        batch_size = train_dataloader.batch_size # 需要设置
        dt_size = len(train_dataloader.dataset)
        epoch_loss = 0
        step = 0
        for i,data in enumerate(train_dataloader): # 每个batch更新一次参数
            step += 1
            mask_image = data[2]
            gt_image = data[3]
            if flag == 'kspace':
                mask_image = mask_kspace.float().cuda()
                gt_image = gt_kspace.float().cuda()
                out = model(mask_image)
            else:
                mask_image = mask_image.float().cuda()
                gt_image = gt_image.float().cuda()
                # print(mask_image.size())
                out = model(mask_image)
            if criterion_type == 'perceptual':
                out_t = torch.unsqueeze(out,1)
                gt_image_t = torch.unsqueeze(gt_image,1)

                out_3 = torch.cat([out_t, out_t, out_t],1)
                gt_image_3 = torch.cat([gt_image_t, gt_image_t, gt_image_t],1)
                loss = criterion(out_3, gt_image_3)
            if criterion_type == 'l1':
                loss = criterion(out, gt_image)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            writer.add_scalar('loss', loss.item(), (epoch-start_epoch-1)*len(train_dataloader)+i)
            print("{}/{},train_loss:{:.6}".format(step, (dt_size - 1) // batch_size + 1, loss.item()))


        # lr_scheduler.step()
        print("epoch %d loss:%0.6f" % (epoch, epoch_loss))
        torch.save(model.state_dict(), os.path.join(save_dirname,'epoch_{}.pth'.format(epoch)))

        model.eval()
        torch.set_grad_enabled(False)
        print('VAL')
        epoch_loss, nmse, psnr, ssim = val(val_dataloader, model, criterion, epoch, writer)
        print(epoch_loss, nmse, psnr, ssim)
        torch.set_grad_enabled(True)
        model.train()



def val(dataloader, model, criterion, epoch, writer):
    total_loss = []
    total_psnr = []
    total_ssim = []
    slic = 0
    epoch_loss = 0
    for i,data in enumerate(dataloader): # 每个batch更新一次参数

        mask_image = data[2]
        gt_image = data[3]
        mean = data[7]
        std = data[8]
        data_max = data[9]
        # print(mean, std)

        if flag == 'kspace':
            mask_image = mask_kspace.float().cuda()
            gt_image = gt_kspace.float().cuda()
            mean = mean.cuda()
            std = std.cuda()
            out = model(mask_image)
        else:
            mask_image = mask_image.float().cuda()
            gt_image = gt_image.float().cuda()
            mean = mean.cuda()
            std = std.cuda()
            out = model(mask_image)
        out = out * std + mean
        gt_image = gt_image * std + mean
        out = out.detach().cpu().numpy()[0,0,:,:]
        gt_image = gt_image.detach().cpu().numpy()[0,0,:,:]
        data_max = data_max.detach().cpu().item()
        total_loss.append(nmse(out, gt_image))
        total_psnr.append(psnr(out, gt_image, data_max))
        total_ssim.append(ssim(out, gt_image, data_max))
        # slic += gt_image.shape[0]
    writer.add_scalar('psnr', np.mean(total_psnr), epoch)
    writer.add_scalar('ssim', np.mean(total_ssim), epoch)
    return epoch_loss, np.mean(total_loss), np.mean(total_psnr), np.mean(total_ssim)

if __name__ == '__main__':
    seed = 0
    epoch = 100
    
    # device_ids = [0, 1]
    batchsize = 28#unet_l1
    # batchsize = 1 #ffa
    # batchsize = 16 #unet_pLoss
    learnrate = 1e-4
    global criterion_type
    continue_train = False
    global start_epoch
    start_epoch = 0
    global flag
    flag = 'image'

    criterion_type = 'l1'
    # criterion_type = 'perceptual'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    vla_root = '/home/data/ljh/fastmri/knee_singlecoil/singlecoil_val'
    train_root = '/home/data/ljh/fastmri/knee_singlecoil/singlecoil_train'
    test_dataset = dataset_fastMRI(vla_root, seed)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=1)
    train_dataset = dataset_fastMRI(train_root, seed)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batchsize)
    print(len(test_dataloader))


    criterion = torch.nn.L1Loss().cuda()
    # criterion = Vgg16Loss('l1').to(device)
    # dataset_model_lossfunc = 'knee_single_dc_unet_image_gtkspace_stdNorm_l1'
    # dataset_model_lossfunc = 'knee_single_dc_unet_image_gtkspace_0-1Norm_l1'
    dataset_model_lossfunc = 'kneesingle_unet_image_1c_l1_LR_withoutLR'
    # dataset_model_lossfunc = 'knee_single_dc_unet_perceptual'
    # dataset_model_lossfunc = 'knee_single_dc_unet_perceptual_bn'
    # dataset_model_lossfunc = 'knee_single_dc_ffa_perceptual'

    # model = Unet(1, 1, 32, 4, 'in')
    model = Unet(1, 1, 32, 4, 'in').to(device)
    # model = nn.DataParallel(model,device_ids=device_ids)
    # model = Unet(2, 2, 32, 4, 'bn').to(device)
    # model = FFA(gps=3,blocks=19).to(device)

    if continue_train:
        # checkpoint_path = '/home/westdata/ljh/checkpoints/knee_single_dc_ffa_perceptual/epoch_{}.pth'.format(start_epoch-1)
        # checkpoint_path = '/home/westdata/ljh/checkpoints/knee_single_dc_unet_l1/epoch_{}.pth'.format(start_epoch-1)
        checkpoint_path = '/home/westdata/ljh/checkpoints/knee_single_dc_unet_image_noNorm_l1/epoch_{}.pth'.format(start_epoch-1)
        # checkpoint_path = '/home/westdata/ljh/checkpoints/knee_single_dc_unet_perceptual_bn/epoch_{}.pth'.format(start_epoch-1)
        state_dict = torch.load(checkpoint_path)
        model.load_state_dict(state_dict)
    num_params = sum(param.nelement() for param in model.parameters())
    print(num_params / 1e6)
    optimizer = torch.optim.Adam(model.parameters(), lr=learnrate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=0, last_epoch=-1)
    # epoch_loss, nmse, psnr, ssim = val(test_dataloader, model, criterion)
    # print(epoch_loss, nmse, psnr, ssim)
    # assert 1 > 4
    train(model, train_dataloader, optimizer, scheduler, dataset_model_lossfunc, criterion, test_dataloader,epoch)
