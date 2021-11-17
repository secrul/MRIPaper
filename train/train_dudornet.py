from matplotlib import image
import sys
sys.path.append('/home/data/ljh/Brain_MRI/Brain_MRI/dataset')
sys.path.append('/home/data/ljh/Brain_MRI/Brain_MRI/util')
sys.path.append('/home/data/ljh/Brain_MRI/Brain_MRI/model/DuDoRNet')
from Brain_single_dc import dataset_Brain
import torch
from torch.nn import functional as F
import time
from du_recurrent_model import RecurrentModel

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


def train(model, train_dataloader, optimizer,lr_scheduler, dataset_model_lossfunc,criterion, val_dataloader, epochs, log_name = '/home/data/ljh/Brain_MRI/Brain_MRI/train_log',
                    checkpoint_name = '/home/data/ljh/Brain_MRI/Brain_MRI/checkpoints'):
    log_dirname = os.path.join(log_name, dataset_model_lossfunc) # loss_log
    if not os.path.exists(log_dirname):
        os.makedirs(log_dirname)
    save_dirname =  os.path.join(checkpoint_name, dataset_model_lossfunc)
    if not os.path.exists(save_dirname):
        os.makedirs(save_dirname)
    writer = SummaryWriter(log_dirname)
    print('starting training ---------')
    print(epochs)
    for epoch in range(start_epoch, epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
        print('-' * 10)
        batch_size = train_dataloader.batch_size # 需要设置
        dt_size = len(train_dataloader.dataset)
        epoch_loss = 0
        step = 0
        for i,data in enumerate(train_dataloader): # 每个batch更新一次参数
            #data:kspace_data, mask_kspace, mask_image_2c, gt_img_2c, file_name, slice_num, mask, std
            step += 1
            gt_kspace = data[0]
            mask = data[6]
            mask_image = data[2].permute(0, 3, 1, 2)
            kspace_norm = data[1]
            gt_image_2c =data[3]

            mask_image = mask_image.float().cuda(non_blocking=True)
            gt_image_2c = gt_image_2c.float().cuda(non_blocking=True)
            gt_kspace = gt_kspace.float().cuda(non_blocking=True)
            mask = mask.float().cuda(non_blocking=True)
            # print(mask_image.shape, gt_kspace.shape, mask.shape)
            # ans_I, ans_K, I = model(mask_image, gt_kspace, mask)
            ans_I, ans_K = model(mask_image, gt_kspace, mask)
            # gt_image = complex_abs(gt_image_2c)
            # gt_kspace = complex_abs(gt_kspace)
            loss = 0
            for tmp in range(n_recurrent):
                loss += criterion(ans_I[tmp].permute(0,2,3,1), gt_image_2c)
                loss += criterion(ans_K[tmp].permute(0,2,3,1), gt_image_2c)
                # loss += criterion(complex_abs(ans_I[tmp].permute(0,2,3,1)), gt_image)
                # loss += criterion(complex_abs(ans_K[tmp].permute(0,2,3,1)), gt_image)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            print("{}/{},train_loss:{:.6}".format(step, (dt_size - 1) // batch_size + 1, loss.item()))
            
            
            writer.add_scalar('trian/loss:', loss.item(), epoch*dt_size+batch_size*i)
        # lr_scheduler.step()
        print("epoch %d loss:%0.6f" % (epoch, epoch_loss))
        torch.save(model.state_dict(), os.path.join(save_dirname,'epoch_{}.pth'.format(epoch)))
        torch.save(optimizer.state_dict(), os.path.join(save_dirname,'optimizer.pth'))
        model.eval()
        torch.set_grad_enabled(False)
        print('VAL')
        nmse, psnr, ssim = val(val_dataloader, model, epoch, writer)
        print(nmse, psnr, ssim)
        torch.set_grad_enabled(True)
        model.train()



def val(dataloader, model, epoch, writer):
    total_loss = []
    total_psnr = []
    total_ssim = []
    slic = 0
    epoch_loss = 0
    for i,data in enumerate(dataloader): # 每个batch更新一次参数
        #data:mask_kspace, kspace_crop,  mask_image_2c, gt_image, mask, file_name, slice_num, mean, std, attributes['max']

        std = data[7]
        std = std.float().cuda(non_blocking=True)
        gt_kspace = data[0]
        mask = data[6]
        mask_image = data[2].permute(0, 3, 1, 2)
        gt_image_2c = data[3]
        mask_image = mask_image.float().cuda(non_blocking=True)
        gt_image_2c = gt_image_2c.float().cuda(non_blocking=True)
        gt_kspace = gt_kspace.float().cuda(non_blocking=True)
        mask = mask.float().cuda(non_blocking=True)
        
        # ans_I, ans_K, I = model(mask_image, gt_kspace, mask)
        ans_I, ans_K = model(mask_image, gt_kspace, mask)
        # print(out.shape)
        # assert 1 > 5
        # out = model(mask_image).permute(0, 2, 3, 1)
        out = complex_abs(ans_K[-1].permute(0, 2, 3, 1)) * std
        # out = complex_abs(I.permute(0, 2, 3, 1)) * std
        gt_image = complex_abs(gt_image_2c) * std

        out = out.detach().cpu().numpy()
        gt_image = gt_image.detach().cpu().numpy()
        data_max = np.max(gt_image)
        total_loss.append(nmse(out, gt_image))
        total_psnr.append(psnr(out, gt_image, data_max))
        total_ssim.append(ssim(out, gt_image, data_max))
        # slic += gt_image.shape[0]
    writer.add_scalar('psnr', np.mean(total_psnr), epoch)
    writer.add_scalar('ssim', np.mean(total_ssim), epoch)
    return np.mean(total_loss), np.mean(total_psnr), np.mean(total_ssim)

if __name__ == '__main__':
    seed = 0
    epoch = 200

    batchsize = 5
    learnrate = 1e-4
    global criterion_type
    continue_train = False
    global start_epoch
    start_epoch = 0
    global n_recurrent
    n_recurrent = 5
    

    criterion_type = 'l1'
    # criterion_type = 'perceptual'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    vla_root = '/home/data/ljh/Brain_MRI/Val'
    train_root = '/home/data/ljh/Brain_MRI/Train'
    test_dataset = dataset_Brain(vla_root, acc=4, seed = seed)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=1,num_workers=4,pin_memory=True)
    train_dataset = dataset_Brain(train_root, acc=4, seed = seed)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batchsize,num_workers=8,pin_memory=True)
    print(len(test_dataloader))


    criterion = torch.nn.L1Loss()
    # criterion = Vgg16Loss('l1').to(device)
    # dataset_model_lossfunc = 'knee_single_dc_unet_dc_image_noNorm_l1'
    # dataset_model_lossfunc = 'kneesingledc_unetDC_image_2c_l1_withoutLR'
    dataset_model_lossfunc = 'Braindc_dudornet_image_2c_l1x4_lossafter'
    
    model = RecurrentModel(n_recurrent).to(device)

    if continue_train:
        # checkpoint_path = '/home/westdata/ljh/checkpoints/knee_single_dc_ffa_perceptual/epoch_{}.pth'.format(start_epoch-1)
        checkpoint_path = '/home/westdata/ljh/checkpoints/knee_single_dc_unet_dc_l1/epoch_{}.pth'.format(start_epoch-1)
        # checkpoint_path = '/home/westdata/ljh/checkpoints/knee_single_dc_unet_perceptual/epoch_{}.pth'.format(start_epoch-1)
        # checkpoint_path = '/home/westdata/ljh/checkpoints/knee_single_dc_unet_perceptual_bn/epoch_{}.pth'.format(start_epoch-1)
        state_dict = torch.load(checkpoint_path)
        model.load_state_dict(state_dict)
    num_params = sum(param.nelement() for param in model.parameters())
    print(num_params / 1e6)
    optimizer = torch.optim.Adam(model.parameters(), lr=learnrate,betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=0, last_epoch=-1)
    print('vvvv')
    # nmse, psnr, ssim = val(test_dataloader, model, criterion)

    train(model, train_dataloader, optimizer, scheduler, dataset_model_lossfunc, criterion, test_dataloader,epoch)
