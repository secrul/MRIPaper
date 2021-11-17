import contextlib
import torch
from torch.utils import data
import os
import math
from PIL import Image
import numpy as np
import h5py
import cv2
import random
from typing import Dict, Optional, Sequence, Tuple, Union
import sys
sys.path.append('/home/westdata/ljh/projects/Brain_MRI/Brain_MRI/util')
# pytorch >= 1.7.0使用ifft2c_new
from fftc import ifft2c_old as ifft2c
from fftc import fft2c_old as fft2c
from math_2 import complex_abs
import cv2
"""
import matplotlib as mpl
mpl.use('Agg')
plt.imshow(gt, cmap ='gray')

plt.savefig('gtplt.jpg')
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')


rng = np.random.RandomState()


@contextlib.contextmanager
def temp_seed(rng: np.random, seed=None):
    if seed is None:
        try:
            yield
        finally:
            pass
    else:
        state = rng.get_state()
        rng.seed(seed)
        try:
            yield
        finally:
            rng.set_state(state)


def normalize(data, mean, stddev, eps=0.0):

    return (data - mean) / (stddev + eps)


def normalize_instance(data, eps=0.0):

    mean = data.mean()
    std = data.std()

    return normalize(data, mean, std, eps), mean, std


def complex_to_tensor(data: np.ndarray) -> torch.Tensor:

    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)
    data = data.astype(np.float32)
    return torch.from_numpy(data)


def mask_func_random_unique(shape, acc, seed=None):
    """
    Args:
        shape:[320, 320, 2]

    Return:
        [1, 320, 1]非0即1的tensor
    """
    if len(shape) < 3:
        raise ValueError("Shape should have 3 or more dimensions")

    with temp_seed(rng, seed):
        num_cols = shape[-2]
        if acc == 4:
            center_fraction, acceleration = 0.08, 4#中心采样比例，加速比
        elif acc == 8:
            center_fraction, acceleration = 0.04, 8#中心采样比例，加速比
        else:
            print('the acc factor must be 4 or 8')
        # create the mask
        num_low_freqs = int(round(num_cols * center_fraction))
        #
        #(采样条数-中心采样条数) / 所有未采样条数 计算每一行的采样概率
        #
        prob = (num_cols / acceleration - num_low_freqs) / (
                num_cols - num_low_freqs
        )
        mask = rng.uniform(size=num_cols) < prob
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad: pad + num_low_freqs] = True

        # reshape the mask
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = num_cols
        mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float64)) # mask.shape=[col, 1]
        # print(mask.shape)
        # assert 3> 4
    return mask


def transform(kspace_data, acc, normType = 'std', seed=None):
    """
    Args:
        kspace_data:numpy,[256,256,2],复实
    Returns:
        mask_kspace:tensor:叠加mask后的kspace，两个通道[256,256,2]
        mask:tensor;[1,256,1]
        mask_image:tensor;[256,256,2]
    """

    mask = mask_func_random_unique([256,256,2], acc, seed)
    img = torch.ifft(kspace_data, 2, normalized=True)

    kspace_data = fft2c(img)
    gt_img_2c = ifft2c(kspace_data)
    gt_img = complex_abs(gt_img_2c)
    std = torch.std(gt_img)

    gt_img_2c /= std

    kspace_data = fft2c(gt_img_2c)

    # kspace_data /= std

    # print(kspace_data.dtype)
    # print(mask.dtype)
    mask_kspace = kspace_data * mask
    mask_image_2c = ifft2c(mask_kspace)
    
    return kspace_data, mask_kspace, mask, mask_image_2c, gt_img_2c, std


def norm_kspace(kspace, std):

    img = ifft2c(complex_to_tensor(kspace)) / std

    kspace_norm = fft2c(img)
    return kspace_norm

def min_maxNorm(img):
    #img:tonsor,[h,w]
    minn = img.min()
    maxx = img.max()

    # return img
    return (img - minn) / (maxx - minn)

class dataset_Brain(data.Dataset):
    """
    采用的是先裁剪，再加mask的形式，对于gt进行0-1归一化
    因为返回了gt_kspace，所以可以用来进行dc
    """
    def __init__(self, root, premodel = 'unet', ty = 'val', acc=4, seed=None):
        self.seed = seed
        self.acc = acc
        self.premodel = premodel
        self.type = ty
        list_files = os.listdir(root)
        self.list_files = [os.path.join(root,list_file) for list_file in list_files]
        self.examples = []
        ct = 0
        for fe in self.list_files:
            # ct += 1
            # if ct < 2:
                npyfile = np.load(fe)
                # print(npyfile.shape)
                # assert 1 > 8
                for slice_num in range(npyfile.shape[0]):
                    self.examples.append([fe, slice_num])
        print("all slices:",len(self.examples))

    def __getitem__(self, index: int):
        file_name = self.examples[index][0]
        slice_num = self.examples[index][1]
        npyfile = np.load(file_name)
        pre_F = ''
        if self.premodel == 'unet':
            if self.type == 'val':
                if self.acc == 4:
                    pre_F = '/home/westdata/ljh/projects/Brain_MRI/ValUnetx4'
                else:
                    pre_F = '/home/westdata/ljh/projects/Brain_MRI/ValUnetx8'
            if self.type == 'train':        
                if self.acc == 4:
                    pre_F = '/home/westdata/ljh/projects/Brain_MRI/TrainUnetx4'
                else:
                    pre_F = '/home/westdata/ljh/projects/Brain_MRI/TrainUnetx8'
            if self.type == 'test':        
                if self.acc == 4:
                    pre_F = '/home/westdata/ljh/projects/Brain_MRI/TestUnetx4'
                else:
                    pre_F = '/home/westdata/ljh/projects/Brain_MRI/TestUnetx8'
        if self.premodel == 'dcnn':
            if self.type == 'val':
                if self.acc == 4:
                    pre_F = '/home/westdata/ljh/projects/Brain_MRI/ValDcnnx4'
                else:
                    pre_F = '/home/westdata/ljh/projects/Brain_MRI/ValDcnnx8'
            if self.type == 'train':        
                if self.acc == 4:
                    pre_F = '/home/westdata/ljh/projects/Brain_MRI/TrainDcnnx4'
                else:
                    pre_F = '/home/westdata/ljh/projects/Brain_MRI/TrainDcnnx8'
            if self.type == 'test':        
                if self.acc == 4:
                    pre_F = '/home/westdata/ljh/projects/Brain_MRI/TestDcnnx4'
                else:
                    pre_F = '/home/westdata/ljh/projects/Brain_MRI/TestDcnnx8'
        kspace = torch.from_numpy(npyfile[slice_num])

        preOut = torch.from_numpy(np.load(os.path.join(pre_F, file_name.split('/')[-1]+str(slice_num)+'.npy')))
        # print(preOut.shape)
        kspace_data, mask_kspace, mask, mask_image_2c, gt_img_2c, std = transform(kspace, self.acc, self.seed)
        # print(gt_img_2c.shape)
        # assert 2 > 8
        sample = preOut.squeeze(0), kspace_data, mask_kspace, mask_image_2c, gt_img_2c, file_name, slice_num, mask, std

        return sample

    def __len__(self):
        return len(self.examples)


if __name__ == '__main__':
    path = '/home/westdata/ljh/projects/Brain_MRI/Train'
    datas = dataset_Brain(path, ty='train', acc=4)
    dataloader = torch.utils.data.DataLoader(datas, batch_size=1, shuffle=False)
    for data in dataloader:
        kspace_data = data[0]
        mask_kspace = data[1]
        mask_image_2c = data[2]
        gt_img_2c = data[3]
        print(gt_img_2c.shape, mask_image_2c.shape)
        #torch.Size([1, 256, 256, 2]) torch.Size([1, 256, 256, 2])
        assert 1 <0