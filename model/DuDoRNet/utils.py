import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

'''
K-Space
'''
def data_consistency(k, k0, mask, noise_lvl=None):
    """
    k    - input in k-space
    k0   - initially sampled elements in k-space
    mask - corresponding nonzero location
    """
    v = noise_lvl
    if v:  # noisy case
        out = (1 - mask) * k + mask * (k + v * k0) / (1 + v)
    else:  # noiseless case
        out = (1 - mask) * k + mask * k0
    return out


class DataConsistencyInKspace_I(nn.Module):
    """ Create data consistency operator

    Warning: note that FFT2 (by the default of torch.fft) is applied to the last 2 axes of the input.
    This method detects if the input tensor is 4-dim (2D data) or 5-dim (3D data)
    and applies FFT2 to the (nx, ny) axis.

    """

    def __init__(self, noise_lvl=None):
        super(DataConsistencyInKspace_I, self).__init__()
        self.noise_lvl = noise_lvl

    def forward(self, *input, **kwargs):
        return self.perform(*input)

    def perform(self, x, k0, mask):
        """
        x    - input in image domain, of shape (n, 2, nx, ny)
        k0   - initially sampled elements in k-space
        mask - corresponding nonzero location
        """

        if x.dim() == 4: # input is 2D
            x = x.permute(0, 2, 3, 1)
        else:
            raise ValueError("error in data consistency layer!")

        k = fft2(x)
        out = data_consistency(k, k0, mask.repeat(1, 1, 1, 2), self.noise_lvl)
        x_res = ifft2(out)

        if x.dim() == 4:
            x_res = x_res.permute(0, 3, 1, 2)
        else:
            raise ValueError("Iuput dimension is wrong, it has to be a 2D input!")

        return x_res, out


class DataConsistencyInKspace_K(nn.Module):
    """ Create data consistency operator

    Warning: note that FFT2 (by the default of torch.fft) is applied to the last 2 axes of the input.
    This method detects if the input tensor is 4-dim (2D data) or 5-dim (3D data)
    and applies FFT2 to the (nx, ny) axis.

    """

    def __init__(self, noise_lvl=None):
        super(DataConsistencyInKspace_K, self).__init__()
        self.noise_lvl = noise_lvl

    def forward(self, *input, **kwargs):
        return self.perform(*input)

    def perform(self, k, k0, mask):
        """
        k    - input in frequency domain, of shape (n, 2, nx, ny)
        k0   - initially sampled elements in k-space
        mask - corresponding nonzero location
        """

        if k.dim() == 4:  # input is 2D
            k = k.permute(0, 2, 3, 1)
        else:
            raise ValueError("error in data consistency layer!")

        out = data_consistency(k, k0, mask.repeat(1, 1, 1, 2), self.noise_lvl)
        x_res = ifft2(out)

        if k.dim() == 4:
            x_res = x_res.permute(0, 3, 1, 2)
        else:
            raise ValueError("Iuput dimension is wrong, it has to be a 2D input!")

        return x_res, out

# Basic functions / transforms
def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)
    return torch.from_numpy(data)


def fft2(data):
    """
    Apply centered 2 dimensional Fast Fourier Transform.
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
    Returns:
        torch.Tensor: The FFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = torch.fft(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    return data


def ifft2(data):
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
    Returns:
        torch.Tensor: The IFFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = torch.ifft(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    return data


def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)


def complex_abs(data):
    """
    Compute the absolute value of a complex valued input tensor.
    Args:
        data (torch.Tensor): A complex valued tensor, where the size of the final dimension
            should be 2.
    Returns:
        torch.Tensor: Absolute value of data
    """
    assert data.size(-1) == 2
    return (data ** 2).sum(dim=-1).sqrt()


def complex_abs_eval(data):
    assert data.size(1) == 2
    return (data[:, 0:1, :, :] ** 2 + data[:, 1:2, :, :] ** 2).sqrt()


def to_spectral_img(data):
    """
    Compute the spectral images of a kspace data
    with keeping each column for creation of one spectral image
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
    Returns:
        torch.Tensor: The IFFT of the input.
    """
    assert data.size(-1) == 2

    spectral_vol = torch.zeros([data.size(-2), data.size(-2), data.size(-2)])

    for i in range(data.size(-2)):
        kspc1 = torch.zeros(data.size())
        kspc1[:, i, :] = data[:, i, :]
        img1 = ifft2(kspc1)
        img1_abs = complex_abs(img1)

        spectral_vol[i, :, :] = img1_abs

    return spectral_vol