import torch
import numpy as np

import torchvision.transforms as T
from kornia.color import rgb_to_hsv, hsv_to_rgb, yuv_to_rgb, rgb_to_yuv


class PCT():
    '''
    Pixel-Wise Color Transform
    applies specified PCT function to image given a parameter map

    transform_type : str
        PCT function name
    dim : int
        dimension of input vector (usually 3)
    affine : bool
        tranform has a translational component
    color_space : str
        transforms input to 'HSV' or 'YUV' to apply transform, does nothing for RGB 
    mean : list
        input normalization mean value
    std : list
        input normalization standard deviation value
    unnorm : bool
        before applying the transformation the input normalization is reversed
    clamp: bool
        clamp output to [0, 1] (before applying input normalization again)
    '''

    def __init__(self, transform_type, dim, affine, 
                    color_space='RGB', mean = [.485, .456, .406], std = [.229, .224, .225], 
                    unnorm=False, clamp=True):
        
        # Color Space
        self.color_trf_in = lambda x: x
        self.color_trf_out = lambda x: x
        if color_space == 'HSV':
            self.color_trf_in = rgb_to_hsv()
            self.color_trf_out = hsv_to_rgb()
        elif color_space == 'YUV':
            self.color_trf_in = rgb_to_yuv()
            self.color_trf_out = yuv_to_rgb()
        
        # Normalization
        self.norm = T.Normalize(mean=mean, std=std)
        self.unnorm = unnorm
        if color_space in ['HSV', 'YUV'] or unnorm:
            self.unnorm = torch.transforms.Normalize(mean=-mean/std, std=1/std)

        self.clamp = clamp

        # Transform Functions
        if transform_type == 'identity':
            self.transform = lambda input, param: param
            self.out_dim = 3
        elif transform_type == 'mul':
            self.transform = lambda input, param: input * param
            self.out_dim = 3
        elif transform_type == 'add':
            self.transform = lambda input, param: input + param
            self.out_dim = 3
        elif 'linear' in transform_type:
            type = transform_type.split('_')[-1]
            self.transform = Linear_PCT(dim, affine, type)
            self.out_dim = self.transform.out_dim
        elif transform_type == 'polynomial':
            self.transform = Linear_PCT(dim, affine, 'linear', 'polynomial')
            self.out_dim = 27
        elif transform_type == 'quadratic':
            self.transform == Polynomial_PCT(dim, 2)
            self.out_dim = 6
        elif transform_type == 'cubic':
            self.transform == Polynomial_PCT(dim, 3)
            self.out_dim = 9
        else:
            self.out_dim = 0
            print('Error: Invalid transform type')

    def __call__(self, input, param):
        
        if self.unnorm:
            input = self.unnorm(input)
        input = self.color_trf_in(input)

        output = self.transform(input, param)
        
        output = self.color_trf_out(output)
        if self.clamp:
            output = torch.clamp(output, 0, 1)
        output = self.norm(output)
        
        return output

    def get_out_dim(self):
        return self.out_dim


class Linear_PCT():

    def __init__(self, dim, affine, type='linear', projection=None):
        
        self.dim = dim
        self.affine = affine
        self.type = type
        self.projection = projection

        if type=='linear':
            self.out_dim = 9
        elif type=='sym':
            self.out_dim = 6
        if affine:
            self.out_dim += 3

    def __call__(self, input, param):
        
        N, C_in, H, W = input.shape
        out = torch.zeros_like(input)

        L0 = 3
        L = self.dim
        for n in range(N):
            x = input[n].movedim(0, -1).view(-1, C_in).unsqueeze(2)                                         # (HW, C_in, 1)
            
            if self.projection == 'polynomial':
                xr, xb, xg = x[:,:1], x[:,0:1], x[:,2:]
                x = torch.cat([xr, xb, xg, xr*xg, xr*xb, xg*xb, xr**2, xg**2, xb**2], axis=1)
                L0 = 9
            elif self.projection == 'sine':
                x = torch.cat([torch.sin(2**(i//2) * x * np.pi/2 * (i % 2) ) for i in range(9)], axis=1)    # (H*W, 3*L, 1)
                L0 = 9

            # Linear Matrix Multiplication
            if self.type == 'sym':
                L0 = 2
                M = param[n, 0:L0*L].movedim(0, -1).view(-1, L0*L)  
                M = torch.stack( [  torch.stack([M[:,0], M[:,3], M[:,5]], dim=1), 
                                    torch.stack([M[:,3], M[:,1], M[:,4]], dim=1),
                                    torch.stack([M[:,5], M[:,4], M[:,2]], dim=1)], dim=2)
            else:
                M = param[n, 0:L0*L].movedim(0, -1).view(-1, L, L0)                                         # (HW, L, L0)
            
            y = torch.matmul(M, x)    
            if self.affine:
                b = param[n,L0*L:(L0+1)*L].movedim(0, -1).view(-1, L).unsqueeze(2)                          # (HW, L, L0)
                y = y + b
            out[n] = y.view(H, W, C_in).movedim(-1, 0)                                                      # (H, W, 3)

        return out


class Polynomial_PCT():

    def __init__(self, dim, deg):
        self.dim = dim
        self.deg = deg

    def __call__(self, input, param):

        N, C_in, H, W = input.shape
        out = torch.zeros_like(input)

        param = param.view(N, self.C_in, self.dim, H, W)
        for l in range(self.deg+1):
            out += param[:, l] * torch.pow(input, l)
        
        return out