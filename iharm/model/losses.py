import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import vgg16
import kornia as K
import numpy as np

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from iharm.utils import misc
from iharm.model.color_transfer import pre_process, transfer_chrom, transfer_lum

def inv_norm(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    if tensor.get_device() > -1:
        mean = mean.to(tensor.get_device())
        std = std.to(tensor.get_device())
    if len(tensor.size()) == 3 and tensor.size(0) == 3:
        out = tensor * std[:,None,None] + mean[:,None,None]
    elif len(tensor.size()) == 4 and tensor.size(1) == 3:
        out = tensor * std[None,:,None,None] + mean[None,:,None,None]
    else:
        out = None
        print(f"Error! Can't normalize shape {tensor.size()}")

    return out

class Loss(nn.Module):
    def __init__(self, pred_outputs, gt_outputs):
        super().__init__()
        self.pred_outputs = pred_outputs
        self.gt_outputs = gt_outputs

class L1(Loss):
    def __init__(self, pred_name='images', gt_image_name='target_images'):
        super(L1, self).__init__(pred_outputs=(pred_name,), gt_outputs=(gt_image_name,))
        self.loss = torch.nn.L1Loss()

    def forward(self, pred, label):
        if torch.is_tensor(pred):
            if pred.device != label.device:
                label = label.to(pred.device)
            loss = self.loss(pred, label)
        else:
            loss = []
            for pred_img, label_img in zip(pred, label):
                if pred_img.device != label_img.device:
                    label_img = label_img.to(pred_img.device)
                loss.append(self.loss(pred_img, label_img))
            loss = torch.stack(loss)

        return loss            

class Huber(Loss):
    def __init__(self, pred_name='images', gt_image_name='target_images'):
        super(Huber, self).__init__(pred_outputs=(pred_name,), gt_outputs=(gt_image_name,))
        self.loss = torch.nn.HuberLoss()

    def forward(self, pred, label):
        if torch.is_tensor(pred):
            if pred.device != label.device:
                label = label.to(pred.device)
            loss = self.loss(pred, label)
        else:
            loss = []
            for pred_img, label_img in zip(pred, label):
                if pred_img.device != label_img.device:
                    label_img = label_img.to(pred_img.device)
                loss.append(self.loss(pred_img, label_img))
            loss = torch.stack(loss)
        return loss

class MSE(Loss):
    def __init__(self, pred_name='images', gt_image_name='target_images'):
        super(MSE, self).__init__(pred_outputs=(pred_name,), gt_outputs=(gt_image_name,))

    def forward(self, pred, label):
        label = label.view(pred.size())
        loss = torch.mean((pred - label) ** 2, dim=misc.get_dims_with_exclusion(label.dim(), 0))
        return loss


class MaskWeightedMSE(Loss):
    def __init__(self, min_area=1000.0, pred_name='images',
                 gt_image_name='target_images', gt_mask_name='masks'):
        super(MaskWeightedMSE, self).__init__(pred_outputs=(pred_name, ),
                                              gt_outputs=(gt_image_name, gt_mask_name))
        self.min_area = min_area

    def forward(self, pred, label, mask):

        if torch.is_tensor(pred):
            label = label.view(pred.size())
            reduce_dims = misc.get_dims_with_exclusion(label.dim(), 0)
            if pred.device != label.device:
                label = label.to(pred.device)
                mask = mask.to(pred.device)
            
            loss = (pred - label) ** 2
            delimeter = pred.size(1) * torch.clamp_min(torch.sum(mask[:, 0:1, :, :], dim=reduce_dims), self.min_area)
            loss = torch.sum(loss, dim=reduce_dims) / delimeter
        else:
            loss = []
            for pred_img, label_img, mask_img in zip(pred, label, mask):
                if pred_img.device != label_img.device:
                    label_img = label_img.to(pred_img.device)
                    mask_img = mask_img.to(pred_img.device)
                mse = torch.sum((pred_img - label_img) ** 2)
                delimeter = pred_img.size(0) * torch.clamp_min(torch.sum(mask_img[0:1, :, :]), self.min_area)
                loss.append(mse / delimeter)
            loss = torch.stack(loss)

        return loss

class ParamSmoothness(Loss):
    def __init__(self, pred_name='params'):
        super(ParamSmoothness, self).__init__(pred_outputs=(pred_name,), gt_outputs=())

    def forward(self, params):
        loss = torch.mean(K.filters.sobel(params))
        return loss

class SCS_CR_loss(Loss):

    def __init__(self, pred_name='images', gt_image_name='target_images', gt_mask_name='masks', gt_comp_name='images', k=5, writer=None) -> None:
        super(SCS_CR_loss, self).__init__(pred_outputs=(pred_name, ),
                                              gt_outputs=(gt_image_name, gt_mask_name, gt_comp_name))
        
        self.D = nn.L1Loss()
        self.k = k
        self.writer = writer
        VGG16_model = vgg16(pretrained=True)
        self.SR_extractor = VGG16_model.features[:23]

    def forward(self, pred, gt_image, mask, composition, return_neg_samp=False):
        '''
        composition (N, 3, H, W):           composition image (input image to the network)
        pred        (N, 3, H, W):           predicted image (output of the network)
        gt_image    (N, 3, H, W):           ground truth image
        mask        (N, 1, H, W):           image mask         
        '''
        
        if self.k > pred.size(0)-1:
            self.k = pred.size(0)-1

        # create negative samples
        with torch.no_grad():
            neg_samp = self.create_negative_samples(gt_image.detach(), composition.detach(), mask.detach())

        if return_neg_samp:
            return neg_samp

        # calculate foreground and background style representations    
        self.SR_extractor = self.SR_extractor.to(pred.get_device())
        self.SR_extractor.eval()

        f = self.SR_extractor(pred * mask)                                            # (N, 512, 32, 32)
        f_plus = self.SR_extractor(gt_image * mask)                                   # (N, 512, 32, 32)
        f_minus = [self.SR_extractor(neg_samp[:,k] * mask) for k in range(self.k)]    # (K, N, 512, 32, 32)
        b_plus = self.SR_extractor(gt_image * (1-mask))                               # (N, 512, 32, 32)
        
        # self-style contrastistive regularization (SS-CR)
        l_ss_cr = self.D(f, f_plus) / ( self.D(f, f_plus) + torch.sum(torch.tensor([self.D(f, f_minus_k) for f_minus_k in f_minus])) + 1e-8 )

        # calculate Gram matrices based on style representations
        c = self.Gram(f, b_plus)                                                    # (N, 512, 512)
        c_plus = self.Gram(f_plus, b_plus)                                          # (N, 512, 512)
        c_minus = [self.Gram(f_minus_k, b_plus) for f_minus_k in f_minus]           # (K, N, 512, 512)

        # Consistent style contrastive regularization (CS-CR)
        l_cs_cr = self.D(c, c_plus) / ( self.D(c, c_plus) + torch.sum(torch.tensor([self.D(c, c_minus_k) for c_minus_k in c_minus])) + 1e-8 )

        return l_ss_cr + l_cs_cr

    def create_negative_samples(self, gt_images, composition, mask, gamma=2.2):
        '''
        samples K-1 negative samples for each composited image based on the other images in the batch

        gt_images   (N, 3, H, W):           ground truth images
        composition (N, 3, H, W):           composition image (input image to the network)
        mask        (N, 1, H, W):           image mask   

        neg_samp    (N, K, 3, H, W):        K negative samples 
        '''

        # Pre-processing
        N, C, H, W, = gt_images.size()        
        img = inv_norm(gt_images)
        img = torch.pow(img, gamma)
        img = K.color.rgb_to_lab(img)

        for n in range(N):
            img[n, 0] = pre_process(img[n, 0].clone())

        # Change color statistics for each composition
        neg_samp = torch.ones((N, self.k-1, C, H, W)).to(gt_images.get_device())
        ref_img = np.zeros((N, self.k-1), np.int16)
        for n in range(N):
            for j, n2 in enumerate(np.random.choice([i for i in range(N) if i!=n], self.k-1, replace=False)):

                ## No mask consideration for color transformation                
                c_img = transfer_chrom(img[n], img[n2])                                                 # (3, H, W)                           
                t_img = transfer_lum(img[n, 0], img[n2, 0])
                
                neg_img = torch.stack([t_img, c_img[1], c_img[2]])                # (3, H, W)
      
                neg_samp[n, j] = neg_img
                ref_img[n, j] = n2

        # Post-processing
        neg_samp = K.color.lab_to_rgb(neg_samp)
        neg_samp = torch.pow(neg_samp, 1/gamma)
        normalize = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        neg_samp = normalize(neg_samp)
        for n in range(N):
            neg_samp[n] = (1-mask[n].unsqueeze(0)) * gt_images[n] + mask[n].unsqueeze(0) * neg_samp[n]
        neg_samp = torch.cat([composition.unsqueeze(1), neg_samp], dim=1)

        return neg_samp

    def Gram(self, mat1, mat2):
        '''
        caculates the Gram matrix

        mat1 (N, 512, 32, 32):              feature map
        mat2 (N, 512, 32, 32):              feature map

        out (N, 512, 512):                  Gram matrix of both feature maps
        '''

        out = []
        for f1, f2 in zip(mat1, mat2):
            out.append( torch.matmul(f1.view(512, -1).T, f2.view(512, -1)) )

        return torch.stack(out)