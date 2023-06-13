import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from iharm.model.base.ssam_model import SpatialSeparatedAttention
from iharm.model.modeling.conv_autoencoder import ConvEncoder, DeconvDecoderUpsample
from iharm.model.modeling.unet import UNetEncoder, UNetDecoderUpsample
from iharm.model.modeling.vit_base import ViT_Harmonizer
from iharm.model.pct_functions import PCT

class PCTNet(nn.Module):
    
    def __init__(
        self,
        backbone_type='idih', 
        input_normalization = {'mean': [.485, .456, .406], 'std': [.229, .224, .225]},
        dim=3, transform_type='linear', affine=True,
        clamp=True, color_space = 'RGB', use_attn = True,
        depth=4, norm_layer=nn.BatchNorm2d, batchnorm_from=0, attend_from=-1,
        image_fusion=False, ch=64, max_channels=512, attention_mid_k=2.0, 
        backbone_from=-1, backbone_channels=None, backbone_mode='',
    ):
        super(PCTNet, self).__init__()
              
        self.color_space = color_space
        self.use_attn = use_attn

        self.dim = dim
        self.transform_type = transform_type
        self.affine = affine
        self.PCT = PCT(transform_type, dim, affine, color_space, input_normalization['mean'], input_normalization['std'], clamp=clamp)
        self.out_dim = self.PCT.get_out_dim()

        input_dim = self.out_dim
        self.backbone_type = backbone_type
        if backbone_type == 'idih':
            self.encoder = ConvEncoder(
                depth, ch,
                norm_layer, batchnorm_from, max_channels,
                backbone_from, backbone_channels, backbone_mode
            )
            self.decoder = DeconvDecoderUpsample(depth, self.encoder.blocks_channels, norm_layer, attend_from, image_fusion)
            input_dim = 32
        elif backbone_type == 'ssam':
            print('depth', depth)
            self.encoder = UNetEncoder(
                depth, ch, 
                norm_layer, batchnorm_from, max_channels,
                backbone_from, backbone_channels, backbone_mode
            )
            self.decoder =  UNetDecoderUpsample(
                depth, self.encoder.block_channels,
                norm_layer,
                attention_layer=partial(SpatialSeparatedAttention, mid_k=attention_mid_k),
                attend_from=attend_from,
                image_fusion=image_fusion
            )
            input_dim=32
        elif backbone_type == 'ViT':
            self.encoder = ViT_Harmonizer(output_nc=input_dim)
            self.decoder = lambda intermediates, img, mask: (intermediates, mask)

        self.get_params = nn.Conv2d(input_dim, self.out_dim, kernel_size=1)


    def forward(self, image, image_fullres=None, mask=None, mask_fullres=None, backbone_features=None):

        self.device = image.get_device()

        # Low resolution branch
        x = torch.cat((image, mask), dim=1)
        
        intermediates = self.encoder(x, backbone_features)
        latent, attention_map = self.decoder(intermediates, image, mask) # (N, 32, 256, 256), (N, 1, 256, 256)
        params = self.get_params(latent)

        output_lowres = self.PCT(image, params)

        if self.use_attn:
            output_lowres = output_lowres * attention_map + image * (1-attention_map)
        else:
            output_lowres = output_lowres * mask + image * (1 - mask)

        outputs = dict()
        outputs['images'] = output_lowres
        outputs['params'] = params
        outputs['attention'] = attention_map
    

        # Full resolution branch
        if torch.is_tensor(image_fullres):
            fr_imgs = [image_fullres]
            fr_masks = [mask_fullres]
            idx = [[n for n in range(image_fullres.shape[0])]]
        else:
            fr_imgs = [img_fr.unsqueeze(0).to(image.get_device()) for img_fr in image_fullres]
            fr_masks = [mask_fr.unsqueeze(0).to(image.get_device()) for mask_fr in mask_fullres]
            idx = [[n] for n in range(len(image_fullres))]

        out_fr = []
        param_fr = []
        for id, fr_img, fr_mask in zip(idx, fr_imgs, fr_masks):
            H = fr_img.size(2)
            W = fr_img.size(3)
            params_fullres = F.interpolate(params[id], size=(H, W), mode='bicubic')
            output_fullres = self.PCT(fr_img, params_fullres)
            if self.use_attn:
                attention_map_fullres = F.interpolate(attention_map[id], size=(H, W), mode='bicubic')
                output_fullres = output_fullres * attention_map_fullres + fr_img * (1-attention_map_fullres)
            else:
                output_fullres = output_fullres * fr_mask + fr_img * (1 - fr_mask)
            out_fr.append(output_fullres.squeeze())
            param_fr.append(params_fullres.squeeze())
        
        if len(out_fr) == 1:
            out_fr = out_fr[0]
        outputs['images_fullres'] = out_fr
        if len(param_fr) == 1:
            param_fr = param_fr[0]
        outputs['params_fullres'] = param_fr
            
        return outputs 
