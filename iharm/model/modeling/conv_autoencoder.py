import torch
from torch import nn as nn
import torch.nn.functional as F


from iharm.model.modeling.basic_blocks import ConvBlock
from iharm.model.ops import MaskedChannelAttention, FeaturesConnector


class ConvEncoder(nn.Module):
    def __init__(
        self,
        depth, ch,
        norm_layer, batchnorm_from, max_channels,
        backbone_from, backbone_channels=None, backbone_mode='', pad_mode='',
    ):
        super(ConvEncoder, self).__init__()
        self.depth = depth
        self.backbone_from = backbone_from
        backbone_channels = [] if backbone_channels is None else backbone_channels[::-1]

        in_channels = 4
        out_channels = ch

        self.block0 = ConvBlock(in_channels, out_channels, norm_layer=norm_layer if batchnorm_from == 0 else None)
        self.block1 = ConvBlock(out_channels, out_channels, norm_layer=norm_layer if 0 <= batchnorm_from <= 1 else None)
        self.blocks_channels = [out_channels, out_channels]

        self.blocks_connected = nn.ModuleDict()
        self.connectors = nn.ModuleDict()
        for block_i in range(2, depth):
            if block_i % 2:
                in_channels = out_channels
            else:
                in_channels, out_channels = out_channels, min(2 * out_channels, max_channels)

            if 0 <= backbone_from <= block_i and len(backbone_channels):
                stage_channels = backbone_channels.pop()
                connector = FeaturesConnector(backbone_mode, in_channels, stage_channels, in_channels)
                self.connectors[f'connector{block_i}'] = connector
                in_channels = connector.output_channels

            if pad_mode == 'final':
                self.blocks_connected[f'block{block_i}'] = ConvBlock(
                    in_channels, out_channels,
                    norm_layer=norm_layer if 0 <= batchnorm_from <= block_i else None,
                    padding=1
                )
            else:
                self.blocks_connected[f'block{block_i}'] = ConvBlock(
                    in_channels, out_channels,
                    norm_layer=norm_layer if 0 <= batchnorm_from <= block_i else None,
                    padding=int(block_i < depth - 1)
                )       
            self.blocks_channels += [out_channels]

    def forward(self, x, backbone_features):
        backbone_features = [] if backbone_features is None else backbone_features[::-1]

        outputs = [self.block0(x)]
        outputs += [self.block1(outputs[-1])]

        for block_i in range(2, self.depth):
            block = self.blocks_connected[f'block{block_i}']
            output = outputs[-1]
            connector_name = f'connector{block_i}'
            if connector_name in self.connectors:
                stage_features = backbone_features.pop()
                connector = self.connectors[connector_name]
                output = connector(output, stage_features)
            outputs += [block(output)]

        return outputs[::-1]


class DeconvDecoder(nn.Module):
    def __init__(self, depth, encoder_blocks_channels, norm_layer, attend_from=-1, image_fusion=False):
        super(DeconvDecoder, self).__init__()
        self.image_fusion = image_fusion
        self.deconv_blocks = nn.ModuleList()

        in_channels = encoder_blocks_channels.pop()
        out_channels = in_channels
        for d in range(depth):
            out_channels = encoder_blocks_channels.pop() if len(encoder_blocks_channels) else in_channels // 2
            self.deconv_blocks.append(SEDeconvBlock(
                in_channels, out_channels,
                norm_layer=norm_layer,
                padding=0 if d == 0 else 1,
                with_se=0 <= attend_from <= d
            ))
            in_channels = out_channels

        if self.image_fusion:
            self.conv_attention = nn.Conv2d(out_channels, 1, kernel_size=1)
        self.to_rgb = nn.Conv2d(out_channels, 3, kernel_size=1)

    def forward(self, encoder_outputs, image, mask=None):
        output = encoder_outputs[0]
        for block, skip_output in zip(self.deconv_blocks[:-1], encoder_outputs[1:]):
            output = block(output, mask)
            output = output + skip_output
        output = self.deconv_blocks[-1](output, mask)

        attention_map = None
        if self.image_fusion:
            attention_map = torch.sigmoid(3.0 * self.conv_attention(output))
            rgb = self.to_rgb(output)
            output = attention_map * image + (1.0 - attention_map) * rgb
        else:
            rgb = self.to_rgb(output)
            output = mask * rgb + (1 - mask) * image

        return output, attention_map, rgb


class DeconvDecoderMhead(nn.Module):
    def __init__(self, depth, encoder_blocks_channels, norm_layer, attend_from=-1, image_fusion=False):
        super(DeconvDecoderMhead, self).__init__()
        self.image_fusion = image_fusion
        self.deconv_blocks = nn.ModuleList()

        in_channels = encoder_blocks_channels.pop()
        out_channels = in_channels
        for d in range(depth):
            out_channels = encoder_blocks_channels.pop() if len(encoder_blocks_channels) else in_channels // 2
            self.deconv_blocks.append(SEDeconvBlock(
                in_channels, out_channels,
                norm_layer=norm_layer,
                padding=0 if d == 0 else 1,
                with_se=0 <= attend_from <= d
            ))
            in_channels = out_channels

        if self.image_fusion:
            self.conv_attention = nn.Conv2d(out_channels, 1, kernel_size=1)
        self.to_rgb = nn.Conv2d(out_channels, 3, kernel_size=1)
        self.to_h = nn.Conv2d(out_channels, 1, kernel_size=1)
        self.to_s = nn.Conv2d(out_channels, 3, kernel_size=1)
        self.to_v = nn.Conv2d(out_channels, 1, kernel_size=1)

    def forward(self, encoder_outputs, image, hsv, mask=None):
        output = encoder_outputs[0]
        for block, skip_output in zip(self.deconv_blocks[:-1], encoder_outputs[1:]):
            output = block(output, mask)
            output = output + skip_output
        output = self.deconv_blocks[-1](output, mask)

        if self.image_fusion:
            attention_map = torch.sigmoid(3.0 * self.conv_attention(output))
            output_rgb = attention_map * image + (1.0 - attention_map) * self.to_rgb(output)
            output_h = attention_map * hsv.get('h') + (1.0 - attention_map) * self.to_h(output)
            output_s = attention_map * hsv.get('s') + (1.0 - attention_map) * self.to_s(output)
            output_v = attention_map * hsv.get('v') + (1.0 - attention_map) * self.to_v(output)
        else:
            output_rgb = self.to_rgb(output)
            output_h = self.to_h(output)
            output_s = self.to_s(output)
            output_v = self.to_v(output)

        return output_rgb, output_h, output_s, output_v

class DeconvDecoderUpsample(nn.Module):
    def __init__(self, depth, encoder_blocks_channels, norm_layer, attend_from=-1, image_fusion=False):
        super(DeconvDecoderUpsample, self).__init__()
        self.image_fusion = image_fusion
        self.deconv_blocks = nn.ModuleList()

        in_channels = encoder_blocks_channels.pop()
        out_channels = in_channels
        for d in range(depth):
            out_channels = encoder_blocks_channels.pop() if len(encoder_blocks_channels) else in_channels // 2
            self.deconv_blocks.append(SEDeconvBlock(
                in_channels, out_channels,
                norm_layer=norm_layer,
                padding=0 if d == 0 else 1,
                with_se=0 <= attend_from <= d
            ))
            in_channels = out_channels

        if self.image_fusion:
            self.conv_attention = nn.Conv2d(out_channels, 1, kernel_size=1)

    def forward(self, encoder_outputs, image, mask=None):
        output = encoder_outputs[0]
        for block, skip_output in zip(self.deconv_blocks[:-1], encoder_outputs[1:]):
            output = block(output, mask)
            output = output + skip_output
        output = self.deconv_blocks[-1](output, mask)
        attention_map = torch.sigmoid(3.0 * self.conv_attention(output))
  
        return output, attention_map



class DeconvDecoderUpsamplePconv(nn.Module):
    def __init__(self, depth, encoder_blocks_channels, norm_layer, attend_from=-1, image_fusion=False):
        super(DeconvDecoderUpsamplePconv, self).__init__()
        self.image_fusion = image_fusion
        self.deconv_blocks = nn.ModuleList()

        in_channels = encoder_blocks_channels.pop()
        out_channels = in_channels
        for d in range(depth):
            out_channels = encoder_blocks_channels.pop() if len(encoder_blocks_channels) else in_channels // 2
            self.deconv_blocks.append(SEDeconvBlock(
                in_channels, out_channels,
                norm_layer=norm_layer,
                padding=0 if d == 0 else 1,
                with_se=0 <= attend_from <= d
            ))
            in_channels = out_channels

        self.pconv1 = PartialConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pconv2 = PartialConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pconv3 = PartialConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool_mask = nn.MaxPool2d(kernel_size=2)
        self.activation = nn.ELU()
        self.norm1 = nn.BatchNorm2d(64)
        self.norm2 = nn.BatchNorm2d(64)
        self.norm3 = nn.BatchNorm2d(64)

        if self.image_fusion:
            self.conv_attention = nn.Conv2d(out_channels, 1, kernel_size=1)

    def forward(self, encoder_outputs, image, mask=None):
        output = encoder_outputs[0]
        cnt = 0
        for block, skip_output in zip(self.deconv_blocks[:-1], encoder_outputs[1:]):
            output = block(output, mask)
            output = output + skip_output
            #print(cnt, output.shape)
            
            if cnt == 5:
                #print(skip_output.shape)
                pc_enc = skip_output
                pc_mask = self.pool_mask(mask)

                pc_enc, pc_mask = self.pconv1(pc_enc, pc_mask)
                pc_enc = self.norm1(pc_enc)
                pc_enc = self.activation(pc_enc)

                pc_enc, pc_mask = self.pconv2(pc_enc, pc_mask)
                pc_enc = self.norm2(pc_enc)
                pc_enc = self.activation(pc_enc)

                pc_enc, pc_mask = self.pconv3(pc_enc, pc_mask)
                pc_enc = self.norm3(pc_enc)
                pc_enc = self.activation(pc_enc)
                #print('pc_enc', pc_enc.shape)
                output = output + pc_enc
            cnt += 1
        output = self.deconv_blocks[-1](output, mask)
        attention_map = torch.sigmoid(3.0 * self.conv_attention(output))
  
        return output, attention_map



class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False

        self.return_mask = True

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0],
                                                 self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * \
                             self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2],
                                          input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride,
                                            padding=self.padding, dilation=self.dilation, groups=1)

                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)


        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output

class DeconvDecoderUpsampleTrans(nn.Module):
    def __init__(self, depth, encoder_blocks_channels, norm_layer, attend_from=-1, image_fusion=False):
        super(DeconvDecoderUpsampleTrans, self).__init__()
        self.image_fusion = image_fusion
        self.deconv_blocks = nn.ModuleList()

        in_channels = encoder_blocks_channels.pop()
        out_channels = in_channels
        for d in range(depth):
            out_channels = encoder_blocks_channels.pop() if len(encoder_blocks_channels) else in_channels // 2
            self.deconv_blocks.append(SEDeconvBlock(
                in_channels, out_channels,
                norm_layer=norm_layer,
                #padding=0 if d == 0 else 1,
                padding=1,
                with_se=0 <= attend_from <= d
            ))
            in_channels = out_channels

        self.decode_trans = SEDeconvBlock(
                768, 128,
                norm_layer=norm_layer,
                padding=1,
                with_se=False
            )

        if self.image_fusion:
            self.conv_attention = nn.Conv2d(out_channels, 1, kernel_size=1)

    def forward(self, encoder_outputs, image, mask=None, trans_embed=None):
        output = encoder_outputs[0]
        #print('trans_emb', trans_embed.shape)
        trans_up = self.decode_trans(trans_embed)
        #print('trans_up', trans_up.shape)
        output = output + trans_up
        for block, skip_output in zip(self.deconv_blocks[:-1], encoder_outputs[1:]):
            output = block(output, mask)
            output = output + skip_output
        output = self.deconv_blocks[-1](output, mask)
        attention_map = torch.sigmoid(3.0 * self.conv_attention(output))
  
        return output, attention_map


class SEDeconvBlock(nn.Module):
    def __init__(
        self,
        in_channels, out_channels,
        kernel_size=4, stride=2, padding=1,
        norm_layer=nn.BatchNorm2d, activation=nn.ELU,
        with_se=False
    ):
        super(SEDeconvBlock, self).__init__()
        self.with_se = with_se
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            norm_layer(out_channels) if norm_layer is not None else nn.Identity(),
            activation(),
        )
        if self.with_se:
            self.se = MaskedChannelAttention(out_channels)

    def forward(self, x, mask=None):
        out = self.block(x)
        if self.with_se:
            out = self.se(out, mask)
        return out
