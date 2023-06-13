from einops.layers.torch import Rearrange
from torch import nn
import math

class ViT_Harmonizer(nn.Module):
    def __init__(self, output_nc, ksize=4, tr_r_enc_head=2, tr_r_enc_layers=9, input_nc=3, dim_forward=2, tr_act='gelu'):
        super(ViT_Harmonizer, self).__init__()
        dim = 256
        self.patch_to_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = ksize, p2 = ksize),
                nn.Linear(ksize*ksize*(input_nc+1), dim)
            )
        self.transformer_enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(dim,nhead=tr_r_enc_head, dim_feedforward=dim*dim_forward, activation=tr_act), num_layers=tr_r_enc_layers)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(dim, output_nc, kernel_size=ksize, stride=ksize, padding=0),
            nn.Tanh()
        )

    def forward(self, inputs, backbone_features=None):
        patch_embedding = self.patch_to_embedding(inputs)
        content = self.transformer_enc(patch_embedding.permute(1, 0, 2))
        bs, L, C  = patch_embedding.size()
        harmonized = self.dec(content.permute(1,2,0).view(bs, C, int(math.sqrt(L)), int(math.sqrt(L))))
        return harmonized