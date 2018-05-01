import torch
import torch.nn as nn
from . import block as B


"""
The Generator described in the paper "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network".
"""
class SRResNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, nf, nb, n_upscale=2, norm_type='batch', act_type='prelu'):
        super(SRResNetGenerator, self).__init__()

        conv1 = B.Conv2dBlock(input_nc, nf, kernel_size=9, norm_type=None, act_type=act_type)
        resnet_blocks = [B.ResNetBlock(nf, nf, nf, norm_type=norm_type, act_type=act_type) for _ in range(nb)]
        convN = B.Conv2dBlock(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None)

        self.net_LR = nn.Sequential(conv1, B.ShortcutBlock(nn.Sequential(*resnet_blocks, convN)))
        self.net_HR = nn.Sequential(*[B.SubPixelConvBlock(nf, nf, upscale_factor=2, kernel_size=3, norm_type=None, act_type=act_type) for _ in range(n_upscale)])
        self.reducer = B.Conv2dBlock(nf, output_nc, kernel_size=9, norm_type=None, act_type=None)

    def forward(self, input):
        x = self.net_LR(input)
        x = self.net_HR(x)
        output = self.reducer(x)
        return output
