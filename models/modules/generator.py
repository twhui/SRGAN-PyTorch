import torch
import torch.nn as nn
from . import block as B

"""
The Generator described in the paper "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network".
"""
class SRResNetGenerator(nn.Module):
    def __init__(self, input_ngc, output_ngc, ngf, ngb=16, n_upscale=2, norm_type='batch', act_type='prelu'):
        super(SRResNetGenerator, self).__init__()

        first_conv = B.Conv2dBlock(input_ngc, ngf, kernel_size=9, norm_type=None, act_type=act_type)
        resnet_blocks = [B.ResNetBlock(ngf, ngf, ngf, norm_type=norm_type, act_type=act_type) for _ in range(ngb)]
        before_up_conv = B.Conv2dBlock(ngf, ngf, kernel_size=3, norm_type=norm_type, act_type=None)

        self.features_LR = nn.Sequential(first_conv, B.ShortcutBlock(nn.Sequential(*resnet_blocks, before_up_conv)))
        self.features_HR = nn.Sequential(*[B.SubPixelConvBlock(ngf, ngf, upscale_factor=2, kernel_size=3, norm_type=None, act_type=act_type) for _ in range(n_upscale)])
        self.reducer = B.Conv2dBlock(ngf, output_ngc, kernel_size=9, norm_type=None, act_type=None)

    def forward(self, input):
        features_LR = self.features_LR(input)
        features_HR = self.features_HR(features_LR)
        output = self.reducer(features_HR)
        return output
