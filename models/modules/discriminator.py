import math
import torch.nn as nn
from . import block as B

"""
The Discriminator described in the paper "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network".
"""
class SingleLabelDiscriminator_96(nn.Module):
    def __init__(self, input_ndc, ndf=64, n_dense_feat=1024, norm_type='batch', act_type='leaky_relu'):
        super(SingleLabelDiscriminator_96, self).__init__()

        first_block = [B.Conv2dBlock(input_ndc, ndf, kernel_size=3, norm_type=None, act_type=act_type)]
        first_block += [B.Conv2dBlock(ndf, ndf, kernel_size=3, stride=2, norm_type=norm_type, act_type=act_type)]

        other_blocks = []
        for i in range(3):
            other_blocks += [B.Conv2dBlock(ndf * (2 ** i), ndf * (2 ** (i+1)), kernel_size=3, norm_type=norm_type, act_type=act_type)]
            other_blocks += [B.Conv2dBlock(ndf * (2 ** (i+1)), ndf * (2 ** (i+1)), kernel_size=3, stride=2, norm_type=norm_type, act_type=act_type)]
        self.features = nn.Sequential(*first_block, *other_blocks)

        dense_blocks = [B.Conv2dBlock(ndf * 8, n_dense_feat, kernel_size=6, pad_type=None, norm_type=None, act_type=act_type)]
        dense_blocks += [B.Conv2dBlock(n_dense_feat, 1, kernel_size=1, pad_type=None, norm_type=None, act_type=None)]
        self.reducer = nn.Sequential(*dense_blocks)

    def forward(self, input):
        output = self.features(input)
        output = self.reducer(output)
        return output
