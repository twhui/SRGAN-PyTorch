import collections
import torch
import torch.nn as nn
from torch.autograd import Variable


"""
Helper to select activation layer with string
"""
def act(act_type, **kwargs):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(True, **kwargs)
    elif act_type == 'leaky_relu':
        layer = nn.LeakyReLU(0.2, True, **kwargs)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=1, init=0.25, **kwargs)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act_type)
    return layer

"""
Helper to select normalization layer with string
"""
def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return layer

"""
Helper to select padding layer with string
It also infers suitable pad size with kernel_size if exact_pad is not given.
exact_pad overrides kernel_size.
"""
def pad(pad_type, kernel_size=None, exact_pad_size=None):
    pad_type = pad_type.lower()
    if kernel_size:
        pad_size = (kernel_size - 1) // 2
    if exact_pad_size:
        pad_size = exact_pad_size
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(pad_size)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(pad_size)
    elif pad_type == 'zero':
        layer = nn.ZeroPad2d(pad_size)
    else:
        raise NotImplementedError('padding layer [%s] is not implemented' % pad_type)
    return layer


def dropout(p=0.2):
    return nn.Dropout(p)

"""
Dummy function for simplifying syntax
"""
def identity(input):
    return input


"""
Concat the output of a submodule to its input
"""
class ConcatBlock(nn.Module):
    def __init__(self, submodule):
        super(ConcatBlock, self).__init__()
        self.submodule = submodule

    def forward(self, input):
        output = torch.cat((input, self.submodule(input)), dim=1)
        return output

    def __repr__(self):
        tmpstr =  'Identity .. \n|'
        modstr = self.submodule.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr

"""
Elementwise sum the input of a submodule to its output
"""
class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.submodule = submodule

    def forward(self, input):
        output = input + self.submodule(input)
        return output

    def __repr__(self):
        tmpstr =  'Identity + \n|'
        modstr = self.submodule.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


"""
Elementwise sum the 2 inputs
"""
class SumBlock(nn.Module):
    def __init__(self):
        super(SumBlock, self).__init__()

    def forward(self, input1, input2):
        output = input1 + input2
        return output

    def __repr__(self):
        tmpstr =  '+ \n|'
        return tmpstr

"""
Conv2d Layer with padding, normalization, activation, dropout
"""
class Conv2dBlock(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
                 pad_type='zero', norm_type=None, act_type='relu', use_dropout=False):
        super(Conv2dBlock, self).__init__()
        self.P = pad(pad_type, kernel_size) if pad_type else identity
        self.C = nn.Conv2d(input_nc, output_nc, kernel_size=kernel_size, stride=stride, dilation=dilation, bias=bias)
        self.N = norm(norm_type, output_nc) if norm_type else identity
        self.A = act(act_type) if act_type else identity
        self.D = dropout() if use_dropout else identity
        self.weight_init()

    def forward(self, input):
        output = self.P(input)
        output = self.C(output)
        output = self.N(output)
        output = self.A(output)
        output = self.D(output)
        return output

    def weight_init(self, mode='fan_in'):
        if isinstance(self.A, nn.LeakyReLU):
            a = 0.2 # LeakyReLU default negative slope
        elif isinstance(self.A, nn.PReLU):
            a = 0.25 # PReLU default initial negative slope
        elif isinstance(self.A, nn.ReLU):
            a = 0.0 # ReLU has zero negative slope
        else:
            a = 1.0
        nn.init.kaiming_normal(self.C.weight, a=a, mode='fan_in')
        if self.C.bias is not None:
            self.C.bias.data.zero_()
        if isinstance(self.N, nn.BatchNorm2d):
            self.N.weight.data.fill_(1)
            self.N.bias.data.zero_()

def ResNetBlock(input_nc, mid_nc, output_nc, kernel_size=3, stride=1, bias=True,
                pad_type='zero', norm_type='batch', act_type='prelu', use_dropout=False):
    conv1 = Conv2dBlock(input_nc, mid_nc,    kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type, use_dropout=use_dropout)
    conv2 = Conv2dBlock(mid_nc,   output_nc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=None,     use_dropout=False)
    residual_features = nn.Sequential(conv1, conv2)
    return ShortcutBlock(residual_features)


class SubPixelConvBlock(nn.Module):
    def __init__(self, input_nc, output_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True,
                 pad_type='zero', norm_type=None, act_type='prelu', use_dropout=False):
        super(SubPixelConvBlock, self).__init__()
        self.conv_block = Conv2dBlock(input_nc, output_nc * (upscale_factor ** 2), kernel_size, stride, bias=bias,
                                      pad_type=pad_type, norm_type=norm_type, act_type=act_type, use_dropout=use_dropout)
        self.PS = nn.PixelShuffle(upscale_factor)

    def forward(self, input):
        output = self.conv_block(input)
        output = self.PS(output)
        return output
