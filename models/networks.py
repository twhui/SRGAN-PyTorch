import torch
import torch.nn as nn
import models.modules.generator as G
import models.modules.discriminator as D
from models.modules.vgg_feat import VGGFeat

# Pick the corresponding model for Generator and initialize it.
def define_G(opt):
    gpu_ids = opt.gpu_ids
    opt = opt.network
    which_model = opt.which_model_G

    if which_model == 'sr_resnet':
        netG = G.SRResNetGenerator(input_ngc=opt.input_ngc, output_ngc=opt.output_ngc, ngf=opt.ngf, ngb=opt.ngb, norm_type=opt.norm_type)
    elif which_model == 'sr_gan':
        netG = G.SRGANGenerator(input_ngc=opt.input_ngc, output_ngc=opt.output_ngc, ngf=opt.ngf, ngb=opt.ngb, norm_type=opt.norm_type)

    else:
        raise NotImplementedError('Generator model [%s] is not recognized' % which_model)

    netG = nn.DataParallel(netG, device_ids=gpu_ids)
    return netG


# Pick the corresponding pretrained model for Feature Extractor and load the weight
def define_F(opt):
    gpu_ids = opt.gpu_ids
    opt = opt.network
    which_model = opt.which_model_F

    # vgg_i_j indicate the feature map obtained by the j-th convolution (after activation)
    # before the i-th maxpooling layer within the VGG network
    if which_model == 'vgg16_2_2':
        netF = VGGFeat(n_layers=16, i_max_pool=1, include_max_pool=True) # i starts at 0
    elif which_model == 'vgg16_5_4':
        netF = VGGFeat(n_layers=16, i_max_pool=4, include_max_pool=False) # dont need last max_pool
    elif which_model == 'vgg19_2_2':
        netF = VGGFeat(n_layers=19, i_max_pool=1, include_max_pool=True) # i starts at 0
    elif which_model == 'vgg19_5_4':
        netF = VGGFeat(n_layers=19, i_max_pool=4, include_max_pool=False) # dont need last max_pool
    else:
        raise NotImplementedError('Feature Extractor model [%s] is not recognized' % which_model)

    netF = nn.DataParallel(netF, device_ids=gpu_ids)
    netF.eval() # No need to train
    return netF

def define_D(opt):
    gpu_ids = opt.gpu_ids
    opt = opt.network
    which_model = opt.which_model_D

    if which_model == 'single_label_96': # Assume input h and w are 96.
        netD = D.SingleLabelDiscriminator_96(input_ndc=opt.output_ngc, ndf=opt.ndf)
    else:
        raise NotImplementedError('Discriminator model [%s] is not recognized' % which_model)

    netD = nn.DataParallel(netD, device_ids=gpu_ids)
    netD.train() # Always in train mode

    return netD
