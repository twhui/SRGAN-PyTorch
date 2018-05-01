import torch
import torch.nn as nn
import models.modules.generator as G

# Pick the corresponding model for Generator and initialize it.
def define_G(opt):
    gpu_ids = opt.gpu_ids
    opt = opt.network
    which_model = opt.which_model_G

    if which_model == 'sr_resnet':
        netG = G.SRResNetGenerator(input_nc=opt.input_nc, output_nc=opt.output_nc, nf=opt.nf, nb=opt.nb, norm_type=opt.norm_type)
    else:
        raise NotImplementedError('Generator model [%s] is not recognized' % which_model)

    netG = nn.DataParallel(netG, device_ids=gpu_ids)
    return netG
