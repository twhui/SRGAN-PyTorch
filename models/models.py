from __future__ import print_function

def create_model(opt):
    if opt.model == 'sr_resnet':
        from .sr_resnet_model import SRResNetModel
        model = SRResNetModel()
    elif opt.model == 'sr_resnet_test':
        from .sr_resnet_test_model import SRResNetTestModel
        model = SRResNetTestModel()

    elif opt.model == 'sr_gan':
        from .sr_gan_model import SRGANModel
        model = SRGANModel()

    else:
        raise NotImplementedError('Model [%s] not recognized.' % opt.model)
    model.initialize(opt)
    print('Model [%s] is created.' % model.name())
    return model
