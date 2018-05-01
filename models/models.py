from __future__ import print_function

def create_model(opt):
    if opt.model == 'sr':
        from .sr_model import SRModel
        model = SRModel()
    elif opt.model == 'sr_test':
        from .sr_test_model import SRTestModel
        model = SRTestModel()
    else:
        raise NotImplementedError('Model [%s] not recognized.' % opt.model)
    model.initialize(opt)
    print('Model [%s] is created.' % model.name())
    return model
