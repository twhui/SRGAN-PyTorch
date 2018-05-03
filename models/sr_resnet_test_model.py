from collections import OrderedDict

from torch.autograd import Variable

import models.networks as networks
import models.modules.block as B
from .modules.util import load_network
from .base_model import BaseModel


class SRResNetTestModel(BaseModel):
    def name(self):
        return 'SRResNetTestModel'

    def initialize(self, opt):
        super(SRResNetTestModel, self).initialize(opt)
        assert not opt.is_train

        self.input_L = self.Tensor()
        self.input_H = self.Tensor()

        self.netG = networks.define_G(opt).eval()
        self.load_path_G = opt.path.pretrain_model_G
        assert self.load_path_G is not None
        self.load()

        print('---------- Model initialized -------------')


    def feed_data(self, data, volatile=True):
        input_H = data['H']
        input_L = data['L']
        self.input_H.resize_(input_H.size()).copy_(input_H)
        self.input_L.resize_(input_L.size()).copy_(input_L)
        self.real_H = Variable(self.input_H, volatile=volatile) # in range [0,1]
        self.real_L = Variable(self.input_L, volatile=volatile) # in range [0,1]

    def test(self):
        self.fake_H = self.netG(self.real_L)

    def val(self):
        self.test()

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['low-resolution'] = self.real_L.data[0]
        out_dict['super-resolution'] = self.fake_H.data[0]
        out_dict['ground-truth'] = self.real_H.data[0]
        return out_dict

    def load(self):
        print('loading model for G [%s] ...' % self.load_path_G)
        load_network(self.load_path_G, self.netG)
