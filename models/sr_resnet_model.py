import os
from stat import S_IREAD, S_IRGRP, S_IROTH
from collections import OrderedDict

import torch
from torch.autograd import Variable

import models.networks as networks
import models.modules.block as B
from models.modules.loss import Loss
from models.modules.util import get_network_description, load_network, save_network
from models.base_model import BaseModel

class SRResNetModel(BaseModel):
    def name(self):
        return 'SRResNetModel'

    def initialize(self, opt):
        super(SRResNetModel, self).initialize(opt)
        assert opt.is_train

        self.input_L = self.Tensor()
        self.input_H = self.Tensor()

        self.use_spatial = opt.train.lambda_spatial is not None
        self.lambda_spatial = opt.train.lambda_spatial if self.use_spatial else 0.0
        if self.use_spatial:
            self.criterion_spatial = Loss(opt.train.criterion_spatial)()
            if opt.gpu_ids:
                self.criterion_spatial.cuda(opt.gpu_ids[0])

        self.netG = networks.define_G(opt)

        # Load pretrained_models
        self.load_path_G = opt.path.pretrain_model_G
        self.load()

        if opt.train.lr_scheme == 'multi_steps':
            self.lr_steps = self.opt.train.lr_steps
            self.lr_gamma = self.opt.train.lr_gamma

        self.optimizers = []

        self.lr_G = opt.train.lr_G
        self.weight_decay_G = opt.train.weight_decay_G if opt.train.weight_decay_G else 0.0
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.lr_G, weight_decay=self.weight_decay_G)
        self.optimizers.append(self.optimizer_G)

        print('---------- Model initialized -------------')
        self.write_description()
        print('-----------------------------------------------')


    def feed_data(self, data, volatile=False):
        input_H = data['H']
        input_L = data['L']
        self.input_H.resize_(input_H.size()).copy_(input_H)
        self.input_L.resize_(input_L.size()).copy_(input_L)
        self.real_H = Variable(self.input_H, volatile=volatile) # in range [0,1]
        self.real_L = Variable(self.input_L, volatile=volatile) # in range [0,1]

    def forward_G(self):
        self.fake_H = self.netG(self.real_L)

    def backward_G(self):
        self.loss_spatial = self.lambda_spatial * self.criterion_spatial(self.fake_H, self.real_H)
        self.loss_spatial.backward()

    def optimize_parameters(self, step):
        # G
        self.forward_G()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def val(self):
        self.fake_H = self.netG(self.real_L)

    def get_current_losses(self):
        out_dict = OrderedDict()
        if self.use_spatial:
            out_dict['spatial'] = self.loss_spatial.data[0]
        return out_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['low-resolution'] = self.real_L.data[0]
        out_dict['super-resolution'] = self.fake_H.data[0]
        out_dict['ground-truth'] = self.real_H.data[0]
        return out_dict

    def write_description(self):
        total_n = 0
        message = ''
        s, n = get_network_description(self.netG.module)
        # print(s)
        print('Number of parameters in G: %d' % n)
        message += '-------------- Generator --------------\n' + s + '\n'
        total_n += n

        network_path = os.path.join(self.save_dir, 'network.txt')
        with open(network_path, 'w') as f:
            f.write(message)
        os.chmod(network_path, S_IREAD|S_IRGRP|S_IROTH)

    def load(self):
        if self.load_path_G is not None:
            print('loading model for G [%s] ...' % self.load_path_G)
            load_network(self.load_path_G, self.netG)

    def save(self, iter_label):
        save_network(self.save_dir, self.netG, 'G', iter_label, self.opt.gpu_ids)

    def update_learning_rate(self, step=None, scheme=None):
        if scheme == 'multi_steps':
            if step in self.lr_steps:
                for optimizer in self.optimizers:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * self.lr_gamma
                print('learning rate switches to next step.')

    def train(self):
        self.netG.train()

    def eval(self):
        self.netG.eval()
