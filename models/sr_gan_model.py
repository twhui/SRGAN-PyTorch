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

class SRGANModel(BaseModel):
    def name(self):
        return 'SRGANModel'

    def initialize(self, opt):
        super(SRGANModel, self).initialize(opt)
        assert opt.is_train

        self.debug = opt.logger.debug
        self.input_L = self.Tensor()
        self.input_H = self.Tensor()

        print('Pytorch version:', torch.__version__)

        # For generator (G)
        # Spatial
        self.use_spatial_G = opt.train.lambda_spatial_G is not None
        self.lambda_spatial_G = opt.train.lambda_spatial_G if self.use_spatial_G else 0.0
        if self.use_spatial_G:
            self.criterion_spatial_G = opt.train.criterion_spatial_G
            self.loss_spatial_G = Loss(self.criterion_spatial_G)()
            if opt.gpu_ids:
                self.loss_spatial_G.cuda(opt.gpu_ids[0])

        # VGG
        self.use_vgg_G = opt.train.lambda_vgg_G is not None
        self.lambda_vgg_G = opt.train.lambda_vgg_G if self.use_vgg_G else 0.0
        if self.use_vgg_G:
            self.netF = networks.define_F(opt)
            self.loss_vgg_G = Loss(opt.train.criterion_vgg_G)()
            if opt.gpu_ids:
                self.loss_vgg_G.cuda(opt.gpu_ids[0])

        # For discriminator (D)
        # Adversarial
        self.use_adversarial_D = opt.train.lambda_adversarial_G is not None and opt.train.lambda_adversarial_D is not None
        self.lambda_adversarial_G = opt.train.lambda_adversarial_G if self.use_adversarial_D else 0.0
        self.lambda_adversarial_D = opt.train.lambda_adversarial_D if self.use_adversarial_D else 0.0
        if self.use_adversarial_D:
            self.netD = networks.define_D(opt)      # Should use model "single_label_96"
            self.update_steps_D = 1                 # Number of updates of D per each training iteration
            self.loss_adversarial_D = Loss(opt.train.criterion_adversarial_D)(opt.train.criterion_adversarial_D)
            if opt.gpu_ids:
                self.loss_adversarial_D.cuda(opt.gpu_ids[0])

        # Always define netG
        self.netG = networks.define_G(opt) # Should use model "sr_resnet"

        # Load pretrained_models (F always pretrained)
        self.load_path_G = opt.path.pretrain_model_G
        self.load_path_D = opt.path.pretrain_model_D
        self.load_path_F = opt.path.pretrain_model_F
        self.load()

        if opt.train.lr_scheme == 'multi_steps':
            self.lr_steps = self.opt.train.lr_steps
            self.lr_gamma = self.opt.train.lr_gamma

        self.optimizers = []

        self.lr_G = opt.train.lr_G
        self.weight_decay_G = opt.train.weight_decay_G if opt.train.weight_decay_G else 0.0
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.lr_G, weight_decay=self.weight_decay_G)
        self.optimizers.append(self.optimizer_G)

        self.lr_D = opt.train.lr_D
        self.weight_decay_D = opt.train.weight_decay_D if opt.train.weight_decay_D else 0.0
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=self.lr_D, weight_decay=self.weight_decay_D)
        self.optimizers.append(self.optimizer_D)

        print('---------- Model initialized -------------')
        self.write_description()
        print('------------------------------------------')

    def feed_data(self, data, volatile=False):
        input_H = data['H']
        input_L = data['L']
        self.input_H.resize_(input_H.size()).copy_(input_H)
        self.input_L.resize_(input_L.size()).copy_(input_L)
        self.real_H = Variable(self.input_H, volatile=volatile) # in range [0,1]
        self.real_L = Variable(self.input_L, volatile=volatile) # in range [0,1]

    def forward_G(self):
        self.fake_H = self.netG(self.real_L)

    def forward_F(self):
        self.real_H_feat = self.netF(self.real_H).detach()
        self.fake_H_feat = self.netF(self.fake_H)

    def forward_D(self, for_update_D):
        if for_update_D:
            self.pred_real_H = self.netD(self.real_H)
            self.pred_fake_H = self.netD(self.fake_H.detach())      # For update D only. No BP to G, so detach fake_H for speed improvement
        else:
            self.pred_fake_H = self.netD(self.fake_H)               # For update G only

    def backward_G(self, retain_graph):
        loss = 0.0

        # Spatial loss
        if self.use_spatial_G:
            self.scaledLoss_spatial_G = self.lambda_spatial_G * self.loss_spatial_G(self.fake_H,self.real_H)
            loss = loss + self.scaledLoss_spatial_G

        # VGG loss
        if self.use_vgg_G:
            self.scaledLoss_vgg_G = self.lambda_vgg_G * self.loss_vgg_G(self.fake_H_feat,self.real_H_feat)
            loss = loss + self.scaledLoss_vgg_G

        # Adversarial loss from D
        if self.use_adversarial_D:
            self.scaledLoss_adversarial_G = self.lambda_adversarial_G * self.loss_adversarial_D(self.pred_fake_H,target_is_real=True)
            loss = loss + self.scaledLoss_adversarial_G

        # Combined loss
        loss.backward(retain_graph=retain_graph)

    def backward_D(self, retain_graph):
        # Adversarial loss
        if self.use_adversarial_D:
            self.scaled_loss_adversarial_pos_D = self.lambda_adversarial_D * self.loss_adversarial_D(self.pred_real_H,target_is_real=True)
            self.scaled_loss_adversarial_neg_D = self.lambda_adversarial_D * self.loss_adversarial_D(self.pred_fake_H,target_is_real=False)
            loss = self.scaled_loss_adversarial_pos_D + self.scaled_loss_adversarial_neg_D

        loss.backward(retain_graph=retain_graph)

    def optimize_parameters(self, step):

        # Generator
        self.forward_G()                            # Forward pass

        if self.use_vgg_G:
            self.forward_F()

        self.forward_D(for_update_D=False)          # Forward pass
        self.optimizer_G.zero_grad()                # Zero the gradients before running the backward pass.
        self.backward_G(retain_graph=True)          # Compute loss and backward pass
        self.optimizer_G.step()                     # Calling the step function on an Optimizer makes an update to its parameters

        # Discriminator
        for _ in range(self.update_steps_D):
            self.forward_D(for_update_D=True)       # Forward pass
            self.optimizer_D.zero_grad()            # Zero the gradients before running the backward pass.
            self.backward_D(retain_graph=False)     # Compute loss and backward pass
            self.optimizer_D.step()                 # Calling the step function on an Optimizer makes an update to its parameters

    def val(self):
        self.forward_G()

    def get_current_losses(self):
        out_dict = OrderedDict()

        if self.use_spatial_G:
            out_dict['Spatial loss for G'] = self.scaledLoss_spatial_G.data[0]
        if self.use_vgg_G:
            out_dict['VGG loss for G'] = self.scaledLoss_vgg_G.data[0]

        if self.use_adversarial_D:
            out_dict['Adv. loss for G'] = self.scaledLoss_adversarial_G.data[0]
            out_dict['Adv. loss for pos. D'] = self.scaled_loss_adversarial_pos_D.data[0]
            out_dict['Adv. loss for neg. D'] = self.scaled_loss_adversarial_neg_D.data[0]

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

        s, n = get_network_description(self.netD.module)
        print('Number of parameters in D: %d' % n)
        message += '-------------- Discriminator --------------\n' + s + '\n'
        total_n += n

        network_path = os.path.join(self.save_dir, 'network.txt')
        with open(network_path, 'w') as f:
            f.write(message)
        os.chmod(network_path, S_IREAD|S_IRGRP|S_IROTH)

    def load(self):
        if self.load_path_G is not None:
            print('loading model for G [%s] ...' % self.load_path_G)
            load_network(self.load_path_G, self.netG)
        if self.load_path_D is not None:
            print('loading model for D [%s] ...' % self.load_path_D)
            load_network(self.load_path_D, self.netD)

    def save(self, iter_label):
        save_network(self.save_dir, self.netG, 'G', iter_label, self.opt.gpu_ids)
        save_network(self.save_dir, self.netD, 'D', iter_label, self.opt.gpu_ids)

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
