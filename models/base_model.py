import os
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
from torch.autograd import Variable

from models.modules.loss import Loss
from models.modules.util import get_network_description, load_network, save_network

"""
Interface for all model
"""
class BaseModel(object):
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.Tensor = torch.cuda.FloatTensor if opt.gpu_ids else torch.Tensor
        self.save_dir = opt.path.trained_models

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_losses(self):
        pass

    def write_description(self):
        pass

    def save(self, label):
        pass

    def load(self):
        pass

    def update_learning_rate(self):
        pass
