import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from . import block as B

import time

def Loss(loss_type):
	if loss_type == 'mse':
		loss = nn.MSELoss
	elif loss_type == 'bce':
		loss = nn.BCELoss
	elif loss_type == 'vanilla':
		loss = AdversarialLoss
	else:
		raise NotImplementedError('Loss [%s] is not found' % loss_type)
	return loss


def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"


class _Loss(nn.Module):
    def __init__(self, size_average=True):
        super(_Loss, self).__init__()
        self.size_average = size_average

"""
Implemented GAN loss: [vanilla | lsgan ]
vanilla --> BCE (With Logits) Loss
lsgan --> MSE Loss
It abstracts away the need to create the target label tensor
that has the same size as the input
"""
class AdversarialLoss(_Loss):
	def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0, size_average=True):
		super(AdversarialLoss, self).__init__(size_average)
		self.gan_type = gan_type.lower()
		self.real_label_val = real_label_val
		self.fake_label_val = fake_label_val
		self.register_buffer('real_label', Variable(torch.Tensor()))
		self.register_buffer('fake_label', Variable(torch.Tensor()))

		self.loss = nn.BCEWithLogitsLoss(size_average=size_average)

	def _get_target_label(self, input, target_is_real):
		if target_is_real:
			if self.real_label.size() != input.size(): # check if new label needed
				self.real_label.data.resize_(input.size()).fill_(self.real_label_val)
			return self.real_label
		else:
			if self.fake_label.size() != input.size(): # check if new label needed
				self.fake_label.data.resize_(input.size()).fill_(self.fake_label_val)
			return self.fake_label

	def forward(self, input, target_is_real):
		target_label = self._get_target_label(input, target_is_real)
		loss = self.loss(input, target_label)
		return loss
