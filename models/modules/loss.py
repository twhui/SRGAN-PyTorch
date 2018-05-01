import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from . import block as B

import time

def Loss(loss_type):
	if loss_type == 'l1':
		loss = nn.L1Loss
	elif loss_type == 'mse':
		loss = nn.MSELoss

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
