import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable

# Assume input in range [0, 1]
class VGGFeat(nn.Module):
    def __init__(self, n_layers=16, use_bn=False, i_max_pool=5, include_max_pool=False):
        super(VGGFeat, self).__init__()
        if n_layers == 11:
            if use_bn:
                model = torchvision.models.vgg11_bn(pretrained=True)
            else:
                model = torchvision.models.vgg11(pretrained=True)
        elif n_layers == 13:
            if use_bn:
                model = torchvision.models.vgg13_bn(pretrained=True)
            else:
                model = torchvision.models.vgg13(pretrained=True)
        elif n_layers == 16:
            if use_bn:
                model = torchvision.models.vgg16_bn(pretrained=True)
            else:
                model = torchvision.models.vgg16(pretrained=True)
        elif n_layers == 19:
            if use_bn:
                model = torchvision.models.vgg19_bn(pretrained=True)
            else:
                model = torchvision.models.vgg19(pretrained=True)
        else:
            raise NotImplementedError('Only support n_layers in [11, 13, 16, 19]')


        self.features = self.__break_layers(model.features, i_max_pool, include_max_pool)

        mean = Variable(torch.Tensor([0.485, 0.456, 0.406]).view(1,3,1,1)) # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
        std = Variable(torch.Tensor([0.229, 0.224, 0.225]).view(1,3,1,1)) # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    # Take up to i th max_pool layer. i start at 0.
    def __break_layers(self, features, i, include_max_pool=False):
        # Find the indices of max_pool layers
        children = list(features.children())
        max_pool_indices = [index for index, m in enumerate(children) if isinstance(m, nn.MaxPool2d)]
        target_features = children[:max_pool_indices[i] + 1] if include_max_pool else children[:max_pool_indices[i]]
        return nn.Sequential(*target_features)

    def forward(self, input):
        input = (input - self.mean) / self.std
        output = self.features(input)
        return output
