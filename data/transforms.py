from PIL import Image
import torchvision.transforms as transforms
import random

def crop(img, i, j, h, w):
    """Crop the given PIL.Image.
    Args:
        img (PIL.Image): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
    Returns:
        PIL.Image: Cropped image.
    """
    return img.crop((j, i, j + w, i + h))

class ModCrop(object):
    """ModCrop the given PIL.Image.
    Args:
        mod (int): Crop to make the output size divisible by mod.
    """

    def __init__(self, mod):
        self.mod = int(mod)

    @staticmethod
    def get_params(img, mod):
        """Get parameters for ``crop`` for mod crop.
        Args:
            img (PIL.Image): Image to be cropped.
            mod (int): Crop to make the output size divisible by mod.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for mod crop.
        """
        w, h = img.size
        tw = w - w % mod
        th = h - h % mod
        return 0, 0, th, tw

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        i, j, h, w = self.get_params(img, self.mod)
        return crop(img, i, j, h, w)

"""
Modified from the source "torchvision.transforms".
Use scale_factor as input instead of outputsize
"""
def scale(img, size, interpolation=Image.BICUBIC):
    assert isinstance(size, tuple) and len(size) == 2
    return img.resize(size[::-1], interpolation) # flip to (h,w) to (w,h)

class Scale(object):
    def __init__(self, scale_factor, interpolation=Image.BICUBIC):
        self.scale_factor = scale_factor
        self.interpolation = interpolation

    @staticmethod
    def get_params(img, scale_factor):
        w, h = img.size
        tw = int(w * scale_factor)
        th = int(h * scale_factor)
        return (th, tw)

    def __call__(self, img):
        size = self.get_params(img, self.scale_factor)
        return scale(img, size, self.interpolation)

def ToTensor():
    return transforms.ToTensor()

def get_transform_H(opt):
    transform_list = []
    if opt.modcrop_size is not None:
        transform_list.append(ModCrop(mod=opt.modcrop_size))
    if opt.crop_size is not None:
        transform_list.append(transforms.RandomCrop(opt.crop_size))
    if opt.use_flip: # Don't flip by default
        transform_list.append(transforms.RandomHorizontalFlip())
    return transforms.Compose(transform_list)

def get_transform_L(opt):
    assert opt.downscale in (2, 4, 8)
    scale = 1 / opt.downscale
    transform_list = []
    transform_list.append(Scale(scale_factor=scale, interpolation=Image.BICUBIC))
    return transforms.Compose(transform_list)
