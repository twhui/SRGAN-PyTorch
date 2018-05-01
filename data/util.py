import os
from PIL import Image
import torchvision.transforms as transforms


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_image_paths(dir):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    assert images, '%s has no valid image file' % dir
    return images

def get_image_paths_recursive(dir, images):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, subdirs, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                fpath = os.path.join(root, fname)
                images.append(fpath)
        for subdir in subdirs:
            subdir = os.path.join(root, subdir)
            get_image_paths_recursive(subdir, images)
    return images
