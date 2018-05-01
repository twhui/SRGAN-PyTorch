from __future__ import print_function
import os
from datetime import datetime
from PIL import Image

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

def save_img_np(img_np, img_path, mode='RGB'):
    img_pil = Image.fromarray(img_np, mode=mode)
    img_pil.save(img_path)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)

def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archivedAt_' + get_timestamp()
        print('Path already exists. Rename it to [%s]' % new_name)
        os.rename(path, new_name)
    os.makedirs(path)
