import os
from stat import S_IREAD, S_IRGRP, S_IROTH
from collections import OrderedDict
from datetime import datetime
import json
from box import Box
"""
Read the config file and parse it to a option object.
"""

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

def parse(opt_path, is_train=True):
    with open(opt_path, 'r') as f:
        opt = json.load(f, object_pairs_hook=OrderedDict)
    opt['is_train'] = is_train

    for key, path in opt['path'].items():
        opt['path'][key] = os.path.expanduser(path)
    if is_train:
        experiments_root = os.path.join(opt['path']['root'], 'experiments', opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['options'] = experiments_root
        opt['path']['trained_models'] = os.path.join(experiments_root, 'trained_models')
        opt['path']['log'] = os.path.join(experiments_root, 'log')
    else:
        results_root = os.path.join(opt['path']['root'], 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = os.path.join(results_root, 'log')
        opt['path']['test_images'] = os.path.join(results_root, 'test_images')

    for dataset in opt['datasets']:
        dataset['dataroot'] = os.path.expanduser(dataset['dataroot'])
    return opt

def save(opt):
    dump_dir = opt['path']['options']
    dump_path = os.path.join(dump_dir, 'options.json')
    with open(dump_path, 'w') as dump_file:
        json.dump(opt, dump_file, indent=2)
    os.chmod(dump_path, S_IREAD|S_IRGRP|S_IROTH)

def dict2box(opt):
    return Box(opt, default_box=True, default_box_attr=None)
