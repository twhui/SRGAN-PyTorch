import time
import os
import math
import argparse
from collections import OrderedDict

import torch
import random
import numpy as np

import options.options as options
import utils.util as util
import utils.metric as metric
import utils.convert as convert
from data.transforms import Scale
from torchvision.transforms import ToTensor
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--options', type=str, required=True, help='Path to options JSON file.')
args = parser.parse_args()
options_path = args.options
opt = options.parse(options_path, is_train=False)
util.mkdirs((path for key , path in opt['path'].items() if not key == 'pretrain_model_G')) # Make all directories needed
opt = options.dict2box(opt)


from data.datasets import create_dataset
from data.data_loader import create_dataloader
from models.models import create_model

# Create test dataset and dataloader
test_loaders = []
test_set_names = []
for dataset_opt in opt.datasets:
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    test_size = len(test_set)
    test_set_name = dataset_opt.name
    print('Number of test images in [%s]: %d' % (test_set_name, test_size))
    test_loaders.append(test_loader)
    test_set_names.append(test_set_name)

# Create model
model = create_model(opt)

# Path for log file
test_log_path = os.path.join(opt.path.log, 'test_log.txt')
if os.path.exists(test_log_path):
    os.remove(test_log_path)
    print('Old test log is removed.')

print('Start Testing ...')

for test_set_name, test_loader in zip(test_set_names, test_loaders):
    print('Testing [%s]...' % test_set_name)
    test_start_time = time.time()
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []

    log_str = '\nTest set: [%s]\n' % test_set_name
    print(log_str)
    for i, data in enumerate(test_loader):

        model.feed_data(data, volatile=True)
        img_path = data['path'][0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img_dir = os.path.join(opt.path.test_images, test_set_name, img_name)
        util.mkdir(img_dir)

        model.test() # test
        visuals = model.get_current_visuals()

        sr_img = convert.tensor2img_np(visuals['super-resolution']) # uint8
        gt_img = convert.tensor2img_np(visuals['ground-truth']) # uint8
        cropped_sr_img = sr_img[4:-4, 4:-4, :]
        cropped_gt_img = gt_img[4:-4, 4:-4, :]

        # Save SR images for reference
        save_img_path = os.path.join(img_dir, 'sr.png')
        util.save_img_np(sr_img, save_img_path)

        # Convert images to luma space
        cropped_sr_img = convert.rgb2y(cropped_sr_img)
        cropped_gt_img = convert.rgb2y(cropped_gt_img)

        # Calculate quantitative performance metric
        psnr = metric.psnr(cropped_sr_img, cropped_gt_img)
        ssim = metric.ssim(cropped_sr_img, cropped_gt_img)

        test_results['psnr'].append(psnr)
        test_results['ssim'].append(ssim)

        tmp_str = '\nImage [%s]\n' % (img_path)
        for label, values in test_results.items():
            tmp_str += '%s: %.4f \n' % (label, values[i])
        log_str += tmp_str
        print(tmp_str)

    test_duration = time.time() - test_start_time

    print('Time taken: %.3f' % test_duration)
    # Calculate Average
    tmp_str = '\n ------------Overall results------------'
    log_str += tmp_str + '\n'
    print(tmp_str)
    for label, values in test_results.items():
        avg_value = sum(values) / len(values)
        tmp_str =  'avg %s: %.4f \n' % (label, avg_value)
        log_str += tmp_str
        print(tmp_str)
    # Write in log file
    with open(test_log_path, "a") as log_file:
        log_file.write('%s' % log_str)
