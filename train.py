import time
import os
import math
import argparse
from collections import OrderedDict

import torch
import random
import numpy as np

import options.options as options
from utils import util, convert, metric

parser = argparse.ArgumentParser()
parser.add_argument('--options', type=str, required=True, help='Path to options JSON file.')
args = parser.parse_args()
options_path = args.options
opt = options.parse(options_path)
util.mkdir_and_rename(opt['path']['experiments_root']) # Rename old experiments if exists
util.mkdirs((path for key , path in opt['path'].items() if not key == 'experiments_root')) # Make all directories needed
options.save(opt) # Save option file to the opt['path']['options']
opt = options.dict2box(opt)

if opt.train.manual_seed is None:
    opt.train.manual_seed = random.randint(1, 10000)
print("Random Seed: ", opt.train.manual_seed)
random.seed(opt.train.manual_seed)
torch.manual_seed(opt.train.manual_seed)

from data.datasets import create_dataset
from data.data_loader import create_dataloader
from models.models import create_model
from utils.logger import Logger

def main():
    # Create train dataset
    train_set_opt = opt.datasets[0]
    train_set = create_dataset(train_set_opt)
    train_size = int(math.ceil(len(train_set) / train_set_opt.batch_size))
    print('Number of train images: %d batches of size %d' % (train_size, train_set_opt.batch_size))
    total_iters = int(opt.train.niter)
    total_epoches = int(math.ceil(total_iters / train_size))
    print('Total epoches needed: %d' % total_epoches)

    # Create val dataset
    val_set_opt = opt.datasets[1]
    val_set = create_dataset(val_set_opt)
    val_size = len(val_set)
    print('Number of val images: %d' % val_size)

    # Create dataloader
    train_loader = create_dataloader(train_set, train_set_opt)
    val_loader = create_dataloader(val_set, val_set_opt)

    # Create model
    model = create_model(opt)
    model.train()

    # Create logger
    logger = Logger(opt)

    current_step = 0
    need_make_val_dir = True
    start_time = time.time()
    for epoch in range(total_epoches):
        for i, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break

            train_start_time = time.time()
            # Training
            model.feed_data(train_data)
            model.optimize_parameters(current_step)
            train_duration = time.time() - train_start_time

            if current_step % opt.logger.print_freq == 0:
                losses = model.get_current_losses()
                logger.print_results(losses, epoch, current_step, train_duration, 'loss')

            if current_step % opt.logger.save_checkpoint_freq == 0:
                print('Saving the model at the end of current_step %d' % (current_step))
                model.save(current_step)

            # Validation
            if current_step % opt.train.val_freq == 0:
                validate(val_loader, val_size, model, logger, epoch, current_step)

            model.update_learning_rate(step=current_step, scheme=opt.train.lr_scheme)

        print('End of Epoch %d' % epoch)

    print('Saving the final model')
    model.save('latest')

    print('End of Training \t Time Taken: %d sec' % (time.time() - start_time))


def validate(val_loader, val_size, model, logger, epoch, current_step):
    print('Start validation phase ...')
    val_start_time = time.time()
    model.eval() # Change to eval mode. It is important for BN layers.

    val_results = OrderedDict()
    avg_psnr = 0.0
    for val_data in val_loader:
        img_path = val_data['path'][0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img_dir = os.path.join(opt.path.val_images, img_name)
        util.mkdir(img_dir)

        model.feed_data(val_data, volatile=True)
        model.val()

        visuals = model.get_current_visuals()

        sr_img = convert.tensor2img_np(visuals['super-resolution']) # uint8
        gt_img = convert.tensor2img_np(visuals['ground-truth']) # uint8
        cropped_sr_img = sr_img[4:-4, 4:-4, :]
        cropped_gt_img = gt_img[4:-4, 4:-4, :]

        cropped_sr_img = convert.rgb2y(cropped_sr_img)
        cropped_gt_img = convert.rgb2y(cropped_gt_img)

        # Calculate quantitative performance metric
        avg_psnr += metric.psnr(cropped_sr_img, cropped_gt_img)

    avg_psnr = avg_psnr / val_size
    val_results['psnr'] = avg_psnr

    val_duration = time.time() - val_start_time
    # Save to log
    logger.print_results(val_results, epoch, current_step, val_duration, 'val')
    model.train() # Change back to train mode.

if __name__ == '__main__':
    main()
