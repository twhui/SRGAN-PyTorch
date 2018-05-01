from __future__ import print_function
import os

class Logger(object):
    def __init__(self, opt):
        self.opt = opt.logger
        self.log_dir = opt.path.log

        self.loss_log_path = os.path.join(self.log_dir, 'loss_log.txt')
        with open(self.loss_log_path, "a") as log_file:
            log_file.write('================ Training Losses ================\n')

        self.val_log_path = os.path.join(self.log_dir, 'val_log.txt')
        with open(self.val_log_path, "a") as log_file:
            log_file.write('================ Validation Results ================\n')


    def print_results(self, results, epoch, iters, time, mode):
        message = '(epoch: %3d, iters: %8d, time: %.3f) ' % (epoch, iters, time)
        for label, value in results.items():
            message += '%s: %.6f ' % (label, value)
        # Print in console
        print(message)
        # Write in log file
        if mode == 'loss':
            with open(self.loss_log_path, "a") as log_file:
                log_file.write('%s\n' % message)
        elif mode == 'val':
            with open(self.val_log_path, "a") as log_file:
                log_file.write('%s\n' % message)
