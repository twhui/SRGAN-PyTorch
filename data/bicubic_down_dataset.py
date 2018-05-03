import os
import numpy as np
import torch
import torch.utils.data as data
from data.util import get_image_paths_recursive
import data.transforms as transforms
import utils.util as util

from PIL import Image

class BicubicDownDataset(data.Dataset):
    def initialize(self, dataset_opt):
        self.phase = dataset_opt.phase
        self.dir = dataset_opt.dataroot

        if dataset_opt.use_subset:
            subset_path = os.path.join(self.dir, 'subset.txt')
            with open(subset_path) as f:
                self.paths = [os.path.join(self.dir, line.rstrip('\n')) for line in f]
        else:
            self.paths = []
            self.paths = get_image_paths_recursive(self.dir, self.paths)
        self.size = len(self.paths)
        self.transform_H = transforms.get_transform_H(dataset_opt.transform_H)
        self.transform_L = transforms.get_transform_L(dataset_opt.transform_L)
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        path = self.paths[index]
        H_img = self.transform_H(Image.open(path).convert('RGB'))
        L_img = self.transform_L(H_img)
        H_tensor = self.to_tensor(H_img)
        L_tensor = self.to_tensor(L_img)
        out_dict = dict(L=L_tensor, H=H_tensor)
        if not self.phase == 'train':
            out_dict.update(path=path)
        return out_dict

    def __len__(self):
        return self.size

    def name(self):
        return 'BicubicDownDataset'
