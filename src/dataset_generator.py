# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 12:52:45 2024

@author: Mico
"""

import os
import PIL
from random import random
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torch.utils.data import Dataset


class DocDataset(Dataset):
    def __init__(self, df, dataset_dir, img_size=448, is_train=False):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train
        self.dataset_dir = dataset_dir
        self.img_size = img_size
        self.image_transforms = T.Compose([T.Resize((self.img_size, self.img_size)),
                                           T.ToTensor(),
                                           T.Normalize((0.5,), (0.5,))])

        self.target_transforms = T.Compose([T.Resize((self.img_size, self.img_size)),
                                            T.ToTensor()])
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.dataset_dir, row['image'])
        target_path = os.path.join(self.dataset_dir, row['target'])
        
        # read images
        img = PIL.Image.open(img_path)
        target = PIL.Image.open(target_path)
        
        # to grayscale
        img = img.convert('L')
        target = target.convert('L')
        
        # apply transforms and data augmentation
        img, target = self.apply_transforms(img, target)

        return img, target

    def apply_transforms(self, img, target):
        # padding
        img = square_padding(img)
        target = square_padding(target)
        crop_size = (self.img_size, self.img_size)
        if self.is_train:
            # random crop
            if random() > 0.5:
                new_size = (int(self.img_size * 1.5), int(self.img_size * 1.5))
                img = img.resize(new_size)
                target = target.resize(new_size)

                crop_params = T.RandomCrop.get_params(img, output_size=crop_size)
                img = img.crop(crop_params)
                target = target.crop(crop_params)

            # random flips
            if random() > 0.5:
                img = F.vflip(img)
                target = F.vflip(target)
            if random() > 0.5:
                img = F.hflip(img)
                target = F.hflip(target)

        # resize to a fixed size
        img = self.image_transforms(img)
        target = self.target_transforms(target)
        return img, target

def square_padding(img, fill_val=255):
    w, h = img.size
    max_side = max((h, w))
    x1 = (max_side - w) // 2
    x2 = max_side - w - x1
    y1 = (max_side - h) // 2
    y2 = max_side - h - y1
    img = F.pad(img, (x1, y1, x2, y2), fill_val, 'constant')
    return img
