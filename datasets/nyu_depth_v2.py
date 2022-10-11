import os

import numpy as np
import pandas as pd

import cv2
import albumentations as A

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class NYU_Depth_V2(Dataset):
    """
    File Structure:
    nyu_data
    ├── nyu2_train.csv
    ├── nyu2_test.csv
    ├── data
    │   ├── nyu2_train (50688 images)
    │   │   ├── basement_0001a_out
    │   │   │   ├── 1.jpg -------------------> image
    │   │   │   ├── 1.png -------------------> depth (8-bit)
    │   │   │   ├── 2.jpg
    │   │   │   ├── 2.png
    │   │   │   ├── ...
    │   │   ├── basement_0001b_out
    │   │   ├── bathroom_0001_out
    │   │   ├── ...
    │   ├── nyu2_test (654 images)
    │   │   ├── 00000_colors.png ------------> image
    │   │   ├── 00000_depth.png -------------> depth (16-bit)
    │   │   ├── 00001_colors.png
    │   │   ├── 00001_depth.png
    │   │   ├── ...

    CSV Structure (no header, no index):
    data/nyu2_train/basement_0001a_out/1.jpg, data/nyu2_train/basement_0001a_out/1.png
    data/nyu2_train/basement_0001a_out/2.jpg, data/nyu2_train/basement_0001a_out/2.png
    ...
    data/nyu2_train/basement_0001b_out/1.jpg, data/nyu2_train/basement_0001b_out/1.png
    ...

    Notice:
    1. The depth image is a 3-channel image, but the depth value is the same in all channels.
    2. The depth value of train folder is stored in 8-bit format, of test folder is stored in 16-bit format.
    3. real maximum depth value is 10m.
    """
    def __init__(self, data_path, csv_path, transforms, is_train=True):
        self.data_path = data_path
        self.transforms = transforms
        self.is_train = is_train

        self.file_csv = pd.read_csv(os.path.join(data_path, csv_path), header=None)

    def __len__(self):
        return len(self.file_csv)

    def __getitem__(self, idx):
        item = self.file_csv.iloc[idx]
        image_path = os.path.join(self.data_path, item[0])
        depth_path = os.path.join(self.data_path, item[1])
        filename = '_'.join(item[0].split('/')[-2:])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(depth_path, -1)
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)

        transformed = self.transforms(image=image, mask=depth)
        image = transformed["image"]
        depth = transformed["mask"]
        depth = depth.permute(2, 0, 1)[[0]]

        if self.is_train:
            depth = depth / 255.0 * 10.0
        else:
            depth = depth / 1000.0

        return image, depth, filename
