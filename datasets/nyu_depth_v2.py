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
    │   │   │   ├── 1.png -------------------> depth
    │   │   │   ├── 2.jpg
    │   │   │   ├── 2.png
    │   │   │   ├── ...
    │   │   ├── basement_0001b_out
    │   │   ├── bathroom_0001_out
    │   │   ├── ...
    │   ├── nyu2_test (654 images)
    │   │   ├── 00000_colors.png ------------> image
    │   │   ├── 00000_depth.png -------------> depth
    │   │   ├── 00001_colors.png
    │   │   ├── 00001_depth.png
    │   │   ├── ...

    """
    def __init__(self, data_path, csv_path, transforms, depth_scale=1000.):
        self.data_path = data_path
        self.transforms = transforms
        self.depth_scale = depth_scale

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
        depth = cv2.imread(depth_path)
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)

        transformed = self.transforms(image=image, mask=depth)
        image = transformed["image"]
        depth = transformed["mask"]
        depth = depth.permute(2, 0, 1)[[0]] / 255.0 * self.depth_scale

        return image, depth, filename
