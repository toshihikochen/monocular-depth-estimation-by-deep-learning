import argparse
import os
import random
import warnings
import yaml

import numpy as np

import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import NYU_Depth_V2
from models import DenseUNet, EfficientUNet, ResNetUNet, VGGUNet
from trainers import EMATrainer
from transforms import val_transforms
from utils import colorize

warnings.filterwarnings("ignore")

# argparse
argparser = argparse.ArgumentParser()
argparser.add_argument("-c", "--config", type=str, help="path to config file")
args = argparser.parse_args()

# read yml file
with open(args.config, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# predicting arguments
# checkpoints path
model_name = config["model_name"]
checkpoint_path = config["checkpoint_path"]
# dataset
dataset_path = config["dataset_path"]
# resolution
resolution = config["resolution"]
# device
device = config["device"]
# outputs
outputs_dir = config["outputs_dir"]
# display
verbose = config["verbose"]
# other
random_seed = config["random_seed"]
num_workers = config["num_workers"]
pin_memory = config["pin_memory"]
prefetch_factor = config["prefetch_factor"]

# outputs directory
os.makedirs(outputs_dir, exist_ok=True)

# random seed
if random_seed is not None:
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
else:
    torch.backends.cudnn.benchmark = True

# transforms
transforms = val_transforms(resolution)

# dataset and dataloaders
dataset = NYU_Depth_V2(
    data_path=dataset_path,
    csv_path="nyu2_test.csv",
    transforms=transforms,
)
dataloader = DataLoader(
    dataset=dataset,
    batch_size=1,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    drop_last=False,
    prefetch_factor=prefetch_factor
)

# model
if model_name.lower() == "vgg_unet":
    model = VGGUNet()
elif model_name.lower() == "res_unet":
    model = ResNetUNet()
elif model_name.lower() == "dense_unet":
    model = DenseUNet()
elif model_name.lower() == "efficient_unet":
    model = EfficientUNet()
else:
    raise ValueError("Invalid model name")

ema_model = optim.swa_utils.AveragedModel(model)

trainer = EMATrainer(
    model=model,
    ema_model=ema_model,
    optimizer=None,
    criterion=None,
    metrics=None,
)
trainer.to(device)
trainer.load_checkpoint(checkpoint_path)

# predict
for y_pred, filenames in trainer.test(dataloader):
    y_pred = y_pred[0, 0].to(torch.uint8).cpu().numpy()
    y_pred = colorize(y_pred)
    cv2.imwrite(os.path.join(outputs_dir, filenames[0]), y_pred)

print("Done!")
