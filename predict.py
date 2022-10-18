import argparse
import os
import random
import warnings
import yaml

import numpy as np

import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import NYU_Depth_V2
from models import DenseUNet, EfficientUNet, ResNetUNet, VGGUNet
from trainers import EMATrainer
from transforms import val_transforms
from utils import Cmapper, Timer

# ignore warnings
warnings.filterwarnings("ignore")

# argparse
argparser = argparse.ArgumentParser()
argparser.add_argument("-c", "--config", type=str, help="path to config file")
args = argparser.parse_args()

# read yml file
with open(args.config, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# predicting arguments
# model
model_name = config["model_name"]
norm = config["norm"]
activation = config["activation"]
dropout = config["dropout"]
model_selection = config["model_selection"]
# checkpoints path
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

# image and depth label transforms
transforms = val_transforms(resolution)

# dataset and dataloaders
dataset = NYU_Depth_V2(
    data_path=dataset_path,
    csv_path="nyu2_test.csv",
    transforms=transforms,
    is_train=False,
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
    model = VGGUNet(pretrained=False, norm=norm, activation=activation, dropout=dropout)
elif model_name.lower() == "res_unet":
    model = ResNetUNet(pretrained=False, norm=norm, activation=activation, dropout=dropout)
elif model_name.lower() == "dense_unet":
    model = DenseUNet(pretrained=False, norm=norm, activation=activation, dropout=dropout)
elif model_name.lower() == "efficient_unet":
    model = EfficientUNet(pretrained=False, norm=norm, activation=activation, dropout=dropout)
else:
    raise ValueError("Invalid model name")

ema_model = optim.swa_utils.AveragedModel(model)

trainer = EMATrainer(
    model=model,
    ema_model=ema_model,
)
trainer.to(device)
trainer.load_checkpoint(checkpoint_path)

# predict
result_cmapper = Cmapper(cmap="plasma" ,maximum=10, minimum=0)
result_color_bar = result_cmapper.color_bar()
result_color_bar.figure.savefig(os.path.join(outputs_dir, "result_color_bar.png"), bbox_inches="tight")

error_cmapper = Cmapper(cmap="PiYG" ,maximum=10, minimum=-10)
error_color_bar = error_cmapper.color_bar()
error_color_bar.figure.savefig(os.path.join(outputs_dir, "error_color_bar.png"), bbox_inches="tight")

timer = Timer()
timer.start()
for y_pred, y_true, filenames in trainer.test(dataloader, model_selection=model_selection, verbose=verbose):
    # upsample
    y_pred = F.interpolate(y_pred, size=(480, 640), mode="nearest")
    y_true = F.interpolate(y_true, size=(480, 640), mode="nearest")

    # convert to numpy
    y_pred = y_pred[0, 0].cpu().numpy()
    y_true = y_true[0, 0].cpu().numpy()

    # error between predicted and true depth and show the result in color map
    error = y_pred - y_true
    error = error_cmapper(error)
    error = cv2.cvtColor(error, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(outputs_dir, filenames[0].replace(".png", "_error.png")), error)

    # show the real depth result in color map
    y_pred = result_cmapper(y_pred)
    y_pred = cv2.cvtColor(y_pred, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(outputs_dir, filenames[0].replace(".png", "_output.png")), y_pred)

elapsed = timer.stop()
print(f"Done! time elapsed: {elapsed:.2f} seconds, FPS: {len(dataset) / elapsed:.2f}")

