import argparse
import os
import random
import warnings
import yaml

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import torchmetrics as tm
from torchsummaryX import summary

from datasets import NYU_Depth_V2
from losses import Loss
from metrics import EdgeF1Score, Log10AverageError, StructuralSimilarityIndexMeasure, ThresholdAccuracy
from models import DenseUNet, EfficientUNet, ResNetUNet, VGGUNet
from trainers import EMATrainer
from transforms import val_transforms, light_train_transforms, standard_train_transforms, heavy_train_transforms

warnings.filterwarnings("ignore")

# argparse
argparser = argparse.ArgumentParser()
argparser.add_argument("-c", "--config", type=str, help="path to config file")
args = argparser.parse_args()

# read yml file
with open(args.config, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# training arguments
# dataset
dataset_path = config["dataset_path"]
# resolution
train_resolution = config["train_resolution"]
val_resolution = config["val_resolution"]
# transforms
transforms_level = config["transforms_level"]
# model
model_name = config["model_name"]
pretrained = config["pretrained"]
# training
num_epochs = config["num_epochs"]
batch_size = config["batch_size"]
learning_rate = config["learning_rate"]
weight_decay = config["weight_decay"]
ema_weight = config["ema_weight"]
# device
device = config["device"]
# checkpoints
checkpoints_dir = config["checkpoints_dir"]
save_checkpoint_per_num_epochs = config["save_checkpoint_per_num_epochs"]
# display
verbose = config["verbose"]
# other
random_seed = config["random_seed"]
num_workers = config["num_workers"]
pin_memory = config["pin_memory"]
prefetch_factor = config["prefetch_factor"]

# outputs directory
os.makedirs(checkpoints_dir, exist_ok=True)

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

# augmentations and transforms
val_transforms = val_transforms(val_resolution)
if transforms_level == "light":
    train_transforms = light_train_transforms(train_resolution)
elif transforms_level == "standard":
    train_transforms = standard_train_transforms(train_resolution)
elif transforms_level == "heavy":
    train_transforms = heavy_train_transforms(train_resolution)
else:
    raise ValueError(f"Invalid transforms level: {transforms_level}")

# dataset and dataloaders
train_dataset = NYU_Depth_V2(
    data_path=dataset_path,
    csv_path="nyu2_train.csv",
    transforms=train_transforms,
)
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
    drop_last=True,
    prefetch_factor=prefetch_factor
)
val_dataset = NYU_Depth_V2(
    data_path=dataset_path,
    csv_path="nyu2_test.csv",
    transforms=val_transforms,
)
val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    drop_last=False,
    prefetch_factor=prefetch_factor
)
test_dataset = NYU_Depth_V2(
    data_path=dataset_path,
    csv_path="nyu2_test.csv",
    transforms=val_transforms,
)
test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    drop_last=False,
    prefetch_factor=prefetch_factor
)

# model
if model_name.lower() == "vgg_unet":
    model = VGGUNet(pretrained=pretrained)
elif model_name.lower() == "res_unet":
    model = ResNetUNet(pretrained=pretrained)
elif model_name.lower() == "dense_unet":
    model = DenseUNet(pretrained=pretrained)
elif model_name.lower() == "efficient_unet":
    model = EfficientUNet(pretrained=pretrained)
else:
    raise ValueError("Invalid model name")

ema_avg = lambda averaged_model_parameter, model_parameter, _: \
    model_parameter * (1 - ema_weight) + averaged_model_parameter * ema_weight
ema_model = optim.swa_utils.AveragedModel(model, avg_fn=ema_avg)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)
criterion = Loss()
metrics = tm.MetricCollection({
    "Log10AE": Log10AverageError(full_state_update=False),
    "MAPE": tm.MeanAbsolutePercentageError(full_state_update=False),
    "RMSE": tm.MeanSquaredError(squared=False),
    "SSIM": StructuralSimilarityIndexMeasure(),
    "TA1": ThresholdAccuracy(threshold=1.25 ** 1),
    "TA2": ThresholdAccuracy(threshold=1.25 ** 2),
    "TA3": ThresholdAccuracy(threshold=1.25 ** 3),
    # "EdgeF1_25": EdgeF1Score(threshold=0.25),
    # "EdgeF1_50": EdgeF1Score(threshold=0.5),
    # "EdgeF1_100": EdgeF1Score(threshold=1.0),
}).requires_grad_(False)

print("Model summary: ")
summary(model, torch.rand(1, 3, *train_resolution))

# trainer
trainer = EMATrainer(
    model=model,
    ema_model=ema_model,
    optimizer=optimizer,
    criterion=criterion,
    metrics=metrics,
    tensorboard_dir=checkpoints_dir,
)
trainer.to(device)
trainer.train(
    num_epochs=num_epochs,
    train_loader=train_dataloader,
    val_loader=val_dataloader,
    verbose=verbose,
    save_checkpoint_per_num_epochs=save_checkpoint_per_num_epochs,
    checkpoints_dir=checkpoints_dir
)
trainer.save_checkpoint(os.path.join(checkpoints_dir, "checkpoint.pkl"))
trainer.history_to_csv(os.path.join(checkpoints_dir, "history.csv"))

print("Done! ")
