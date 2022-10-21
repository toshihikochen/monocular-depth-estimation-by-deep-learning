import argparse
import os
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models import *

from datasets import NYU_Depth_V2
from transforms import val_transforms
from utils import Timer

argparser = argparse.ArgumentParser()
argparser.add_argument("-c", "--config", type=str, default="configs/quantization/quantization.yml", help="path to config file")
argparser.add_argument("-m", "--model", type=str, default="configs/model/model.yml", help="path to model parameter file")
args = argparser.parse_args()

# read yml file
print("config file: ", args.config)
print("model file: ", args.model)
with open(args.config, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
with open(args.model, "r") as f:
    model_config = yaml.load(f, Loader=yaml.FullLoader)

device = config["device"]

# model
model_name = model_config["model_name"]
norm = model_config["norm"]
activation = model_config["activation"]
dropout = model_config["dropout"]

# quantization arguments
quantization = config["quantization"]
model_selection = config["model_selection"]
checkpoint_path = config["checkpoint_path"]
output_dir = config["output_dir"]
qconfig = config["qconfig"]   # 'fbgemm' or 'qnnpack'

# calibration arguments
dataset_path = config["dataset_path"]
num_samples = config["num_samples"]
resolution = config["resolution"]
batch_size = config["batch_size"]
random_seed = config["random_seed"]
num_workers = config["num_workers"]
pin_memory = config["pin_memory"]
prefetch_factor = config["prefetch_factor"]

# load checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device)
# model name
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
# select model
if model_selection == "model":
    weight = checkpoint["model"]
elif model_selection == "ema":
    weight = checkpoint["ema_model"]
else:
    raise ValueError("model selection error")

model.load_state_dict(weight, strict=False)
model.to(device)

transforms = val_transforms(resolution)
dataset = NYU_Depth_V2(
    data_path=dataset_path,
    csv_path="nyu2_test.csv",
    transforms=transforms,
    is_train=False,
)
dataloader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
    drop_last=False,
    prefetch_factor=prefetch_factor
)

if quantization:
    # quantize model
    class QuantizeModel(nn.Module):
        def __init__(self, model):
            super(QuantizeModel, self).__init__()
            self.quant = torch.quantization.QuantStub()
            self.model = model
            self.dequant = torch.quantization.DeQuantStub()

        def forward(self, x):
            x = self.quant(x)
            x = self.model(x)
            x = self.dequant(x)
            return x

    model = QuantizeModel(model)
    model.qconfig = torch.quantization.get_default_qconfig(qconfig)
    model = torch.quantization.prepare(model, inplace=True)

    # calibrate model
    model.eval()
    with torch.no_grad():
        for i in range(num_samples):
            image, _, _ = next(iter(dataloader))
            image = image.to(device, dtype=torch.float32, non_blocking=True)
            model(image)

    # convert
    model = torch.quantization.convert(model, inplace=True)

    # save quantized model
    torch.save(model, os.path.join(output_dir, "quantized_model.pt"))

else:
    # save model
    torch.save(model, os.path.join(output_dir, "model.pt"))

# test
model.eval()
timer = Timer()
timer.start()
with torch.no_grad():
    for i in range(num_samples):
        image, _, _ = next(iter(dataloader))
        image = image.to(device, dtype=torch.float32, non_blocking=True)
        model(image)

elapsed = timer.stop()
print("Average inference time: {:.4f} ms".format(elapsed / num_samples * 1000))
