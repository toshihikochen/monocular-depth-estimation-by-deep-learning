# Monocular Depth Estimation by Deep Learning

This repository is to explore the monocular depth estimation methods by deep learning with single model.

We provide the code for training and testing the data and models.

# Menu

* [Requirements](#requirements)
* [Results](#results)
* [Fast Start](#start)
* [Project Structure](#project)
* [Configs](#configs)
* [Functions Definition](#functions)

# <div id="requirements"/>Requirements

* Python >= 3.7 is OK
* The deep learning algorithm is based on [PyTorch](https://pytorch.org/).
* The image augmentations and transforms is based on [albumentations](https://albumentations.ai/).
* The metrics algorithm is based on [torchmetrics](https://torchmetrics.readthedocs.io/en/latest/).

You can get more information from [requirements.txt](requirements.txt).

# <div id="result">Results

We use [standard_transforms.yml](configs/train/standard_transforms.yml) to train and the result is as below.

* Input resolution: 320x240x3 (RGB)
* Epoch: 20
* Optimizer: Adam
* Learning Rate: 1e-4, multiply 0.1 per 5 epochs
* Weight Decay: 1e-4
* EMA Weight: 0.1


| Model Name        | MParams | GMultAdds | BatchNorm | Activation | Dropout | Val_RMSE | Val_SSIM | Val_TA1 | EMA_RMSE | EMA_SSIM | EMA_TA1 |
| ----------------- | ------- | --------- | --------- | ---------- | ------- | -------- | -------- | ------- | -------- | -------- | ------- |
| Efficient-B0-UNet | 5.8779  | 4.3991    | False     | ReLU       | 0.0     | 0.7073   | 0.9563   | 0.7613  | 0.7514   | 0.9700   | 0.7132  |
| Dense-161-UNet    | 26.4216 | 21.3807   | False     | ReLU       | 0.0     | 1.2688   | 0.9594   | 0.8774  | 1.4152   | 0.9605   | 0.4947  |
| Res-101-UNet      | 62.8516 | 26.8846   | False     | ReLU       | 0.0     | 0.7292   | 0.9631   | 0.7206  | 0.7612   | 0.9652   | 0.7001  |
| VGG-16-UNet       | 18.5869 | 34.1176   | False     | ReLU       | 0.0     | 0.6906   | 0.9584   | 0.7360  | 0.6907   | 0.9588   | 0.7360  |

* The lower the better: Params, MultAdds, RMSE
* The higher the better: SSIM, TA

# <div id="start"/>Fast Start

## Train a model

0. Make sure Python, CUDA and CUDNN (or run with CPU if you like 😄) is perfectly installed.
1. Clone or download the code and change direction `cd` to the project
2. Run `pip install -r requirement.txt` to install necessary libraries.
3. Run `python train.py -c configs/train/light_transforms.yml -m configs/model/vgg_unet.yml`
   you can choose another config file or edit your own one.
4. Wait and done

## Deploy on RaspberryPi

The raspberry pi device is connected with a RGB camera.

It captures video, predicts depth and sends frames and depths to server.

0. Make sure Python and OpenCV is perfectly installed.
1. Clone or download the code and change direction `cd` to the project.
2. Run `pip install -r requirement.txt` to install necessary libraries.
3. Train a model or download the model somewhere😄.
4. Run `python raspi.py -c configs/raspi/raspi.yml` to start client.
   you can configure IP address, port, path of model in the yml file.

## Deploy on server

Server receives frames and depths from raspberry pi and shows the frames and depths with OpenCV.

0. Make sure you can access to the client.
1. Run `python raspi.py -c configs/raspi/raspi_server.yml` to start server .
   you can configure IP address, port in the yml file.

Because the communication between client and server is based on Socket,
you can use any language to implement it.

# <div id="project"/>Project Structure

## datasets

You can put your dataset code in this folder.

We provide [NYU Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) dataset code as an example.

## losses

This folder contains loss functions code for measuring distance and directing model optimized.

We provide depth loss, gradient loss and ssim loss.  See [Functions Definition](#functions) for mathematically details.

## metrics

This folder contains metrics code for measuring model quality

We use [torchmetrics](https://torchmetrics.readthedocs.io/en/latest/) to calculate and summary the metrics.

We provide log10 mean average error, average ssim, edge f1 score and threshold accuracy.
See [Functions Definition](#functions)  for mathematically details.

## models

This folder contains model code.

We provide models:

* DenseUNet(UNet with DenseNet-169 encoder)
* EfficientUNet(UNet with EfficientNet-B0 encoder)
* ResUNet(UNet with ResNet-101 encoder)
* VGGUNet(UNet with VGG-16 encoder)
* DenseFPN(Pyramid with DenseNet-169 backbone)
* EfficientFPN(Pyramid with EfficientNet-B0 backbone)
* ResFPN(Pyramid with ResNet-101 backbone)
* VGGFPN(Pyramid with VGG-16 backbone)

## trainers

This folder contains trainer classes for controlling training process. You can inherit the base trainer and implement your own trainer.

We provide a base trainer and a EMA trainer example.

## transforms

This folder contains image and label augmentation and transforms code.
We provide transforms with different level for ablation study.

We use [albumentations](https://albumentations.ai/) to do the image augmentations and transforms.

## utils

You can put your customized utilities code in this folder.

## train.py

Your can run this file to train your model.

Please add or edit the config/\*/\*.yaml to set your training parameters and
using parameter "--config"(or "-c" in short) and "--model"(or "-m" in short) to specify the config file and model file.

Detail samples can be found in [configs](configs) folder.

## test.py

You can run this file to test your model.

Also config yml file ("--config") and model yml file ("--model") should be provided.

## demo.py

## quantization.py

This code is to compress your model and make it occupy less memory and inference faster.

# <div id="configs"/>Configs

## Models

`model_name`: the name of the model

`pretrained`: encoder/bottom_up module using image-net weights (True/False)

UNet Only

`norm`: adding BatchNormal layers in the decoder (True/False)

`activation`: using ReLU(=0), LeakyReLU(>0) or Sigmoid(<0) in the decoder

`dropout`:  Dropout rate in the decoder, 0 means no dropout

FPN Only

`single`: Single output(True) or multiple outputs(False) of the FPN model

## Train

`dataset_path`: the root directory of the dataset

`train/val_resolution`: the resolution of the inputs when training/validating

`transforms_level`: the level of preset transforms(light, standard, heavy), the higher the more augmentation.
you can write your own one.

`num_epochs` number of training epochs

`batch_size` number of samples putting into model once

`learning_rate`: the learning rate of the Adam optimizer

`weight_decay`: the weight decay rate of the optimizer

`ema_weight`: the weight when updating EMA model. EMA means Exponential Moving Average

## Test

It is same as Train section.

## Quantization

To be updated 😕

# <div id="functions"/>Functions Definition

definition:

* **$y$**: the ground truth depth map of the input image.
* **$\hat y$**: the predicted depth map of the input image.

## depth loss

$$
L(y, \hat y) = \frac{1}{n} \sum_{p}^n |y-\hat y|

$$

## gradient loss

$$
L_{grad}(y, \hat y) = \frac{1}{n} \sum_{p}^n |g_x(y_p) - g_x(\hat y_p)| + |g_y(y_p) - g_y(\hat y_p)| \\ 
g_x(Y)=Y_{ij}-Y_{(i+s)j} \\
g_y(Y)=Y_{ij}-Y_{i(j+s)} \\

$$

where $s$ is the step size. $s=1$ by default.

## ssim loss

$$
L_{ssim}(y, \hat y) = 1-SSIM(y, \hat y)

$$

See [structural similarity](https://en.wikipedia.org/wiki/Structural_similarity) for more detail about SSIM.

## log10 mean absolute error

$$
MAE_{log_{10}}(y, \hat y) = \frac{1}{n} \sum_{p}^n |log_{10}(y_p) - log_{10}(\hat y_p)|

$$

## MAPE (Mean Average Percentage Error)

$$
MAPE(y, \hat y)=\frac1n\sum_p^n|\frac{y-\hat y}{y}|

$$

## RMSE (Root Mean Square Error)

$$
RMSE(y, \hat y)=\sqrt{\frac1n\sum_p^n(y_p-\hat y_p)^2}

$$

## threshold accuracy

$$
TA(y, \hat y) = max(\frac{y_p}{\hat y_p}, \frac{\hat y_p}{y_p}) < thr \in [1.25, 1.25^2, 1.25^3]

$$

## edge f1 score

$$
PixelOnEdge(Y)=\sqrt{Sobel_x^2(Y)+Sobel_y^2(Y)} > thr \in [0.25, 0.5, 1.0] \\
F_1=2\times\frac{precision \times recall}{precision + recall} \\
precision = \frac{TP}{TP+FP}; recall = \frac{TP}{TP+FN}

$$
