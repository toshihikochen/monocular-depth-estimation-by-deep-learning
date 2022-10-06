# Monocular Depth Estimation by Deep Learning

This repository is to explore the monocular depth estimation methods by deep learning with single model.

We provide the code for training and testing the data and models. 
Also, we provide a template for other deep learning methods.

# Requirements

* The deep learning algorithm is based on [PyTorch](https://pytorch.org/).
* The image augmentations and transforms is based on [albumentations](https://albumentations.ai/).
* The metrics algorithm is based on [torchmetrics](https://torchmetrics.readthedocs.io/en/latest/).

You can get more information from [requirements.txt](requirements.txt).

# Results

Not done yet. :)

# Project Structure

## datasets

You can put your dataset code in this folder. 

We provide a [NYU Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) dataset example.

## losses

You can put your loss code in this folder.

We provide depth loss, gradient loss and ssim loss.  See [Functions](#Functions) for mathematically details.

## metrics

You can put your customized metric code in this folder. 

We use [torchmetrics](https://torchmetrics.readthedocs.io/en/latest/) to calculate the metrics. 

We provide log10 mean average error, average ssim and threshold accuracy. See [Functions](#Functions) for 
mathematically details.

## models

You can put your model code in this folder.

We provide models: 

* DenseUNet(UNet with DenseNet-169 encoder)
* EfficientUNet(UNet with EfficientNet-B0 encoder)
* ResUNet(UNet with ResNet-101 encoder)
* VGGUNet(UNet with VGG-16 encoder)

## trainers

You can put your trainer code in this folder. You can inherit the base trainer and implement your own trainer.

We provide a base trainer and a EMA trainer example. 

## transforms

You can put your customized transform code in this folder. 
We provide transforms with different level for ablation study.

We use [albumentations](https://albumentations.ai/) to do the image augmentations and transforms.

## utils

You can put your customized utility code in this folder.

## train.py

Your can run this file to train your model. 

Please add or edit the *config/\*.yaml* to set your training parameters and 
using parameter "--config" or "-c" to specify the config file. Detail samples can be found in [configs](configs) folder.

## test.py

Your can run this file to test your model.


# <div id="Functions"></div> Functions 

definition:

* **$y$**: the ground truth depth map of the input image.
* **$\hat y$**: the predicted depth map of the input image.

## depth loss

$$
L(y, \hat y) = \lambda L_1(y, \hat y) + L_{grad}(y, \hat y) + L_{ssim}(y, \hat y)
$$

## gradient loss

$$
L_{grad}(y, \hat y) = \frac{1}{n} \sum_{p}^n |g_x(y_p - \hat y_p)| + |g_y(y_p - \hat y_p)|
$$

## ssim loss

$$
L_{ssim}(y, \hat y) = \frac{1-SSIM(y, \hat y)}{2}
$$

## log10 mean average error

$$
log10MAE = \frac{1}{n} \sum_{p}^n |log_{10}(y_p) - log_{10}(\hat y_p)|
$$

## threshold accuracy

$$
TA = max(\frac{y_p}{\hat y_p}, \frac{\hat y_p}{y_p}) < thr \in [1.25, 1.25^2, 1.25^3]
$$
