from functools import partial

import cv2
import albumentations as A
import albumentations.pytorch as AP

from transforms.mask_resize import _mask_resize


def light_train_transforms(resolution):
    return A.Compose([
        A.Resize(*resolution),
        A.HorizontalFlip(p=0.5),

        A.Lambda(mask=partial(_mask_resize, height=resolution[0] // 2, width=resolution[1] // 2)),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        AP.ToTensorV2()
    ])


def standard_train_transforms(resolution):
    return A.Compose([
        A.Resize(*resolution),
        A.HorizontalFlip(p=0.5),

        A.ChannelShuffle(p=0.25),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),

        A.Lambda(mask=partial(_mask_resize, height=resolution[0] // 2, width=resolution[1] // 2)),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        AP.ToTensorV2()
    ])


def heavy_train_transforms(resolution):
    return A.Compose([
        A.Resize(*resolution),
        A.HorizontalFlip(p=0.5),

        A.ChannelShuffle(p=0.25),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),

        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 15), sigma_limit=(0.05, 3), p=0.5),
            A.AdvancedBlur(blur_limit=(3, 15), sigmaX_limit=(0.05, 3), sigmaY_limit=(0.05, 3), beta_limit=(0.5, 3),
                           p=0.5),
        ], p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(0.5, 10), p=0.5),
            A.ISONoise(color_shift=(0.01, 0.05), p=0.5),
        ], p=0.5),
        A.OneOf([
            A.MultiplicativeNoise(per_channel=False, elementwise=True, p=1),
            A.MultiplicativeNoise(per_channel=True, elementwise=True, p=1),
        ], p=0.2),

        A.Lambda(mask=partial(_mask_resize, height=resolution[0] // 2, width=resolution[1] // 2)),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        AP.ToTensorV2()
    ])
