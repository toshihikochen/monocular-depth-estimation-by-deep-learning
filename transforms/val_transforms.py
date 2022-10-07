from functools import partial

import cv2
import albumentations as A
import albumentations.pytorch as AP

from transforms.mask_resize import _mask_resize


def val_transforms(resolution):
    return A.Compose([
        A.Resize(*resolution),

        A.Lambda(mask=partial(_mask_resize, height=resolution[0] // 2, width=resolution[1] // 2)),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        AP.ToTensorV2()
    ])
