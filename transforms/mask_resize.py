import cv2

def _mask_resize(mask, height, width, interpolation=cv2.INTER_NEAREST, **kwargs):
    return cv2.resize(mask, (width, height), interpolation=interpolation)
