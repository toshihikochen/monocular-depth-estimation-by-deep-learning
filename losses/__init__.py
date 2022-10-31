from losses.depth_loss import DepthLoss
from losses.gradient_loss import GradientLoss
from losses.ssim_loss import SSIMLoss

from losses.loss import Loss, MultiScaleLoss

__all__ = ["Loss", "MultiScaleLoss", "DepthLoss", "GradientLoss", "SSIMLoss"]
