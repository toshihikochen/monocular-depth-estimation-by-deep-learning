import numpy as np
import matplotlib.cm as cm


class Cmapper:

    def __init__(self, cmap, maximum=None, minimum=None):
        self.maximum = maximum
        self.minimum = minimum

        self.cmapper = cm.get_cmap(cmap)

    def __call__(self, depth):
        # normalize
        if self.minimum is None:
            self.minimum = depth.min()
        if self.maximum is None:
            self.maximum = depth.max()

        depth = (depth - self.minimum) / (self.maximum - self.minimum)

        depth = self.cmapper(depth, bytes=True)[:, :, :3]
        return depth
