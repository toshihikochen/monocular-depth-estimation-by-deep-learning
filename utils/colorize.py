import numpy as np
import matplotlib.cm as cm

cmap = "plasma"
cmapper = cm.get_cmap(cmap)


def colorize(depth, maximum=None, minimum=None):
    if len(depth.shape) == 4:
        depth = depth[0]

    # normalize
    if minimum is None:
        minimum = depth.min()
    if maximum is None:
        maximum = depth.max()

    depth = (depth - minimum) / (maximum - minimum)

    depth = cmapper(depth, bytes=True)[:, :, :3]
    return depth
