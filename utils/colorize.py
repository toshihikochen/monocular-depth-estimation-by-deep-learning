import numpy as np
import matplotlib.cm as cm

cmap = "plasma"
cmapper = cm.get_cmap(cmap)


def colorize(depth):
    if len(depth.shape) == 4:
        depth = depth[0]

    # normalize
    min = depth.min()
    max = depth.max()
    depth = (depth - min) / (max - min)

    depth = cmapper(depth, bytes=True)[:, :, :3]
    return depth
