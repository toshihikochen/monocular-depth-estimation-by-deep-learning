import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt


class Cmapper:

    def __init__(self, cmap, maximum=None, minimum=None):
        self.maximum = maximum
        self.minimum = minimum

        self.cmapper = cm.get_cmap(cmap)

    def color_bar(self):
        # generate color bar image
        depth = np.linspace((self.minimum, ), (self.maximum, ), 1000, axis=1)
        # resize
        # depth = np.expand_dims(depth, axis=0)
        depth = np.resize(depth, (100, 1000))
        # generate color bar
        depth = self(depth)
        # plot
        img = plt.imshow(depth)
        # adjust axis
        img.axes.get_yaxis().set_visible(False)
        # new x axis
        plt.xticks(np.linspace(0, 1000, 11), np.linspace(self.minimum, self.maximum, 11))
        return img

    def __call__(self, depth):
        # normalize
        if self.minimum is None:
            self.minimum = depth.min()
        if self.maximum is None:
            self.maximum = depth.max()

        depth = (depth - self.minimum) / (self.maximum - self.minimum)

        depth = self.cmapper(depth, bytes=True)[:, :, :3]
        return depth
