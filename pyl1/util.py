"""Utility functions for tomography."""
import numpy as np
import matplotlib.pyplot as plt


class RealtimeImager(object):

    """Image picture animation in realtime."""

    def __init__(self, image_0, vmin=None, vmax=None, cmap='gray'):
        """Initialize object."""
        self.figure = plt.figure()
        self.axis = self.figure.add_subplot(111)
        self.figure.canvas.draw()
        self.vmin = vmin
        self.vmax = vmax
        if vmin is None:
            vmin = np.quantile(image_0, 0.05)
        if vmax is None:
            vmax = np.quantile(image_0, 0.95)
        self.axes_image = plt.imshow(image_0, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.show(block=False)
        plt.draw()

    def update(self, image):
        """Show the next image frame."""
        if self.vmin is None:
            vmin = np.quantile(image, 0.05)
        else:
            vmin = self.vmin
        if self.vmax is None:
            vmax = np.quantile(image, 0.95)
        else:
            vmax = self.vmax

        #self.axes_image.set_array(image / image.max())
        self.axes_image.set_clim(vmin, vmax)
        self.axes_image.set_data(image)
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
