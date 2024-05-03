"""Utilities."""

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray


class RealtimeImager:
    """Image picture animation in realtime."""

    def __init__(
        self,
        image_0: NDArray,
        vmin: float = None,
        vmax: float = None,
        cmap: str = "gray",
    ):
        """Real time imager.

        This imager can show images with minimal delay, making it possible to display
        animations.

        Args:
            image_0: The initial image.
            vmin: The minimum gray value.
            vmax: The maximum gray value.
            cmap: The colormap name.
        """
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

    def update(self, image: NDArray) -> None:
        """Show the next image frame."""
        vmin = self.vmin or np.quantile(image, 0.05)
        vmax = self.vmax or np.quantile(image, 0.95)

        self.axes_image.set_clim(vmin, vmax)
        self.axes_image.set_data(image)
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
