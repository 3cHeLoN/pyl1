"""Chambolle Pock 3D example."""

import astra
import imageio
import matplotlib.pyplot as plt
import numpy as np

from pyl1.operators import OpTV2D
from pyl1.solvers import ChambollePock


def main() -> None:
    """Apply Chambolle-Pock.

    This method needs the ASTRA toolbox (https://astra-toolbox.com/)
    """
    image = imageio.imread("shepp256.png").astype(float)
    image = image / image.max()
    image[image == 0] = 0.2

    proj_geom = astra.create_proj_geom("parallel", 1.0, 384, np.linspace(0, np.pi, 20))
    vol_geom = astra.create_vol_geom(256, 256)
    proj_id = astra.create_projector("cuda", proj_geom, vol_geom)

    # Create tomography operator
    w_operator = astra.optomo.OpTomo(proj_id)
    tv_operator = OpTV2D(256, 256)

    rhs = w_operator * image.ravel()

    # solve tv-minimization
    cp_alg = ChambollePock(
        w_operator,
        tv_operator,
        rhs,
        max_iter=800,
        tv_weight=10,
        nonnegative=True,
        show=True,
    )

    x = cp_alg.run()

    plt.imshow(x.reshape(astra.geom_size(vol_geom)), cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()
