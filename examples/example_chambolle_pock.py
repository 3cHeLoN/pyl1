"""Chambolle Pock 3D example."""

import numpy as np
import imageio
import astra
import matplotlib.pyplot as plt
from pyl1.operators import ChambollePock
from pyl1.operators import OpTV2D


def main() -> None:
    image = imageio.imread("shepp256.png").astype(float)
    image = image / image.max()
    image[image == 0] = 0.2

    proj_geom = astra.create_proj_geom("parallel", 1.0, 384, np.linspace(0, np.pi, 20))
    vol_geom = astra.create_vol_geom(256, 256)
    proj_id = astra.create_projector("cuda", proj_geom, vol_geom)

    # Create tomography operator
    W = astra.optomo.OpTomo(proj_id)
    TV = OpTV2D(256, 256)

    rhs = W * image.ravel()

    # solve tv-minimization
    cp_alg = ChambollePock(
        W, TV, rhs, max_iter=800, tv_weight=10, nonnegative=True, show=True
    )

    x = cp_alg.run()

    plt.imshow(x.reshape(astra.geom_size(vol_geom)), cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()
