"""Chambolle Pock 3D example."""

import numpy as np
import astra
from pyl1.solvers import ChambollePock
from pyl1.solvers import OpTV3D
from phantom_3d import phantom3d


def main() -> None:
    volume = phantom3d(n=256)

    proj_geom = astra.create_proj_geom(
        "parallel3d", 1.0, 1.0, 384, 384, np.linspace(0, np.pi, 20)
    )
    vol_geom = astra.create_vol_geom(256, 256, 256)
    proj_id = astra.create_projector("cuda3d", proj_geom, vol_geom)

    W = astra.optomo.OpTomo(proj_id)
    TV = OpTV3D(256, 256, 256)

    rhs = W * volume.ravel()

    tv = TV * volume.ravel()

    cp_alg = ChambollePock(W, TV, rhs, show=True, max_iter=100, nonnegative=True)

    x = cp_alg.run()


if __name__ == "__main__":
    main()
