"""Chambolle Pock 3D example."""

import astra
import numpy as np
from phantom_3d import phantom3d

from pyl1.operators import OpTV3D
from pyl1.solvers import ChambollePock


def main() -> None:
    """Apply Chambolle-Pock."""
    volume = phantom3d(n=256)

    proj_geom = astra.create_proj_geom(
        "parallel3d", 1.0, 1.0, 384, 384, np.linspace(0, np.pi, 20)
    )
    vol_geom = astra.create_vol_geom(256, 256, 256)
    proj_id = astra.create_projector("cuda3d", proj_geom, vol_geom)

    w_operator = astra.optomo.OpTomo(proj_id)
    tv_operator = OpTV3D(256, 256, 256)

    rhs = w_operator * volume.ravel()

    cp_alg = ChambollePock(
        w_operator, tv_operator, rhs, show=True, max_iter=100, nonnegative=True
    )

    x = cp_alg.run()
    print("The solution is:", x)


if __name__ == "__main__":
    main()
