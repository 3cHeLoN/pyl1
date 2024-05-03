"""Linear operators."""

import numpy as np
import scipy.sparse.linalg
from numpy.typing import NDArray


class OpTV3D(scipy.sparse.linalg.LinearOperator):
    """Object that implements the total variation operator."""

    def __init__(self, row_count, col_count, slice_count):
        """Initialize TV operator."""
        self.dtype = np.float32

        self.row_count = row_count
        self.col_count = col_count
        self.slice_count = slice_count

        matrix_row_count = row_count * col_count * slice_count
        self.shape = (3 * matrix_row_count, matrix_row_count)

        super().__init__(self.dtype, self.shape)

        self.input_size = (slice_count, row_count, col_count)
        self.transpose_optv3d = OpTVTranspose(self)

    def rmatvec(self, x):
        """Backward product."""
        grad_x = x[: self.shape[1]].reshape(self.input_size).copy()
        grad_y = x[self.shape[1] : 2 * self.shape[1]].reshape(self.input_size).copy()
        grad_z = x[2 * self.shape[1] :].reshape(self.input_size).copy()

        vol_x = -grad_x
        vol_x[:, :, 1:] += grad_x[:, :, :-1]
        vol_y = -grad_y
        vol_y[:, 1:, :] += grad_y[:, :-1, :]
        vol_z = -grad_z
        vol_z[1:, :, :] += grad_z[:-1, :, :]

        return (vol_x + vol_y + vol_z).ravel()

    def _transpose(self):
        return self.transpose_optv3d

    def _matvec(self, x):
        """Forward product."""
        volume = x.reshape(self.input_size)

        grad_x = -volume
        grad_x[:, :, :-1] += volume[:, :, 1:]
        grad_y = -volume
        grad_y[:, :-1, :] += volume[:, 1:, :]
        grad_z = -volume
        grad_z[:-1, :, :] += volume[1:, :, :]
        return np.concatenate((grad_x.ravel(), grad_y.ravel(), grad_z.ravel()))


class OpTV2D(scipy.sparse.linalg.LinearOperator):
    """Object that implements the total variation operator."""

    def __init__(self, row_count, col_count):
        """Initialize TV operator."""
        self.dtype = np.float32

        self.row_count = row_count
        self.col_count = col_count

        matrix_row_count = row_count * col_count
        self.shape = (2 * matrix_row_count, matrix_row_count)

        super().__init__(self.dtype, self.shape)

        self.input_size = (row_count, col_count)
        self.transpose_optv2d = OpTVTranspose(self)

    def _transpose(self):
        return self.transpose_optv2d

    def _matvec(self, x):
        """Forward product."""
        volume = x.reshape(self.input_size)

        # compute the diff in each direction
        grad_x = -volume
        grad_x[:, :-1] += volume[:, 1:]
        grad_y = -volume
        grad_y[:-1, :] += volume[1:, :]

        # combine the results
        return np.concatenate((grad_x.ravel(), grad_y.ravel()))

    def rmatvec(self, x):
        """Backward product."""
        grad_x = x[: self.shape[1]].reshape(self.input_size)
        grad_y = x[self.shape[1] :].reshape(self.input_size)

        vol_x = -grad_x
        vol_x[:, 1:] += grad_x[:, :-1]
        vol_y = -grad_y
        vol_y[1:, :] += grad_y[:-1, :]

        return (vol_x + vol_y).ravel()


class OpTVTranspose(scipy.sparse.linalg.LinearOperator):
    """Object that provides the transpose operator ".T" of an OpTV object."""

    def __init__(self, parent: scipy.sparse.linalg.LinearOperator) -> None:
        """Create the total variation transposed operator.

        Args:
            parent: The parent linear operator.
        """
        self.parent = parent
        self.dtype = np.float32
        self.shape = (parent.shape[1], parent.shape[0])

        super().__init__(self.dtype, self.shape)

    def rmatvec(self, x: NDArray) -> NDArray:
        """Adjoint matrix-vector multiplication."""
        return self.parent.matvec(x)

    def _tranpose(self) -> scipy.sparse.linalg.LinearOperator:
        return self.parent

    def _matvec(self, x: NDArray) -> NDArray:
        return self.parent.rmatvec(x)


class OpTranspose(scipy.sparse.linalg.LinearOperator):
    """Object that provides the transpose operator ".T" of an operator object."""

    def __init__(self, parent: scipy.sparse.linalg.LinearOperator) -> None:
        """Create a transposed operator.

        Args:
            parent: The parent operator to transpose.
        """
        self.parent = parent
        self.dtype = np.float32
        self.shape = (parent.shape[1], parent.shape[0])

        super().__init__(self.dtype, self.shape)

    def rmatvec(self, x: NDArray) -> NDArray:
        """Adjoint matrix-vector multiplication.

        Args:
            x: The input vector.

        Returns:
            The adjoint matrix-vector product.
        """
        return self.parent.matvec(x)

    def _tranpose(self):
        return self.parent

    def _matvec(self, x):
        return self.parent.rmatvec(x)
