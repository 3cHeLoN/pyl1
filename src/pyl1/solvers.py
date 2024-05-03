"""Linear sparse solvers."""

from typing import Callable

import scipy
import numpy as np
from numpy.typing import NDArray
from pyl1.util import RealtimeImager


def power_method(
    a_matrix: NDArray | scipy.sparse.linalg.LinearOperator, max_iter: int = 10
) -> float:
    """Power method.

    The power method is an algorithm for computing  the largest singular value of a
    linear operator.

    Args:
        a_matrix: The linear operator.
        max_iter: The maximum number of iterations.

    Returns:
        The largest singular value of the linear operator.
    """
    x_vec = np.random.random((a_matrix.shape[1],))
    y_vec = a_matrix * x_vec
    largest_singular_value = 0.0

    for iter_count in range(max_iter):
        x_vec = a_matrix.rmatvec(y_vec)
        x_vec = x_vec / np.linalg.norm(x_vec)
        y_vec = a_matrix * x_vec
        largest_singular_value = np.linalg.norm(y_vec)
        print(f"{iter_count} singular value: {largest_singular_value}")
    return largest_singular_value


class ChambollePock:
    """Chambolle-Pock algorithm for total-variation minimization."""

    def __init__(
        self,
        a_matrix: NDArray | scipy.sparse.linalg.LinearOperator,
        tv_operator: NDArray | scipy.sparse.linalg.LinearOperator,
        rhs: NDArray,
        x_0: NDArray = None,
        max_iter: int = 20,
        tv_weight: float = 10,
        nonnegative: bool = False,
        show: bool = False,
        a_scale: float = None,
        tv_scale: float = None,
        lipschitz_factor: float = None,
    ) -> None:
        """Initialize the Chambolle-Pock algorithm.

        Args:
            a_matrix: The linear operator.
            tv_operator: The total-variation operator.
            rhs: The right-hand-side vector.
            x_0: The initial solution.
            max_iter: The maximum number of iterations.
            tv_weight: The total variation weight.
            nonnegative: Whether a nonnegativity constraint is applied.
            show: Whether to show the intermediate residuals.
            a_scale: The linear operator scale.
            tv_scale: The total-variation scale.
            lipschitz_factor: The Lipschitz factor.
        """
        if isinstance(a_matrix, np.ndarray):
            a_matrix = scipy.sparse.linalg.aslinearoperator(a_matrix)
        if isinstance(tv_operator, np.ndarray):
            tv_operator = scipy.sparse.linalg.aslinearoperator(tv_operator)

        # save properties
        self.a_matrix = a_matrix
        self.tv_operator = tv_operator
        self.rhs = rhs
        self.x_0 = x_0
        self.max_iter = max_iter
        self.tv_weight = tv_weight
        self.nonnegative = nonnegative
        self.show = show
        self.a_scale = a_scale
        self.tv_scale = tv_scale
        self.lipschitz_factor = lipschitz_factor

    # pylint: disable=too-many-statements
    def run(self) -> NDArray:
        """Run the algorithm.

        Returns:
            The solution vector.
        """
        print("Determine A-scaling")
        largest_eigenvalue_a = self.a_scale or power_method(self.a_matrix, 15)

        print("Determine TV-scaling")
        largest_eigenvalue_tv = self.tv_scale or power_method(self.tv_operator, 15)

        # scale operators
        a_matrix = (1.0 / largest_eigenvalue_a) * self.a_matrix
        rhs = self.rhs / largest_eigenvalue_a
        tv_operator = (1.0 / largest_eigenvalue_tv) * self.tv_operator

        tv_weight = largest_eigenvalue_tv / (largest_eigenvalue_a**2) * self.tv_weight

        print("Determine Lipschitz-constant")
        largest_eigenvalue = self.lipschitz_factor or 1

        x_0 = self.x_0 or np.zeros((a_matrix.shape[1],))

        u_vec = x_0
        p_vec = np.zeros((len(rhs),))
        q_vec = np.zeros((tv_operator.shape[0],))

        x_len = len(x_0)

        # scaling parameters
        tau = 1 / largest_eigenvalue
        sigma = 1 / largest_eigenvalue
        theta = 1

        norm_b_0 = np.linalg.norm(rhs)
        u_bar_vec = u_vec

        residual_vec = np.zeros((self.max_iter,))
        relres_vec = np.zeros((self.max_iter,))
        objective_vec = np.zeros((self.max_iter,))
        dual_gap_vec = np.zeros((self.max_iter,))

        a_matrix_u_bar = a_matrix.matvec(u_bar_vec)
        tv_matrix_u_bar = tv_operator.matvec(u_bar_vec)

        increasing_objective_count = 0

        if self.show:
            print("Iter\tobjective\tprimal-dual gap\tresidual-norm^2\tTV-norm")
            print(
                "===================================================================="
            )
            imager = RealtimeImager(self.get_slice(u_bar_vec))

        for iter_count in range(self.max_iter):
            p_vec = (p_vec + sigma * (a_matrix_u_bar - rhs)) / (1 + sigma)
            q_vec = q_vec + sigma * tv_matrix_u_bar
            q_mat = q_vec.reshape((x_len, -1))
            magnitude = np.linalg.norm(q_mat, axis=1)
            magnitude = np.maximum(magnitude, tv_weight)
            q_mat = np.divide(
                tv_weight * q_mat, np.tile(magnitude, (q_mat.shape[1], 1)).T
            )
            q_vec = q_mat.flatten()

            if self.nonnegative:
                u_new_vec = np.maximum(
                    u_vec
                    - tau * a_matrix.rmatvec(p_vec)
                    - tau * tv_operator.rmatvec(q_vec),
                    0,
                )
            else:
                u_new_vec = u_vec - tau * a_matrix.rmatvec(p_vec)
                u_new_vec = u_new_vec - tau * tv_operator.rmatvec(q_vec)

            u_bar_vec = u_new_vec + theta * (u_new_vec - u_vec)
            u_vec = u_new_vec
            a_matrix_u_bar = a_matrix.matvec(u_bar_vec)
            tv_matrix_u_bar = tv_operator.matvec(u_bar_vec)
            residual_vec[iter_count] = np.linalg.norm(a_matrix_u_bar - rhs)
            relres_vec[iter_count] = residual_vec[iter_count] / norm_b_0
            objective_vec[iter_count] = 0.5 * np.linalg.norm(
                a_matrix_u_bar - rhs
            ) ** 2 + tv_weight * np.linalg.norm(tv_matrix_u_bar, ord=1)
            dual_gap_vec[iter_count] = (
                0.5 * np.linalg.norm(a_matrix_u_bar - rhs) ** 2
                + tv_weight * np.linalg.norm(tv_matrix_u_bar, ord=1)
                + 0.5 * np.linalg.norm(p_vec) ** 2
                - np.dot(p_vec, rhs)
            )

            if self.show:
                print(
                    f"{iter_count}\t{objective_vec[iter_count]:e}"
                    f"\t{dual_gap_vec[iter_count]:e}"
                    f"\t{residual_vec[iter_count]:e}"
                    f"\t{np.linalg.norm(tv_matrix_u_bar, ord=1):e}"
                )
                imager.update(self.get_slice(u_bar_vec))

            if iter_count > 0 and (
                objective_vec[iter_count] > objective_vec[iter_count - 1]
            ):
                increasing_objective_count += 1
                if increasing_objective_count >= 5:
                    x_vec = u_bar_vec
                    return u_bar_vec

        if self.show:
            print(f"Converged after {iter_count} iterations")

        x_vec = u_bar_vec
        return x_vec

    def get_slice(self, u_vec: NDArray) -> NDArray:
        """Get a slice of the 3d volume.

        Args:
            u_vec: The 3d volume as a vector.

        Returns:
            The image of the corresponding slice.
        """

        """Get slice of volume."""
        middle_slice_index = int(self.a_matrix.vshape[0] / 2)
        if len(self.a_matrix.vshape) == 3:
            im_slice = u_vec.reshape(self.a_matrix.vshape)[middle_slice_index, :, :]
        elif len(self.a_matrix.vshape) == 2:
            im_slice = u_vec.reshape(self.a_matrix.vshape)
        return im_slice

    @staticmethod
    def tvmin_power_method(
        w_matrix: NDArray | scipy.sparse.linalg.LinearOperator,
        d_matrix: NDArray | scipy.sparse.linalg.LinearOperator,
        max_iter: int = 10,
    ):
        """Compute the largest singular value of [W, D]."""
        x_vec = np.random.random((w_matrix.shape[1],))
        y_vec = w_matrix.matvec(x_vec)
        z_vec = d_matrix.matvec(x_vec)

        for iter_count in range(max_iter):
            # power iteration
            x_vec = w_matrix.rmatvec(y_vec) + d_matrix.rmatvec(z_vec)
            # normalize
            x_vec = x_vec / np.linalg.norm(x_vec)
            y_vec = w_matrix.matvec(x_vec)
            z_vec = d_matrix.matvec(x_vec)
            largest_eigenvalue = np.sqrt(np.dot(y_vec, y_vec) + np.dot(z_vec, z_vec))
            print(f"{iter_count}) singular value: {largest_eigenvalue}")
        return largest_eigenvalue


class Fista:  # pylint: disable=too-many-instance-attributes,too-few-public-methods
    """Implementation of FISTA l1-optimization.

    Based on the paper:
    [1] Amir Beck, Marc Teboulle, "A Fast Iterative Shrinkage-Thresholding Algorithm
        for Linear Inverse Problems", SIAM J. Imaging sciences, 2(1), pp. 183-202, 2009.
        https://people.rennes.inria.fr/Cedric.Herzet/Cedric.Herzet/Sparse_Seminar/Entrees/2012/11/12_A_Fast_Iterative_Shrinkage-Thresholding_Algorithmfor_Linear_Inverse_Problems_(A._Beck,_M._Teboulle)_files/Breck_2009.pdf
    """

    def __init__(
        self,
        operator: NDArray | Callable,
        rhs: NDArray,
        input_data: NDArray,
        regularizer: float = 0.5,
        max_iter: int = 10,
        operator_transposed: Callable = None,
        seed: float = None,
    ):  # pylint:disable=too-many-arguments
        """Create a FISTA instance.

        Args:
            operator: The linear operator.
            rhs: The right hand side.
            input_data: The input data.
            regularizer: The regularization factor.
            max_iter: The maximum number of iterations.
            operator_transposed: The function handle to the transposed operator.
            seed: The input seed.
        """
        if not callable(operator):
            self.operator_fun = operator.dot
            self.operator_transposed_fun = operator.T.dot
        else:
            if operator_transposed is None:
                raise ValueError("Missing transposed operator function handle!")
            if not callable(operator_transposed):
                raise ValueError("Transposed operator should be a function handle!")
            self.operator_fun = operator
            self.operator_transposed_fun = operator_transposed

        self.seed = seed
        self.rhs = rhs.copy()
        self.data = input_data.copy()
        self.max_iter = max_iter
        self.operator_size = (rhs.size, input_data.size)
        self.normalization = self._power_iteration() ** 2
        self.regularizer = regularizer

    def initialize(self, initial_vector: NDArray) -> None:
        """Initialize method.

        Args:
            initial_vector: The initial vector.
        """
        self.data = initial_vector.copy()

    def run(self) -> tuple[NDArray, NDArray]:
        """Run fista method.

        Returns:
            The solution vector and the residual vector.
        """
        residual = np.zeros((self.max_iter,))
        threshold = 1
        update = self.data.copy()

        for iteration in range(self.max_iter):
            data_copy = self.data.copy()
            update = (
                update
                + self.operator_transposed_fun(self.rhs - self.operator_fun(update))
                / self.normalization
            )
            self.data = self._soft_threshold(
                update, self.regularizer / self.normalization
            )
            threshold_copy = threshold
            threshold = (1 + np.sqrt(1 + 4 * threshold**2)) / 2
            update = self.data + ((threshold_copy - 1) / threshold) * (
                self.data - data_copy
            )
            residual[iteration] = 0.5 * np.linalg.norm(
                self.operator_fun(self.data) - self.rhs
            ) ** 2 + self.regularizer * np.linalg.norm(self.data, ord=1)
        return self.data, residual

    @staticmethod
    def _soft_threshold(data: NDArray, threshold: float) -> NDArray:
        """Threshold data."""
        return np.sign(data) * np.maximum(np.abs(data) - threshold, 0)

    def _normal_matrix(self, input_vector: NDArray) -> NDArray:
        """Combine A_transpose * A."""
        return self.operator_transposed_fun(self.operator_fun(input_vector))

    def _power_iteration(self, max_iter: int = 10) -> float:
        """Compute largest singular value."""
        if self.seed is not None:
            np.random.seed(self.seed)
        eigen_vector = np.random.rand(self.operator_size[1])

        for _ in range(max_iter):
            eigen_vector_1 = self._normal_matrix(eigen_vector)
            # calculate norm
            eigen_vector_1_norm = np.linalg.norm(eigen_vector_1)
            # re-normalize vector
            eigen_vector = eigen_vector_1 / eigen_vector_1_norm

        largest_eigen_value = np.linalg.norm(self.operator_fun(eigen_vector))
        return largest_eigen_value


# class DCA:
#     """Convex-Concave optimization."""
#
#     def __init__(self):
#         """Create object."""
#         # convergence criteria
#         self.eps_1 = 3e-3
#         self.eps_2 = 1e-3
#
#     def run(self):
#         """Run the algorithm."""
#         mu = 0
#         e = np.ones((self.operator.shape[1],))
#         x = e / 2
#
#         lambda_weight = power_method(Q)
#         mu_delta = 1e-3 * lambda_weight
#
#         counter = 0
#         outer_counter = 0
#
#         while np.max(np.minimum(x, 1 - x)) > self.eps_2:
#             do = True
#             inner_counter = 0
#             while do or norm(x - x_old) / norm(x_old) > eps_1 and inner_count < 1e3:
#                 counter += 1
#                 x_old = x
#                 grad = (
#                     (lambda_weight + mu) - Q.matvec(x)
#                     + operator.T * rhs - 0.5 * mu * e
#                 )
#                 x[grad < 0] = 0
#                 x[grad >= lambda_weight] = 1
#                 x[0 < grad <= lambda_weight] = (
#                     grad[0 < grad <= lambda_weight] / lambda_weight
#                 )
#
#                 do = False
#                 inner_counter += 1
#             x_old = e
#             mu = mu + mu_delta
#             outer_counter += 1
