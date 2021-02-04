import scipy
import numpy as np
import matplotlib.pyplot as plt
from .util import RealtimeImager


def power_method(A_matrix, max_iter=10):
    """Compute the largest singular value of A."""
    x_vec = np.random.random((A_matrix.shape[1],))
    y_vec = A_matrix * x_vec

    for iter_count in range(max_iter):
        x_vec = A_matrix.rmatvec(y_vec)
        x_vec = x_vec / np.linalg.norm(x_vec)
        y_vec = A_matrix * x_vec
        L = np.linalg.norm(y_vec)
        print("%d singular value: %f" % (iter_count, L))
    return L


class ChambollePock(object):
    """Chambolle-Pock algorithm for total-variation minimization."""


    def __init__(self, A, TV, rhs, x_0=None, max_iter=20, tv_weight=10,
                 nonnegative=False, show=False,
                 a_scale=None, tv_scale=None, lipschitz_factor=None):
        if isinstance(A, np.ndarray):
            A = scipy.sparse.linalg.aslinearoperator(A)
        if isinstance(TV, np.ndarray):
            TV = scipy.sparse.linalg.aslinearoperator(TV)

        # save properties
        self.A = A
        self.TV = TV
        self.rhs = rhs
        self.x_0 = x_0
        self.max_iter = max_iter
        self.tv_weight = tv_weight
        self.nonnegative = nonnegative
        self.show = show
        self.a_scale = a_scale
        self.tv_scale = tv_scale
        self.lipschitz_factor = lipschitz_factor

    def run(self):
        """Run the algorithm."""
        print("Determine A-scaling")
        if self.a_scale is None:
            L_A = power_method(self.A, 15)
        else:
            L_A = self.a_scale

        print("Determine TV-scaling")
        if self.tv_scale is None:
            L_TV = power_method(self.TV, 15)
        else:
            L_TV = self.tv_scale

        # scale operators
        A = (1.0 / L_A) * self.A
        rhs = self.rhs / L_A
        TV = (1.0 / L_TV) * self.TV

        tv_weight = L_TV / (L_A ** 2) * self.tv_weight

        print("Determine Lipschitz-constant")
        if self.lipschitz_factor is None:
            # L = self.tvmin_power_method(A, TV, 15)
            # assume this is 1 after normalization
            L = 1
        else:
            L = self.lipschitz_factor

        if self.x_0 is None:
            x_0 = np.zeros((A.shape[1], ))
        else:
            x_0 = self.x_0

        u_vec = x_0
        p_vec = np.zeros((len(rhs), ))
        q_vec = np.zeros((TV.shape[0], ))

        x_len = len(x_0)

        # scaling parameters
        tau = 1 / L
        sigma = 1 / L
        theta = 1

        norm_b_0 = np.linalg.norm(rhs)
        u_bar_vec = u_vec

        residual_vec = np.zeros((self.max_iter, ))
        relres_vec = np.zeros((self.max_iter, ))
        objective_vec = np.zeros((self.max_iter, ))
        dual_gap_vec = np.zeros((self.max_iter, ))

        A_u_bar = A.matvec(u_bar_vec)
        TV_u_bar = TV.matvec(u_bar_vec)

        increasing_objective_count = 0

        if self.show:
            print("Iter\tobjective\tprimal-dual gap\tresidual-norm^2\tTV-norm")
            print("====================================================================")
            imager = RealtimeImager(self.get_slice(u_bar_vec))

        for iter_count in range(self.max_iter):
            p_vec = (p_vec + sigma * (A_u_bar - rhs)) / (1 + sigma)
            q_vec = q_vec + sigma * TV_u_bar
            q_mat = q_vec.reshape((x_len, -1))
            magnitude = np.linalg.norm(q_mat, axis=1)
            magnitude = np.maximum(magnitude, tv_weight)
            q_mat = np.divide(tv_weight * q_mat, np.tile(magnitude, (q_mat.shape[1], 1)).T)
            q_vec = q_mat.flatten()

            if self.nonnegative:
                u_new_vec = np.maximum(u_vec - tau * A.rmatvec(p_vec) - tau * TV.rmatvec(q_vec), 0)
            else:
                u_new_vec = u_vec - tau * A.rmatvec(p_vec)
                u_new_vec = u_new_vec - tau * TV.rmatvec(q_vec)

            u_bar_vec = u_new_vec + theta * (u_new_vec - u_vec)
            u_vec = u_new_vec
            A_u_bar = A.matvec(u_bar_vec)
            TV_u_bar = TV.matvec(u_bar_vec)
            residual_vec[iter_count] = np.linalg.norm(A_u_bar - rhs)
            relres_vec[iter_count] = residual_vec[iter_count] / norm_b_0
            objective_vec[iter_count] = 0.5 * np.linalg.norm(A_u_bar - rhs) ** 2 \
                + tv_weight * np.linalg.norm(TV_u_bar, ord=1)
            dual_gap_vec[iter_count] = 0.5 * np.linalg.norm(A_u_bar - rhs) ** 2 \
                + tv_weight * np.linalg.norm(TV_u_bar, ord=1) \
                + 0.5 * np.linalg.norm(p_vec) ** 2 - np.dot(p_vec, rhs)

            if self.show:
                print('%d\t%e\t%e\t%e\t%e' % (
                    iter_count, objective_vec[iter_count], dual_gap_vec[iter_count],
                    residual_vec[iter_count], np.linalg.norm(TV_u_bar, ord=1)))
                imager.update(self.get_slice(u_bar_vec))

            if iter_count > 0 and (objective_vec[iter_count] > objective_vec[iter_count - 1]):
                increasing_objective_count += 1
                if increasing_objective_count >= 5:
                    x_vec = u_bar_vec
                    return u_bar_vec
                    if self.show:
                        print("Converged after %d iterations" %  iter_count)
        x_vec = u_bar_vec
        return x_vec

    def get_slice(self, u_vec):
        """Get slice of volume."""
        middle_slice_index = int(self.A.vshape[0] / 2)
        if len(self.A.vshape) == 3:
            im_slice = u_vec.reshape(self.A.vshape)[
                middle_slice_index, :, :]
        elif len(self.A.vshape) == 2:
            im_slice = u_vec.reshape(self.A.vshape)
        return im_slice

    @staticmethod
    def tvmin_power_method(W, D, max_iter=10):
        """Compute the largest singular value of [W, D]."""
        x_vec = np.random.random((W.shape[1],))
        y_vec = W.matvec(x_vec)
        z_vec = D.matvec(x_vec)
    
        for iter_count in range(max_iter):
            # power iteration
            x_vec = W.rmatvec(y_vec) + D.rmatvec(z_vec)
            # normalize
            x_vec = x_vec / np.linalg.norm(x_vec)
            y_vec = W.matvec(x_vec)
            z_vec = D.matvec(x_vec)
            L = np.sqrt(np.dot(y_vec, y_vec) + np.dot(z_vec, z_vec))
            print("%d) singular value: %f" % (iter_count, L))
        return L
   

class Fista(object):  # pylint: disable=too-many-instance-attributes,too-few-public-methods
    """Implementation of FISTA l1-optimization.

    Based on the paper:
    [1] Amir Beck, Marc Teboulle, "A Fast Iterative Shrinkage-Thresholding Algorithm
        for Linear Inverse Problems", SIAM J. Imaging sciences, 2(1), pp. 183-202, 2009.
        https://people.rennes.inria.fr/Cedric.Herzet/Cedric.Herzet/Sparse_Seminar/Entrees/2012/11/12_A_Fast_Iterative_Shrinkage-Thresholding_Algorithmfor_Linear_Inverse_Problems_(A._Beck,_M._Teboulle)_files/Breck_2009.pdf
    """

    def __init__(self, operator, rhs, input_data,  # pylint:disable=too-many-arguments
                 regularizer=0.5, max_iter=10, operator_transposed=None, seed=None):  # pylint:disable=too-many-arguments
        """Create FISTA instance."""
        # check whether "operator" is a matrix or function handle
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

    def initialize(self, initial_vector):
        """Initialize method."""
        self.data = initial_vector.copy()

    @staticmethod
    def _soft_threshold(data, threshold):
        """Threshold data."""
        return np.sign(data) * np.maximum(np.abs(data) - threshold, 0)

    def _normal_matrix(self, input_vector):
        """Combine A_transpose * A."""
        return self.operator_transposed_fun(self.operator_fun(input_vector))

    def _power_iteration(self, max_iter=10):
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

    def run(self):
        """Run fista method."""
        residual = np.zeros((self.max_iter,))
        threshold = 1
        update = self.data.copy()

        for iteration in range(self.max_iter):
            data_copy = self.data.copy()
            update = update + self.operator_transposed_fun(self.rhs - self.operator_fun(update)) / self.normalization
            self.data = self._soft_threshold(update, self.regularizer / self.normalization)
            threshold_copy = threshold
            threshold = (1 + np.sqrt(1 + 4 * threshold**2)) / 2
            update = self.data + ((threshold_copy - 1) / threshold) * (self.data - data_copy)
            residual[iteration] = 0.5 * np.linalg.norm(self.operator_fun(self.data) - self.rhs) ** 2 + \
                self.regularizer * np.linalg.norm(self.data, ord=1)
        return self.data, residual


class DCA(object):
    """Convex-Concave optimization."""

    def __init__(self):
        """Create object."""
        # convergence criteria
        self.eps_1 = 3e-3
        self.eps_2 = 1e-3

    def run(self):
        """Run the algorithm."""
        mu = 0
        e = np.ones((self.operator.shape[1], ))
        x = e / 2

        lambda_weight = self.power_method(Q)
        mu_delta = 1e-3 * lambda_weight

        counter = 0
        outer_counter = 0

        while np.max(np.minimum(x, 1 - x)) > self.eps_2:
            do = True
            inner_counter = 0
            while do or norm(x - x_old)/norm(x_old) > eps_1 and inner_count < 1e3:
                counter += 1
                x_old = x
                grad = (lambda_weight + mu) - Q.matvec(x) + operator.T * rhs - 0.5 * mu * e
                x[grad < 0] = 0
                x[grad >= lambda_weight] = 1
                x[0 < grad <= lambda_weight] = grad[0 < grad <= lambda_weight] / lambda_weight

                do = false
                inner_counter += 1
            x_old = e
            mu = mu + mu_delta
            outer_counter += 1

