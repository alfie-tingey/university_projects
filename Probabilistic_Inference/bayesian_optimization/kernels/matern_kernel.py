import numpy as np

from kernels.abstract_kernel import Kernel


class MaternKernel(Kernel):
    def get_covariance_matrix(self, X: np.ndarray, Y: np.ndarray):
        """
        :param X: numpy array of size n_1 x l for which each row (x_i) is a data point at which the objective function can be evaluated
        :param Y: numpy array of size n_2 x m for which each row (y_j) is a data point at which the objective function can be evaluated
        :return: numpy array of size n_1 x n_2 for which the value at position (i, j) corresponds to the value of
        k(x_i, y_j), where k represents the kernel used.
        """
        # TODO
        amplitude = np.exp(self.log_amplitude)
        length_scale = np.exp(self.log_length_scale)

        # Initialise the gaussian covariance matrix with zeros

        matern_kernel_matrix = np.zeros(shape = (X.shape[0],Y.shape[0]))

        # Use the formula to input values into gaussian covariance matrix
        for i in range(int(X.shape[0])):
            for j in range(int(Y.shape[0])):
                matern_kernel_matrix[i][j] = (amplitude**2)*(1+(np.sqrt(3)*np.linalg.norm(X[i,] - Y[j,]))/length_scale)*np.exp(-(np.sqrt(3)*np.linalg.norm(X[i,] - Y[j,]))/length_scale)

        #print(gauss_kernel_matrix)

        return matern_kernel_matrix
