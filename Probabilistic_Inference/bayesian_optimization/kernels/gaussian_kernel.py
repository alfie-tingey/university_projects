import numpy as np

from kernels.abstract_kernel import Kernel

class GaussianKernel(Kernel):
    def __init__(self,
                 log_amplitude: float,
                 log_length_scale: float,
                 log_noise_scale: float,
                 ):
        super(GaussianKernel, self).__init__(log_amplitude,
                                             log_length_scale,
                                             log_noise_scale,
                                             )
        self.log_amplitude = log_amplitude
        self.log_length_scale = log_length_scale
        self.log_noise_scale = log_noise_scale

    def get_covariance_matrix(self,
                              X: np.ndarray,
                              Y: np.ndarray,
                              ) -> np.ndarray:
        """
        :param X: numpy array of size n_1 x l for which each row (x_i) is a data point at which the objective function can be evaluated
        :param Y: numpy array of size n_2 x m for which each row (y_j) is a data point at which the objective function can be evaluated
        :return: numpy array of size n_1 x n_2 for which the value at position (i, j) corresponds to the value of
        k(x_i, y_j), where k represents the kernel used.
        """
        # TODO

        # The dimensions l and m have to be the same
        # load the variables in from the init that we need to use

        amplitude = np.exp(self.log_amplitude)
        length_scale = np.exp(self.log_length_scale)

        # Initialise the gaussian covariance matrix with zeros

        gauss_kernel_matrix = np.zeros(shape = (X.shape[0],Y.shape[0]))

        # Use the formula to input values into gaussian covariance matrix
        for i in range(int(X.shape[0])):
            for j in range(int(Y.shape[0])):
                gauss_kernel_matrix[i][j] = (amplitude**2)*np.exp((-1/(2*length_scale**2))*(np.linalg.norm(X[i,] - Y[j,]))**2)

        #print(gauss_kernel_matrix)

        return gauss_kernel_matrix


    def __call__(self,
                 X: np.ndarray,
                 Y: np.ndarray,
                 ) -> np.ndarray:
        return self.get_covariance_matrix(X, Y)
