from typing import Union

import numpy as np
from scipy.stats import norm

from acquisition_functions.abstract_acquisition_function import AcquisitionFunction
from gaussian_process import GaussianProcess


class ExpectedImprovement(AcquisitionFunction):
    def _evaluate(self,
                  gaussian_process: GaussianProcess,
                  data_points: np.ndarray
                  ) -> np.ndarray:
        """
        Evaluates the acquisition function at all the data points
        :param gaussian_process:
        :param data_points: numpy array of dimension n x m where n is the number of elements to evaluate
        and m is the number of variables used to calculate the objective function
        :return: a numpy array of shape n x 1 (or a float) representing the estimation of the acquisition function at
        each point
        """

        # TODO

        mean_data_points, std_data_points = gaussian_process.get_gp_mean_std(data_points.reshape((-1, gaussian_process.array_dataset.shape[1])))

        mean = mean_data_points

        mean = mean.flatten()
        mean_opt = np.min(gaussian_process.array_objective_function_values)

        difference = mean_opt - mean
        Z = difference/std_data_points

        ei = difference*norm.cdf(Z) + std_data_points*norm.pdf(Z)
        ei[std_data_points == 0.0] = 0.0

        return ei
