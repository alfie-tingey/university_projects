import numpy as np
import scipy
from scipy.stats import norm, multivariate_normal


from distribution_prediction.metropolis_hastings.utils_plots import plot_metropolis_hastings_logistics
from distribution_prediction.utils import sigmoid

def get_log_upper_proba_distribution(X: np.ndarray,
                                     y: np.ndarray,
                                     theta: np.ndarray,
                                     sigma_prior: float
                                     ) -> float:
    """
    This functions evaluates log( p_1(theta | X, y) ) where:
     - p_1 = Z * p
     - p is the posterior distribution
     - p_1 is easy to calculate

    You may use the sigmoid function in the file utils.py
    BE CAREFUL: be sure to reshape theta before passing it as an argument to the function sigma!!

    :param X: data points of shape (N, 2) where N is the number of data points, and 2 is the number of components for
    each data point x. Each row of X represents one data point.
    :param y: column vector of shape (N, 1) indicating the class of p. for each point, y_i = 0 or 1.
    In addition, y_i = 1 is equivalent to "x_i is in C_1"
    :param theta: parameters at which we evaluate p_1. In our example, it is a numpy array (row vector) of shape (2,).
    :param sigma_prior: standard deviation of the prior on the parameters
    :return: log( p_1(theta | X, y) )
    """
    # TODO

    theta = theta.reshape((1,2))

    f = sigmoid(X,theta)

    size = X.shape[1]

    likelihood = np.sum(y*np.log(f) + (1-y)*np.log(1 - f))

    mean_prior = np.zeros(size)

    covariance_matrix = (sigma_prior**2)*np.identity(size)

    prior = scipy.stats.multivariate_normal(mean_prior.flatten(),covariance_matrix).pdf(theta)

    log_p1 = np.sum(likelihood + np.log(prior))

    return log_p1

def metropolis_hastings(X: np.ndarray,
                        y: np.ndarray,
                        number_expected_samples: int,
                        sigma_exploration_mh: float=1,
                        sigma_prior: float=1):
    """
    Performs a Metropolis Hastings procedure.
    This function is a generator. After each step, it should yield a tuple containing the following elements
    (in this order):
    -  is_sample_accepted (type: bool) which indicates if the last sample from the proposal density has been accepted
    -  np.array(list_samples): numpy array of size (S, 2) where S represents the total number of previously accepted
    samples, and 2 is the number of components in theta in this logistic regression task.
    -  newly_sampled_theta: in this example, numpy array of size (2,)
    -  u (type: float): last random number used for deciding if the newly_sampled_theta should be accepted or not.


    :param X: data points of shape (N, 2) where N is the number of data points. There is one data point per row
    :param y: column vector of shape (N, 1) indicating the class of p. for each point, y_i = 0 or 1.
    In addition, y_i = 1 is equivalent to "x_i is in C_1"
    :param number_expected_samples: Number of samples expected from the Metropolis Hastings procedure
    :param sigma_exploration_mh: Standard deviation of the proposal density.
    We consider that the proposal density corresponds to a multivariate normal distribution, with:
    - mean = null vector
    - covariance matrix = (sigma_proposal_density ** 2) identity matrix
    :param sigma_prior: standard deviation of the prior on the parameters
    """


    # ----- These are some the variables you should manipulate in the main loop of that function ----------
    list_samples = []  # Every newly_sampled_theta  which is accepted should be added to the list of samples

    newly_sampled_theta = None  # Last sampled parameters (from the proposal density q)

    is_sample_accepted = False  # Should be True if and only if the last sample has been accepted

    u = np.random.uniform(0,1)  # Random number used for deciding if newly_sampled_theta should be accepted or not

    # For first iteration do this

    first_theta = np.zeros(X.shape[1])

    # -------------------------------------------------------------------------------------------------

    while len(list_samples) < number_expected_samples:
        #########################

        # uniform sample for u

        u = np.random.uniform(0,1)

        # get the covariance matrix for multivariate random norm

        covariance_matrix = np.identity(X.shape[1])*sigma_exploration_mh**2

        # get the newly sampled theta from the first theta

        newly_sampled_theta = np.random.multivariate_normal(first_theta, covariance_matrix)

        # get a likelihood / log upper proba distribution from first theta

        theta_likelihood = get_log_upper_proba_distribution(X,y,first_theta, sigma_prior)

        # get same thing from newly sampled theta

        theta_new_likelihood = get_log_upper_proba_distribution(X,y,newly_sampled_theta,sigma_prior)

        # get density proposal from first theta and newly sampled theta

        proposal_density_log_new = np.log(multivariate_normal.pdf(first_theta, newly_sampled_theta, covariance_matrix))

        proposal_density_log_old = np.log(multivariate_normal.pdf(newly_sampled_theta, first_theta, covariance_matrix))

        # perform the metropolis hastings algorithm

        if u <= np.exp(theta_new_likelihood + proposal_density_log_new - theta_likelihood - proposal_density_log_old):
            is_sample_accepted = True
        else:
            is_sample_accepted = False

        if is_sample_accepted:
            first_theta = newly_sampled_theta
            list_samples.append(newly_sampled_theta)

        #########################


        yield is_sample_accepted, np.array(list_samples), newly_sampled_theta, u


def get_predictions(X_star: np.ndarray,
                    array_samples_theta: np.ndarray
                    ) -> np.ndarray:
    """
    :param X_star: array of data points of shape (N, 2) where N is the number of data points.
    There is one data point per row
    :param array_samples_theta: np array of shape (M, 2) where M is the number of sampled set of parameters
    generated by the Metropolis-Hastings procedure. Each row corresponds to one sampled theta.
    :return: estimated predictions at each point in X_star: p(C_1|X,y,x_star)=p(y_star=1|X,y,x_star),
    where each x_star corresponds to a row in X_star. The result should be a column vector of shape (N, 1), its i'th
    row should be equal to the prediction p(C_1|X,y,x_star_i) where x_star_i corresponds to the i'th row in X_star
    """

    # TODO

    m = array_samples_theta.shape[0]

    f = sigmoid(X_star, array_samples_theta)

    g = []

    for i in range(X_star.shape[0]):
        g.append(np.mean(f[i]))

    g = np.array(g).T

    return g


if __name__ == '__main__':
    plot_metropolis_hastings_logistics(num_samples=1000,
                                       interactive=True,
                                       sigma_exploration_mh=1,
                                       sigma_prior=1,
                                       number_points_per_class=25)
