import numpy as np
import means
import kernels
import scipy


class gaussian_process():
    def __init__(self,
                 mean=means.zero,
                 kernel=kernels.linear,
                 mu: np.ndarray = None,
                 covariance: np.ndarray = None,
                 X: np.ndarray = None,
                 Y: np.ndarray = None):
        self._mean = mean
        self._kernel = kernel

        self._mu = mu
        self._covariance = covariance
        self._X = X
        self._Y = Y

    def set(self, S: np.ndarray, V: np.ndarray) -> None:
        if np.ndim(S) == 0:
            S = np.array([[S]])
        if np.ndim(S) == 1:
            S = np.array([[s] for s in S])

        if np.ndim(V) == 1:
            V = V.reshape(-1, 1)
        """ 
        S: Samples from the Domain
        V: Values in the Codomain
        """
        self._covariance = np.zeros((len(S), len(S)))
        self._mu = np.zeros(len(S))
        """ Build distribution """
        for xi, si in enumerate(S):
            for xj, sj in enumerate(S):
                self._covariance[xi][xj] = self._kernel(si, sj)

            self._mu[xi] = self._mean(si)

        self._X = S
        self._Y = V

    def sample(self) -> np.ndarray:
        """ Return a sample from GP. """
        return np.random.multivariate_normal(self._mu, self._covariance)

    def posterior(self, S: np.ndarray) -> (np.ndarray, np.ndarray):
        if np.ndim(S) == 0:
            S = np.array([[S]])
        if np.ndim(S) == 1:
            S = np.array([[s] for s in S])

        S_cov = np.zeros((len(S), len(self._X)), dtype=np.float64)
        for xi, si in enumerate(S):
            for xj, sj in enumerate(self._X):
                S_cov[xi][xj] = self._kernel(si, sj)
        """ Bottle Neck of All Things using Gaussian Inversion. """
        inverse = np.linalg.inv(
            self._covariance + np.identity(len(self._covariance)) * np.random.
            normal(size=len(self._covariance)) * 1e-3)

        samples = len(S)
        mu = S_cov @ inverse @ self._Y

        indep_variance = np.array([self._kernel(x, x) for x in S])
        samples, domain = S_cov.shape

        variance = []
        for indep_var, var in zip(indep_variance, S_cov):
            variance.append(
                (indep_var - var.reshape(1, -1) @ inverse @ var.reshape(-1, 1)
                 ).reshape(-1)[0])

        variance = np.abs(np.array(variance))

        return mu.reshape(-1), np.sqrt(variance.reshape(-1))
