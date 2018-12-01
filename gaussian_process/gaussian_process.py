import numpy as np
import means
import kernels


class gaussian_process():
    def __init__(self, mean=means.zero, kernel=kernels.linear):
        self._mean = mean
        self._kernel = kernel

        self._covariance = None
        self._mu = None
        self._decomposition = None

    def set(self, S: np.ndarray) -> None:
        """ Create GP from the set S. """
        self._covariance = np.zeros((len(S), len(S)))
        self._mu = np.zeros(len(S))
        """ Build distribution """
        for xi, si in enumerate(S):
            for xj, sj in enumerate(S):
                self._covariance[xi][xj] = self._kernel(si, sj)

            self._mu[xi] = self._mean(si)
        """ Properties of Affinity transformations of Gaussians lets 
        us sample using Eigenvalue Decomposition """
        eigval, eigvec = np.linalg.eig(self._covariance)
        self._decomposition = eigvec @ np.diag(np.sqrt(eigval))

    def sample(self) -> np.ndarray:
        """ Return a sample from GP. """
        iinrv = np.random.normal(
            np.zeros_like(self._mu, dtype=np.float64),
            np.ones_like(self._mu, dtype=np.float64))

        return (self._decomposition @ iinrv) + self._mu
