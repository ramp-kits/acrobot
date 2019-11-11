from sklearn.base import BaseEstimator
import numpy as np
from sklearn.linear_model import LinearRegression


class GenerativeRegressor(BaseEstimator):
    def __init__(self, max_dists, target_dim):
        """
        Parameters
        ----------
        max_dists : int
            The maximum number of distributions (kernels) in the mixture.
        target_dim : int
            The index of the target column to be predicted.
        """
        pass

    def fit(self, X, y):
        """Linear regression + residual sigma."""
        self.reg = LinearRegression()
        self.reg.fit(X, y)
        y_pred = self.reg.predict(X)
        y_pred = np.array([y_pred]).reshape(-1, 1)
        residuals = y - y_pred
        # Estimate a single sigma from residual variance
        self.sigma = np.sqrt((1 / (X.shape[0] - 1)) * np.sum(residuals ** 2))

    def predict(self, X):
        """Construct a conditional mixture distribution.
        Return
        ------
        weights : np.array of float
            discrete probabilities of each component of the mixture
        types : np.array of int
            integer codes referring to component types
            see rampwf.utils.distributions_dict
        params : np.array of float tuples
            parameters for each component in the mixture
        """
        types = np.array([[0], ] * len(X))

        # Normal
        y_pred = self.reg.predict(X)
        sigmas = np.array([self.sigma] * len(X))
        sigmas = sigmas[:, np.newaxis]
        params = np.concatenate((y_pred, sigmas), axis=1)
        weights = np.array([[1.0], ] * len(X))
        return weights, types, params