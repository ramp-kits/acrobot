from sklearn.base import BaseEstimator
import numpy as np
from sklearn.linear_model import LinearRegression


class GenerativeRegressor(BaseEstimator):
    def __init__(self,  max_dists, current_dim):
        self.reg = LinearRegression()
        self.max_dists = max_dists
        self.sigma = None

    def fit(self, X, y):
        pass

    def predict(self, X):
        # The first generative regressor is gaussian, the second is beta
        types = np.array([[0, 2], ] * len(X))

        # Normal
        preds = np.zeros(X.shape)
        preds[:-1,:] = X[1:,:]
        sigmas = np.array([self.sigma] * len(X))
        sigmas = sigmas[:, np.newaxis]
        params_normal = np.concatenate((preds, sigmas), axis=1)

        # We give more weight to the gaussian one
        weights = np.array([[1.0], ] * len(X))

        return weights, types, params_normal
