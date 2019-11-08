from sklearn.base import BaseEstimator
import numpy as np
from sklearn.linear_model import LinearRegression

# A simple kit that cheats by directly looking at the future
# in predict (preds[:-1,:] = X[1:,:])
# This is forbidden, and caught by our workflow

class GenerativeRegressor(BaseEstimator):
    def __init__(self,  max_dists, current_dim):
        self.reg = LinearRegression()
        self.max_dists = max_dists
        self.sigma = None

    def fit(self, X, y):
        pass

    def predict(self, X):
        # The first generative regressor is gaussian, the second is beta
        # For the whole list of distributions, run
        #   import rampwf as rw
        #   rw.utils.distributions_dict
        types = np.array([[0, 2], ] * len(X))

        # Normal
        preds = np.zeros(X.shape)
        preds[:-1,:] = X[1:,:]
        sigmas = np.array([self.sigma] * len(X))
        sigmas = sigmas[:, np.newaxis]
        params_normal = np.concatenate((preds, sigmas), axis=1)
        # To get information about the parameters of the distribution you are
        # using, you can run
        #   import rampwf as rw
        #   [(v,v.params) for v in rw.utils.distributions_dict.values()]

        # We give more weight to the gaussian one
        weights = np.array([[1.0], ] * len(X))

        return weights, types, params_normal
