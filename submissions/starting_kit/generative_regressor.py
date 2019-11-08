from sklearn.base import BaseEstimator
import numpy as np
from sklearn.linear_model import LinearRegression


class GenerativeRegressor(BaseEstimator):
    def __init__(self,  max_dists, current_dim):
        self.reg = LinearRegression()
        self.max_dists = max_dists
        self.sigma = None
        self.a = None
        self.b = None

    def fit(self, X, y):
        self.reg.fit(X, y)
        yGuess = self.reg.predict(X)
        yGuess = np.array([yGuess]).reshape(-1, 1)
        error = y - yGuess
        self.sigma = np.sqrt((1 / X.shape[0]) * np.sum(error ** 2))
        self.loc = np.min(y)
        self.max = np.max(y)

    def predict(self, X):
        # The first generative regressor is gaussian, the second is beta
        # For the whole list of distributions, run
        #   import rampwf as rw
        #   rw.utils.distributions_dict
        types = np.array([[0, 2], ] * len(X))

        # Normal
        preds = self.reg.predict(X)
        sigmas = np.array([self.sigma] * len(X))
        sigmas = sigmas[:, np.newaxis]
        params_normal = np.concatenate((preds, sigmas), axis=1)

        # Dumb example of usage of Beta distribution
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html
        padding = 2
        a_array = np.array([6] * len(X))
        b_array = np.array([7] * len(X))
        loc_array = np.array([self.loc - padding] * len(X))
        scale_array = np.array([self.max-self.loc + 2 * padding] * len(X))
        params_beta = np.stack(
            (a_array, b_array, loc_array, scale_array), axis=1)

        # We give more weight to the gaussian one
        weights = np.array([[0.6, 0.4], ] * len(X))

        # We concatenate the params
        # To get information about the parameters of the distribution you are
        # using, you can run
        #   import rampwf as rw
        #   [(v,v.params) for v in rw.utils.distributions_dict.values()]
        params = np.concatenate((params_normal, params_beta), axis=1)
        return weights, types, params
