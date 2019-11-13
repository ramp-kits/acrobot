from sklearn.base import BaseEstimator
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from rampwf.hyperopt import Hyperparameter
 
 
# RAMP START HYPERPARAMETERS
#sigma_multiplier = Hyperparameter(
#    dtype='float', default=1.3, values=[
#        1.3, 1.5, 1.7])
#n_estimators = Hyperparameter(
#    dtype='int', default=200, values=[30, 50, 100, 200, 300])
#max_leaf_nodes = Hyperparameter(
#    dtype='int', default=5000, values=[1000, 2000, 5000, 10000])
# RAMP END HYPERPARAMETERS                                            
 
method_proba = 'estimate_sigma'
EPSILON =1e-8
 
 
class GenerativeRegressor(BaseEstimator):
    def __init__(self, max_dists, model_index):
        self.max_dists = min(100, max_dists) 
        self.clf = BaggingRegressor(
            base_estimator=DecisionTreeRegressor(
                max_leaf_nodes=5000),
            n_estimators=self.max_dists-1, # The last dist is uniform
            max_samples=0.2,
        )
        self.sigma = None
        self.a = None
        self.b = None
 
    def fit(self, X, y):
        self.clf.fit(X, y.ravel())
        yGuess = self.clf.predict(X)
        yGuess = np.array([yGuess]).reshape(-1, 1)
        error = y - yGuess
        self.sigma = np.sqrt((1 / X.shape[0]) * np.sum(error ** 2))
        self.a = np.min(y) - 10
        self.b = np.max(y) + 10
 
    def predict(self, X):
 
        # We give every distribution the same weight
        eps = 10 ** -10
        w = (1.0 - EPSILON)/(self.max_dists - 1) 
        weights = np.stack([[w] * len(X)
                            for _ in range(self.max_dists)],
                           axis=1)
        weights[:, 0] = EPSILON
        # The first generative regressors are gaussian, the last is uniform
        types = np.zeros(self.max_dists)
        types[-1] = 1
        types = np.array([types] * len(X))
 
        # Gaussians
        mus = np.zeros((len(X), len(self.clf.estimators_),))
        for i, est in enumerate(self.clf.estimators_):
            mus[:, i] = est.predict(X)
 
        if method_proba == 'estimate_sigma':
            # Third method, uses the information from the trees to estimate
            # sigma and estimate gaussian noise around outpus
            sigma = mus.std(axis=1)
            sigma = np.clip(sigma, EPSILON, None, out=sigma)
            sigmas = np.stack([sigma for _ in range(len(self.clf.estimators_))],
                              axis=1)
 
        elif method_proba == 'standard':
            sigmas = np.stack([[self.sigma] * len(X)
                               for _ in range(len(self.clf.estimators_))],
                              axis=1)
        else:
            sigmas = np.nan
 
        #sigmas *= float(sigma_multiplier)
        sigmas /= float(np.sqrt(self.max_dists - 1))
 
        # We put each mu next to its sigma
        params_normal = np.empty((len(X), len(self.clf.estimators_)*2))
        params_normal[:, 0::2] = mus
        params_normal[:, 1::2] = sigmas
 
        # Uniform
        a_array = np.array([self.a] * len(X))
        b_array = np.array([self.b] * len(X))
        params_uniform = np.stack((a_array, b_array), axis=1)
 
        # We concatenate the params
        params = np.concatenate((params_normal, params_uniform), axis=1)
        return weights, types, params
