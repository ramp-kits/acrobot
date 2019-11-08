from sklearn.base import BaseEstimator
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import KBinsDiscretizer

np.random.seed(7)
EPSILON = 1e-10

# This kit will create a sort of discretized distribution, by using
# non overlapping uniform distribution

class GenerativeRegressor(BaseEstimator):
    def __init__(self, max_dists, current_dim):
        self.nb_bins = max_dists
        self.enc = KBinsDiscretizer(n_bins=self.nb_bins, encode='ordinal')
        self.clf = RandomForestClassifier(
            n_estimators=40, max_leaf_nodes=150, random_state=61)

    def fit(self, X, y):
        self.enc.fit(y)
        y = self.enc.transform(y)
        self.clf.fit(X, y.ravel())

    def predict(self, X):

        # Only uniform distributions
        # For the whole list of distributions, run
        #   import rampwf as rw
        #   rw.utils.distributions_dict
        types = np.ones((len(X), self.nb_bins))

        preds_proba = self.clf.predict_proba(X)
        weights = np.array(preds_proba)

        bins = self.enc.bin_edges_[0].copy()
        # Otherwise calling the model will modify the bins when padding

        a_array = bins[:-1]
        b_array = bins[1:]

        # We make sure no value falls outside our coverage
        padding = 2
        a_array[0] -= padding
        b_array[-1] += padding
        weights += EPSILON
        weights /= np.sum(weights, axis=1)[:, None]

        # To get information about the parameters of the distribution you are
        # using, you can run
        #   import rampwf as rw
        #   [(v,v.params) for v in rw.utils.distributions_dict.values()]
        params_uniform = np.empty((len(X), self.nb_bins * 2))
        params_uniform[:, 0::2] = a_array
        params_uniform[:, 1::2] = b_array

        return weights, types, params_uniform
