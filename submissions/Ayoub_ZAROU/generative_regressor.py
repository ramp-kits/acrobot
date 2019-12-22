# Author(s): Balazs Kegl <balazs.kegl@gmail.com>
# License: BSD 3 clause

import numpy as np
import pandas as pd
from sklearn.svm import SVR, SVC
from sklearn.ensemble import (RandomForestRegressor,
                              GradientBoostingRegressor, ExtraTreesRegressor)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import KFold, LeavePOut, GridSearchCV
from sklearn.base import RegressorMixin, BaseEstimator, clone
from lightgbm import LGBMRegressor
 
lgb_params = {
    'max_depth': [-1, 7],
    'n_estimators': [100, 50],
    'num_leaves': [30],
    'reg_alpha': [0, 0.5],
    'reg_lambda': [2, 5],
}
svc_params = {'C': [1.0, 2, 4, 6],
              'gamma': ['scale', 'auto'],
              }
 
lgb_regressor = GridSearchCV(LGBMRegressor(), lgb_params, cv=3, iid=False)
svc_classifier = GridSearchCV(SVC(probability=True), svc_params, cv=3, iid=False)
 
 
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
        self.target_dim = target_dim
 
    def create_regressor(self):
        return (
            StackedEstimator().add_first_level_regressors(RandomForestRegressor(n_estimators=100, max_depth=4),
                                                          'random_forrest')
                .add_first_level_regressors(GaussianProcessRegressor(), 'gaussian')
                .add_first_level_regressors(SVR(gamma='scale', C=3), 'svr')
                .add_first_level_regressors(ExtraTreesRegressor(n_estimators=100), 'extra_tree')
                .add_second_level_regressors(GradientBoostingRegressor(), 'gbr')
                .add_second_level_regressors(clone(lgb_regressor), 'lgbm')
                .add_regressor_selector(clone(svc_classifier))
        )
 
    def fit(self, X_array, y_array):
        """Linear regression + residual sigma.
        Parameters
        ----------
        X_array : pandas.DataFrame
            The input array. The features extracted by the feature extractor,
            plus `target_dim` system observables from time step t+1.
        y_array :
            The ground truth array (system observables at time step t+1).
        """
 
        if self.target_dim in [1, 3]:
            errors = []
            models = []
            sigmas = []
            thresholds = [4, 3, 5, 2]
            for thres in thresholds:
                reg = self.create_regressor()
                y_copy = self.change_target(X_array, y_array, threshold=thres).reshape(-1, 1)
                reg.fit(X_array, y_copy.ravel())
                y_pred = reg.predict(X_array).reshape(-1, 1)
                sigmas.append(np.std(y_copy - y_pred))
                y_changed = self.change_target_back(X_array, y_pred)
                error = np.abs(y_array.ravel() - y_changed.ravel()).sum(axis=None)
                errors.append(error)
                models.append(reg)
            i = np.argmin(errors)
            self.reg = models[i]
            self.sigma = sigmas[i]
        else:
            y_copy = y_array.reshape(-1, 1)
            self.reg = self.create_regressor()
            self.reg.fit(X_array, y_copy.ravel())
            y_pred = self.reg.predict(X_array).reshape(-1, 1)
            self.sigma = np.std(y_copy - y_pred)
 
    def _infer_discontinuities(self, real_dot, simulated_dot):
        bad_idx = (real_dot == 0.0)
        ratio = simulated_dot[~bad_idx] / real_dot[~bad_idx]
        good_ratio = np.median(ratio)
        self.good_ratio = good_ratio
        return
 
    def change_target(self, X, y, use_strict=True, threshold=3):
        """
        if the target is either theta1 or theta2, we update the target to be its difference with the last trace,
        we do this to get rid of the discontinuities (jump from -pi to pi or from pi to -pi when the acrobat is
        vertical and passes from one side to another)
        :param threshold: threshold to say that the variable is discontinuous
        :param X: features matrix
        :param y: target column
        :return: updated target
        """
        if self.target_dim not in [1, 3]:
            return y
        if use_strict:
            diff = y.ravel() - X[:, self.target_dim]
            neg_to_pos = diff > threshold
            pos_to_neg = diff < -threshold
            my_return = (pd.Series(diff.ravel()).where(~neg_to_pos, diff - 2 * np.pi)
                         .where(~pos_to_neg, 2 * np.pi + diff)).values.ravel()
            return my_return[:, None]
        else:
            real_dot = X[:, self.target_dim - 1].ravel()
            simulated_dot = y.ravel() - X[:, self.target_dim].ravel()
            cond = np.abs(real_dot) < self.good_ratio * (np.abs(simulated_dot) + 3)
            conds = [cond & (simulated_dot >= 0), cond & (simulated_dot < 0), ~cond]
            selected = np.select(conds, [simulated_dot - 2 * np.pi, simulated_dot + 2 * np.pi, simulated_dot]).ravel()
            return selected
 
    def change_target_back(self, X, pred):
        """
        reverse the operation of change_target, we update the prediction to be from -pi to pi as the desired value
        instead of being the difference with the last trace
        :param X: features matrix
        :param pred: predicted target
        :return: updated prediction
        """
        if self.target_dim not in [1, 3]:
            return pred
        sum_ = pred.ravel() + X[:, self.target_dim]
        r = np.where(sum_ <= -np.pi, sum_ + 2 * np.pi, sum_)
        return np.where(sum_ >= np.pi, sum_ - 2 * np.pi, r)[:, None]
 
    def predict(self, X_array):
 
        """Construct a conditional mixture distribution.
 
        Be careful not to use any information from the future
        (X_array[t + 1:]) when constructing the output.
 
        Parameters
        ----------
        X_array : pandas.DataFrame
            The input array. The features extracted by the feature extractor,
            plus `target_dim` system observables from time step t+1.
 
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

        types = np.array([[0, 0], ] * len(X_array))
        y_pred, stds = self.reg.predict(X_array, return_std=True)
        y_pred = y_pred.reshape(-1, 1)
        y_pred = self.change_target_back(X_array, y_pred)
        sigmas = np.clip(stds, np.maximum(self.sigma * 0.1, 0.001), self.sigma * 1.7)
        sigmas = sigmas[:, np.newaxis]
        params = np.concatenate((y_pred, sigmas), axis=1)
 
        sigmas = np.array([100 * (self.sigma + 0.001)] * len(X_array))
        sigmas = sigmas[:, np.newaxis]
        params_safety = np.concatenate((y_pred, sigmas), axis=1)
        params = np.concatenate((params, params_safety), axis=1)
        weights = np.array([[0.999, 0.001], ] * len(X_array))
        return weights, types, params
 
 
class StackedEstimator(RegressorMixin, BaseEstimator):
    """
    Idea credit to : https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python
    Stack a number of strong estimators into one estimator, a number of estimators on the first layer
    are being trained on a cross validation fashion, the predictions of these layers are being used afterwards
    by a second layer of estimators to predict the output (similar to the ResNet from neural networks), the final
    output is then an average of all of the second layer' estimations
    The general structure is as follows :
 
    F ____
          |
          |______ S _____P3 __* proba3
          |             |      |       |
    F ____|             |      |       |
          |             |      |       |
          |______ S ____|P2 __* proba2 |
          |             |      |       |  Sum = Final mean prediction
    F ____|             |      |       |
          |             |      |       |
          |             |      |       |
          |______ S ____|P1___* proba1 |
          |                    |
    F ____|________________ R _|
 
    Where F represent a first level regressor, S a second level regressor and R a classifier used to choose
    the weights for each of the second level regressors
 
    """
 
    def __init__(self):
        self.first_level_regressors = {}
        self.second_level_regressors = {}
        self.regressor_selector = None
        self.KFold = KFold(n_splits=20)
 
    def add_first_level_regressors(self, clf, name):
        """
        add a regressor to the first layer of the model
        :param clf: regressor
        :param name: name of the regressor
        :return : StackedEstimator object
        """
        self.first_level_regressors[name] = clf
        return self
 
    def add_second_level_regressors(self, clf, name):
        """
        add a regressor to the second layer of the model
        :param clf: regressor
        :param name: name of the regressor
        :return : StackedEstimator object
        """
        self.second_level_regressors[name] = clf
        return self
 
    def _fit_first_level(self, X_train, y_train):
        preds = {}
        for name, clf in self.first_level_regressors.items():
            clf, out = self._get_regressor_cv_prediction(clf, X_train, y_train)
            preds[name] = out
            self.first_level_regressors[name] = clf
        return pd.DataFrame(preds)
 
    def predict_first_level(self, X):
 
        preds = {}
        for name, clf in self.first_level_regressors.items():
            preds[name] = clf.predict(X)
        return pd.DataFrame(preds)
 
    def _get_regressor_cv_prediction(self, regressor, X_train, y_train, cv=None, fit_at_end=True):
 
        if cv is None:
            cv = self.KFold
        out_train = np.zeros_like(y_train)
        for i, (idx_train, idx_test) in enumerate(cv.split(X_train)):
            cloned = clone(regressor)
            cloned.fit(X_train[idx_train], y_train[idx_train])
            out_train[idx_test] = cloned.predict(X_train[idx_test])
        if fit_at_end:
            regressor.fit(X_train, y_train)
        return regressor, out_train
 
    def _fit_second_level_regressors(self, X, y):
        if len(self.second_level_regressors) <= 1 or self.regressor_selector is None:
 
            for name, clf in self.second_level_regressors.items():
                clf.fit(X, y)
            return self
        else:
            cv = KFold(n_splits=10)
            preds = []
            for name, reg in self.second_level_regressors.items():
                _, out_train = self._get_regressor_cv_prediction(reg, X, y, cv=cv, fit_at_end=True)
                preds.append(out_train.ravel())
            preds = np.asarray(preds)
            errors = preds.T - np.asarray(y).reshape(-1, 1)
            self._fit_regressor_selector(X, errors)
 
    def fit(self, X, y):
        preds = self._fit_first_level(X, y)
        X_train = np.hstack((X, preds.values))
        self._fit_second_level_regressors(X_train, y)
        return self
 
    def add_regressor_selector(self, clf):
        """
        a classifier to predict what second level classifier will yield the best result (predictions as probabilities)
        :param clf: the classifier instance, it should impliment predict and predict_proba
        :return: current instance of StackedEstimator
        """
        self.regressor_selector = clf
        return self
 
    def _fit_regressor_selector(self, X_train, errors):
        """
        fits the regressor selector instance using the errors from each regressor and the input data
        :param X_train: the input data to the regressors
        :param errors: the errors vector, a column for each regressor,
        :return: a StackedEstimator instance
        """
        errors = np.asarray(errors)
        targets = np.abs(errors).argmin(axis=1)
        self.regressor_selector.fit(X_train, targets)
        return self
 
    def _select_regressors(self, X_train):
        values = self.regressor_selector.predict_proba(X_train)
        df = pd.DataFrame(np.zeros((X_train.shape[0], len(self.second_level_regressors))))
        for class_, value in zip(self.regressor_selector.classes_, values.T):
            df.iloc[:, class_] = value
        return df.values
 
    def predict(self, X, return_std=False):
        preds = self.predict_first_level(X)
        X_ = np.hstack((X, preds.values))
        preds_second_level = {}
        for name, clf in self.second_level_regressors.items():
            preds_second_level[name] = clf.predict(X_)
        all_preds = pd.DataFrame(preds_second_level)
        if not (len(self.second_level_regressors) <= 1 or self.regressor_selector is None):
            probas = self._select_regressors(X_)
            predictions = (all_preds * probas).sum(axis=1).values
        else:
            predictions = all_preds.mean(axis=1).values
        if return_std:
            stds_array = np.hstack((preds.values, all_preds.values))
            return predictions, np.std(stds_array, axis=1)
        return predictions
 
 
# helper functions
 
def weighted_std_and_average(X, weights):
    mean = np.average(X, weights=weights, axis=1)
    var = np.average((X - mean.reshape(-1, 1)) ** 2, weights=weights, axis=1)
    return mean, np.sqrt(var)
