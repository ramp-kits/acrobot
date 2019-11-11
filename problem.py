import os
import json
import xarray as xr
import rampwf as rw
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

problem_title = 'Acrobot simulator'
_n_burn_in = 0  # number of guaranteed steps in time series history
_max_dists = 100  # max number of kernels to use in generative regressors
_target_column_observation_names = [
    'thetaDot2', 'cos(theta1)', 'sin(theta1)',
    'cos(theta2)', 'sin(theta2)', 'thetaDot1',
]
_target_column_action_names = ['action']
Predictions = rw.prediction_types.make_generative_regression(
    _max_dists, label_names=_target_column_observation_names)

score_types = [
    rw.score_types.LikelihoodRatioDists('likelihood_ratio'),
]

cv = rw.cvs.PerRestart()
get_cv = cv.get_cv

workflow = rw.workflows.TSFEGenReg(
    check_sizes=[137], check_indexs=[13], max_dists=_max_dists,
    target_column_observation_names=_target_column_observation_names,
    target_column_action_names=_target_column_action_names,
    restart_names=['restart'],
    timestamp_name='time',
)

def _read_data(path, y_name=None, X_name=None, X_array=None):
    if y_name is not None:
        # Common data for both reward and observation simulators
        X_array = pd.read_pickle(
            os.path.join(path, 'data', X_name))

    # Target for observation
    y_array_obs = X_array[_target_column_observation_names][1:]

    y_array_obs.reset_index(drop=True, inplace=True)

    # a(t), to be used in observation simulator only
    extraX = X_array[_target_column_action_names][1:]

    extraX.rename(columns=lambda x: x + '_extra', inplace=True)
    extraX.reset_index(drop=True, inplace=True)

    # We drop the last value of a(t-1),o(t-1) : we do not have data
    # for a(t) at last timestep if we don't drop it
    X_array = X_array.iloc[:-1]

    y_array = y_array_obs

    date = X_array.index.copy()
    X_array.reset_index(drop=True, inplace=True)
    X_array = pd.concat([X_array, extraX], axis=1)

    # We now have to add the y in X to account for the correlations in our
    # regressors

    extra_truth = ['y_' + obs for obs in _target_column_observation_names]
    columns_X = list(X_array.columns)

    y_array_no_name = pd.DataFrame(y_array.values)
    X_array.reset_index(drop=True, inplace=True)
    X_array = pd.concat([X_array, y_array_no_name], axis=1)

    new_names = columns_X + extra_truth
    X_array.set_axis(new_names, axis=1, inplace=True)

    X_array.set_index(date, inplace=True)
    X_array = xr.Dataset(X_array)
    X_array.attrs['n_burn_in'] = _n_burn_in
    return X_array, y_array.values


def get_train_data(path='.'):
    return _read_data(path, y_name='y_train', X_name='X_train')


def get_test_data(path='.'):
    return _read_data(path, y_name='y_test', X_name='X_test')
