import os
import json
import xarray as xr
import rampwf as rw
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

problem_title = 'Acrobot system identification'
_n_burn_in = 0  # number of guaranteed steps in time series history
_max_dists = 100  # max number of kernels to use in generative regressors
_target_column_observation_names = [
    'thetaDot2', 'theta2',
    'thetaDot1', 'theta1', 
]
_target_column_action_names = ['action']
_restart_names = ['restart']

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
    restart_names=_restart_names,
    timestamp_name='time',
)


def get_train_data(path='.'):
    return _read_data(path, X_name='X_train')


def get_test_data(path='.'):
     return _read_data(path, X_name='X_test')


def _read_data(path, X_name=None):
    X_df = pd.read_pickle(os.path.join(path, 'data', X_name))
    # reorder columns according to _target_column_observation_names
    X_df = X_df.reindex(
        columns=_target_column_observation_names +
        _target_column_action_names + _restart_names)
    # Target for observation
    y_df = X_df[_target_column_observation_names][1:]
    y_df.reset_index(drop=True, inplace=True)

 
    # We drop the last step of X since we do not have data
    # for a(t) at last timestep
    X_df = X_df.iloc[:-1]
    date = X_df.index.copy()

    # Since in validation we will need to gradually give y to the
    # conditional regressor, we now have to add y in X.

    extra_truth = ['y_' + obs for obs in _target_column_observation_names]
    columns_X = list(X_df.columns)

    y_df_no_name = pd.DataFrame(y_df.values)
    X_df.reset_index(drop=True, inplace=True)
    X_df = pd.concat([X_df, y_df_no_name], axis=1)

    new_names = columns_X + extra_truth
    X_df.set_axis(new_names, axis=1, inplace=True)

    X_df.set_index(date, inplace=True)
    X_ds = xr.Dataset(X_df)
    X_ds.attrs['n_burn_in'] = _n_burn_in
    return X_ds, y_df.values
