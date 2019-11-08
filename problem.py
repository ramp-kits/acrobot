import os
import json
import xarray as xr
import rampwf as rw
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

N_BURN_IN = 0  # number of guaranteed steps in time series history
MAX_DISTS = 300  # number of kernels to use
problem_title = 'Acrobot simulator'
with open(os.path.join('.', 'data_train_hackathon', 'metadata_rl.json'), 'r') as json_file:
    metadata = json.load(json_file)
_target_column_name_observation = metadata['observation']
_target_column_name_action = metadata['action']
Predictions = rw.prediction_types.make_generative_regression(
    MAX_DISTS, label_names=_target_column_name_observation)

score_types = [
    rw.score_types.LikelihoodRatioDists('lk_ratio_obsv'),
]

cv = rw.cvs.PerRestart()
get_cv = cv.get_cv


def _read_data(path, y_name=None, X_name=None, X_array=None):
    if y_name is not None:
        # Common data for both reward and observation simulators
        X_array = pd.read_pickle(
            os.path.join(path, 'data_train_hackathon', X_name))

    # Target for observation
    y_array_obs = X_array[_target_column_name_observation][1:]

    y_array_obs.reset_index(drop=True, inplace=True)

    # a(t), to be used in observation simulator only
    extraX = X_array[_target_column_name_action][1:]

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

    extra_truth = ['y_' + obs for obs in _target_column_name_observation]
    columns_X = list(X_array.columns)

    y_array_no_name = pd.DataFrame(y_array.values)
    X_array.reset_index(drop=True, inplace=True)
    X_array = pd.concat([X_array, y_array_no_name], axis=1)

    new_names = columns_X + extra_truth
    X_array.set_axis(new_names, axis=1, inplace=True)

    X_array.set_index(date, inplace=True)
    X_array = xr.Dataset(X_array)
    X_array.attrs['n_burn_in'] = N_BURN_IN
    return X_array, y_array.values


def get_train_data(path='.'):
    return _read_data(path, y_name='y_train', X_name='X_train')


def get_test_data(path='.'):
    return _read_data(path, y_name='y_test', X_name='X_test')


class FeatureExtractorRegressor:
    def __init__(self,
                 _target_column_name_observation,
                 _target_column_name_action,
                 workflow_element_names=None):

        self._name_restart = metadata['restart']

        if workflow_element_names is None:
            workflow_element_names = ['ts_feature_extractor',
                                      'generative_regressor']
        self.element_names = workflow_element_names

        self.feature_extractor_workflow = \
            rw.workflows.ts_feature_extractor.TimeSeriesFeatureExtractor(
                check_sizes=[132], check_indexs=[13],
                workflow_element_names=[self.element_names[0]],
                restart_name=self._name_restart)

        self.regressor_workflow = rw.workflows.GenerativeRegressor(
            _target_column_name_observation, MAX_DISTS,
            workflow_element_names=[self.element_names[1]],
            restart_name=metadata['restart'],
            check_sizes=[132], check_indexs=[13])

        self._target_column_name_observation = _target_column_name_observation
        self._target_column_name_action = _target_column_name_action

    def train_submission(self, module_path, X_df, y_array, train_is=None):

        # FE uses is o(t-1), a(t-1) concatenated without a(t)
        # If train is none here, it still should not be a slice,
        # because of ts_fe

        fe = self.feature_extractor_workflow.train_submission(
            module_path, X_df, y_array, train_is)

        if train_is is None:
            train_is = slice(None, None, None)

        cols_for_extraction = self._target_column_name_observation + \
                              self._target_column_name_action + \
                              self._name_restart

        X_train_array = self.feature_extractor_workflow.test_submission(
            fe,
            X_df[cols_for_extraction][{metadata['timestamp_name']: train_is}])

        # X_obs is o(t-1), a(t-1) concatenated with a(t)
        extra_actions = [act + '_extra' for act in
                         self._target_column_name_action]

        X_obs = pd.concat([X_train_array,
                           X_df[extra_actions][{
                               metadata['timestamp_name']: train_is
                           }].to_dataframe()],
                          axis=1)
        obs = ['y_' + obs for obs in self._target_column_name_observation]
        reg = self.regressor_workflow.train_submission(
            module_path, X_obs, X_df.to_dataframe()[obs].iloc[train_is].values)
        return fe, reg

    def test_submission(self, trained_model, X_df):

        fe, reg = trained_model

        cols_for_extraction = self._target_column_name_observation + \
                              self._target_column_name_action + \
                              self._name_restart

        X_test_array = self.feature_extractor_workflow.test_submission(
            fe, X_df[cols_for_extraction])

        extra_actions = [act + '_extra' for act in
                         self._target_column_name_action]
        X_obs = pd.concat(
            [X_test_array, X_df[extra_actions].to_dataframe()], axis=1)

        extra_obs = ['y_' + obs for obs in
                     self._target_column_name_observation]
        X_obs = pd.concat([X_obs, X_df[extra_obs].to_dataframe()], axis=1)

        y_pred_obs = self.regressor_workflow.test_submission(reg, X_obs)

        nb_dists = y_pred_obs[0, 0]

        assert nb_dists <= MAX_DISTS, \
            "The maximum number of distributions allowed is {0}" \
            "but you use {1}".format(MAX_DISTS, nb_dists)

        return y_pred_obs

    def step(self, trained_model, X_df, seed=None):

        fe, reg = trained_model

        cols_for_extraction = self._target_column_name_observation + \
                              self._target_column_name_action + \
                              self._name_restart

        X_test_array = self.feature_extractor_workflow.test_submission(
            fe, X_df[cols_for_extraction])

        # We only care about sampling for the last provided timestep
        X_test_array = X_test_array.iloc[-1]

        extra_actions = [act + '_extra' for act in
                         self._target_column_name_action]
        newest_actions = X_df[extra_actions].to_dataframe().iloc[-1]
        X_obs = X_test_array.append(newest_actions)

        sampled = self.regressor_workflow.step(reg, X_obs, seed)

        sampled_df = pd.DataFrame(sampled)

        new_names = self._target_column_name_observation
        sampled_df.set_axis(new_names, axis=1, inplace=True)

        return sampled_df


workflow = FeatureExtractorRegressor(
    _target_column_name_observation, _target_column_name_action)
