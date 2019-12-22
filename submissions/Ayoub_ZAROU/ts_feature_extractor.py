# Author(s): Balazs Kegl <balazs.kegl@gmail.com>
# License: BSD 3 clause

import pandas as pd
import numpy as np
 
FEATURE_NAMES = ['thetaDot2', 'theta2', 'thetaDot1', 'theta1', 'action',
                 'diff_theta2', 'diff_theta1', 'theta_double_dot1',
                 'theta_double_dot2','cos_theta1', 'sin_theta1',
                 'cos_theta2', 'sin_theta2']
 
 
class FeatureExtractor:
    def __init__(self, restart_name):
        """
        Parameters
        ----------
        restart_name : str
            The name of the 0/1 column indicating restarts in the time series.
        """
        self.restart_name = restart_name
        pass
 
    def transform(self, X_df_raw):
        """Transform time series into list of states.
        We use the observables at time t as the state, concatenated to the mean
        of the last ten time steps, handling restarts.
 
        Be careful not to use any information from the future (X_ds[t + 1:])
        when constructing X_df[t].
        Parameters
        ----------
        X_df_raw : xarray.Dataset
            The raw time series.
        Return
        ------
        X_df : pandas Dataframe
 
        """
        X_df = X_df_raw.to_dataframe()
        changed_df = (X_df.pipe(change_df_by_restart)
                      .pipe(add_sin_cos))
        return changed_df[FEATURE_NAMES]
 
 
def add_diff(df, i=2):
    """
    adding the difference between theta and its previous value, while taking into acount the fact that
    theta could jump directly from pi to -pi or the other way around when one of  the acrobat parts is vertical
    :param df: the input data
    :param i: the index for which theta we create a diff operation
    :return: df updated with the new feature, being the difference between theta{i} and its previous value
    """
    diff = df["theta{}".format(i)].diff()
    neg_to_pos = diff > 3
    pos_to_neg = diff < -3
    x = (diff.where(~neg_to_pos, diff - 2 * np.pi)
         .where(~pos_to_neg, 2 * np.pi + diff))
    x[0] = df["theta{}".format(i)][0]
    df['diff_theta{i}'.format(i=i)] = x.values
    return df
 
 
def change_df_by_restart(df):
    """
    add features to the dataframe while being mindful about the restarts
    :param df: the input data
    :return: df to which we added the diff features, sin and cosine features, and double derivatives, the operation
        is done per restart
    """
    copy = df.copy()
    copy['cum_restart'] = copy['restart'].shift(-1).cumsum().ffill()
    return (copy.groupby('cum_restart').apply(lambda sub_df: sub_df.pipe(add_diff).pipe(add_diff, i=1)
                                              .pipe(add_double_derivatives))).sort_index()
 
 
def add_sin_cos(df):
    """
    add cosine and sine of theta1 and theta2 to the dataframe
    :param df: input data
    :return: dataframe with cosine and sine features, those being cosine and sine of theta1 and theta2
    """
    thetas = ['theta1', 'theta2']
    for theta in thetas:
        df['cos_{}'.format(theta)] = np.cos(df[theta])
        df['sin_{}'.format(theta)] = np.sin(df[theta])
    return df
 
 
def add_double_derivatives(df):
    """
    add estimated double derivatives of theta1 and theta2
    :param df: input data
    :return: dataframe updated with estimated double derivatives of theta1 and theta2
    """
    thetas = [1, 2]
    for theta in thetas:
        df['theta_double_dot{}'.format(theta)] = df['thetaDot{}'.format(theta)].diff()
        df['theta_double_dot{}'.format(theta)][0] = df['thetaDot{}'.format(theta)][0]
    return df
