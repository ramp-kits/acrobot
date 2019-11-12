import pandas as pd
import numpy as np


class FeatureExtractor():
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
        We simply use the observables at time t as the state.
        Be careful not to use any information from the future (X_ds[t + 1:])
        when constructing X_df[t].
        Parameters
        ----------
        X_ds : xarray.Dataset
            The raw time series.
        Return
        ------
        X_df : pandas Dataframe

        """
        X_df = X_df_raw.to_dataframe()

        restart = X_df[self.restart_name].values

        # Since we do not use the restart information in our regressor, we have
        # to remove it
        X_df = X_df.drop(columns=self.restart_name)

        tail = 10
        result_array = []
        curr_tail = tail
        for i in range(len(X_df)):
            if restart[i] == 1:
                # If we encounter a restart, tail is set to 0
                curr_tail = 0
            elif curr_tail < tail:
                # And it goes back up to it's normal length if no other restarts
                curr_tail += 1

            X_temp = X_df.iloc[
                [idx for idx in range(i - curr_tail, i + 1) if idx >= 0]
            ].mean(axis=0)
            result_array.append(X_temp)
        result_array = np.vstack(result_array)
        additional_dim = pd.DataFrame(result_array)

        new_names = [item + "_engineered" for item in list(X_df_raw.keys())]
        additional_dim.rename(
            columns={i: item for i, item in enumerate(new_names)}, inplace=True)

        date = X_df.index.copy()
        X_df.reset_index(drop=True, inplace=True)
        X_array = pd.concat([X_df, additional_dim], axis=1)

        X_array.set_index(date, inplace=True)

        # We return a dataframe with additional features (with no clashing names)
        # based on previous values' mean, being mindful about restarts

        return X_array
