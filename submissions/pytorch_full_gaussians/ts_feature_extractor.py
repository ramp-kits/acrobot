import pandas as pd
import numpy as np


class FeatureExtractor():
    def __init__(self, restart_name):
        self.restart_name = restart_name
        pass

    def transform(self, X_df_raw):
        """
        :param X_df_raw: The initial dataframe
        :return: a dataframe with additional features (with no clashing names)
                 bqsed on previous values' mean, being mindfull about restarts
        """

        X_df = X_df_raw.to_dataframe()

        restart = X_df[self.restart_name].values
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

        return X_array
