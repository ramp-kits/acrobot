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
            The list of states.
        """

        X_df = X_df_raw.to_dataframe()

        # Since we do not use the restart information in our regressor, we have
        # to remove it
        restart = X_df[self.restart_name].values
        X_df = X_df.drop(columns=self.restart_name)

        return X_df
