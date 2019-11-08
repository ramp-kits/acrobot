
class FeatureExtractor():
    def __init__(self, restart_name):
        self.restart_name = restart_name
        pass

    def transform(self, X_df_raw):
        X_df = X_df_raw.to_dataframe()

        # Since we do not use the restart information in our regressor, we have
        # to remove it
        restart = X_df[self.restart_name].values
        X_df = X_df.drop(columns=self.restart_name)


        return X_df