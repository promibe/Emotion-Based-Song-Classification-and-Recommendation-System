from sklearn.base import BaseEstimator, TransformerMixin
# feaure engineering class

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return self.feature_eng(X)

    @staticmethod
    def feature_eng(df3):
        #scale loudness column
        df3['loudness'] = df3['loudness'] / df3['loudness'].max()

        #scale spec_rate column
        df3['spec_rate'] = df3['spec_rate'] * 1000000

        #Engineering the feature that affect the emotion label to caption interaction
        df3['all_six'] = df3['danceability'] * df3['energy'] * df3['loudness'] * df3['acousticness'] * df3['instrumentalness'] * df3['valence']
        df3['DE'] = df3['danceability'] * df3['energy']
        df3['DL'] = df3['danceability'] * df3['loudness']
        df3['DV'] = df3['danceability'] * df3['valence']
        df3['EL'] = df3['energy'] + df3['loudness']
        df3['EA'] = df3['energy'] * df3['acousticness']
        df3['EV'] = df3['energy'] * df3['valence']
        df3['LA'] = df3['loudness'] * df3['acousticness']
        df3['LI'] = df3['loudness'] * df3['instrumentalness']
        df3['LV'] = df3['loudness'] * df3['valence']
        df3['AI'] = df3['acousticness'] * df3['instrumentalness']
        df3['IV'] = df3['instrumentalness'] * df3['valence']
        df3['TL'] = df3['tempo'] * df3['loudness']
        df3['TA'] = df3['tempo'] * df3['acousticness']
        return df3