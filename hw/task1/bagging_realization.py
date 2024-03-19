import copy
from typing import Dict

import numpy as np
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor


class MyBaggingRegressor:
    def __init__(self, base_estimator=DecisionTreeRegressor(), n_estimators=10, random_state=0):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.base_estimator.set_params(**{'random_state': random_state})
        self.estimators = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        for i in range(self.n_estimators):
            sample_idx = np.random.choice(X.index, size=len(X), replace=True)
            X_sample, y_sample = X.loc[sample_idx], y.loc[sample_idx]

            estimator = self.base_estimator.fit(X_sample, y_sample)
            self.estimators.append(estimator)

    def predict(self, X):
        y_preds = np.zeros(shape=(len(X), len(self.estimators)))
        for i, estimator in enumerate(self.estimators):
            y_preds[:, i] = estimator.predict(X)

        return np.mean(y_preds, axis=1)

    def get_params(self, deep=True):  # Modify the signature slightly
        return {
            'n_estimators': self.n_estimators
        }

    def set_params(self, **params):
        if not params:
            return self

        valid_params = self.get_params(deep=True)
        base_estimator_params = self.base_estimator.get_params()
        for key, value in params.items():
            split = key.split('__', 1)
            if len(split) > 1:
                # Nested parameter for base_estimator
                base_estimator_param, param_name = split
                if base_estimator_param == 'base_estimator' and param_name in base_estimator_params:
                    self.base_estimator.set_params(**{param_name: value})
            else:
                # Parameter for MyBaggingRegressor
                if key in valid_params:
                    setattr(self, key, value)

        return self

    def score(self, X, y):
        y_pred = self.predict(X)

        return r2_score(y, y_pred)


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import warnings
#
# warnings.simplefilter("ignore")
#
# import numpy as np
# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
# from sklearn.ensemble import BaggingRegressor
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.model_selection import GridSearchCV
#
# from hw.task1.bagging_realization import MyBaggingRegressor
# from hw.task1.ada_boosting_realization import MyAdaBoostRegressor
#
#
# import re
# import ast
# import pandas as pd
#
#
# def preprocess_data(df, best_artists):
#     del_cols = ['id', 'release_date', 'valence', 'artists', 'duration_ms', 'key', 'liveness', 'loudness', 'mode',
#                 'name']
#
#     df['artists'] = df['artists'].apply(lambda x: ast.literal_eval(x))
#     if best_artists is None:
#         top_30_songs = df[df['popularity'] > 70]
#         best_artists = set(top_30_songs['artists'].explode().unique())
#         del_cols += ['popularity']
#
#     df['is_have_best_artist'] = df['artists'].apply(lambda x: len(set(x) & best_artists) > 0)
#     df['name_length'] = df['name'].str.len()
#     df['has_special_chars'] = df['name'].apply(lambda x: bool(re.search(r"[^\w\s]", x))).astype(int)
#     df.drop(del_cols, axis=1, inplace=True)
#     return df, best_artists
#
#
# df = pd.read_csv('../../data/spotify/spotify_dataset.csv')
# df.head()
#
# from sklearn.model_selection import train_test_split
#
# X, Y = df.drop(['popularity'], axis=1), df['popularity']
# train, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=42)
#
# train['popularity'] = train_Y
# train_X, best_artists = preprocess_data(train, None)
#
# test_X, _ = preprocess_data(test_X, best_artists)
#
# df = pd.DataFrame(columns=['Ensemble_Name', 'R2_Score', 'Training_time'])
#
# metrics = [r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error]
#
# base_estimator = DecisionTreeRegressor()
# model = MyBaggingRegressor(base_estimator=base_estimator)
#
# # Create the combined parameter grid
# param_grid = {
#     'base_estimator__max_depth': [2, 4, 6],
#     'base_estimator__max_features': ['auto', 'sqrt'],
#     'n_estimators': [10, 30, 50],
# }
#
# grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', refit=True)
#
# grid_search.fit(train_X[:30], train_Y[:30])
