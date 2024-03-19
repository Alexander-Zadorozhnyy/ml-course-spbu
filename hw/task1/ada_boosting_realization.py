import numpy as np
from typing import Any, List, Tuple, Dict

from sklearn.base import clone
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.tree import DecisionTreeRegressor


class MyAdaBoostRegressor:
    def __init__(self,
                 estimator: Any,
                 n_estimators: int = 100,
                 loss: str = 'linear',
                 random_state: int = 0) -> None:
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.loss_type = loss
        self.f = []
        self.model_weights = []
        self.mean_loss = []
        self.random_state = random_state
        self.estimator.set_params(**{'random_state': random_state})

    @staticmethod
    def __linear_loss(y_pred: np.array, y_train: np.array) -> np.array:
        loss = np.abs(y_pred - y_train)
        loss /= np.max(loss)
        return loss

    @staticmethod
    def __square_loss(y_pred: np.array, y_train: np.array) -> np.array:
        loss = np.abs(y_pred - y_train)
        loss = np.power(loss / np.max(loss), 2)
        return loss

    @staticmethod
    def __exponential_loss(y_pred: np.array, y_train: np.array) -> np.array:
        loss = np.abs(y_pred - y_train)
        loss = 1 - np.exp(-loss / np.max(loss))
        return loss

    def __compute_loss(self, y_pred: np.array, y_train: np.array) -> np.array:
        match self.loss_type:  # select correct loss function
            case 'linear':
                return self.__linear_loss(y_pred, y_train)
            case 'square':
                return self.__square_loss(y_pred, y_train)
            case 'exponential':
                return self.__exponential_loss(y_pred, y_train)
            case _:
                raise ValueError('Incorrect loss type entered, must be linear, square, or exponential')

    def __compute_beta(self, p: np.array, loss: np.array) -> Tuple[float, float]:
        # compute the mean loss
        mean_loss = np.sum(np.multiply(loss, p))
        # calculate beta
        beta = mean_loss / (1 - mean_loss)
        # store model weights & mean loss
        self.model_weights.append(np.log(1.0 / beta))
        self.mean_loss.append(mean_loss)

        return beta, mean_loss

    def __sort_array_by_column(self, arr):
        # Apply sorting function to each row efficiently
        return np.apply_along_axis(lambda x: sorted(x), axis=0, arr=arr)

    def __weighted_median(self, y_samp: np.array) -> np.array:
        # sort sample predictions column-wise
        samp_idx = np.argsort(y_samp, axis=0)
        sorted_y = self.__sort_array_by_column(y_samp)
        # k = samp_idx.T
        # sorted_y = y_samp[samp_idx.T]
        # sorted_y = np.array([sorted_y[i, :, i] for i in range(sorted_y.shape[0])]).T
        # sort the model weights according to samp_idx
        sorted_mw = np.array(self.model_weights)[samp_idx]
        # do cumulative summation on columns
        cumsum_mw = np.cumsum(sorted_mw, axis=0)
        # solve inequality
        pred_idx = cumsum_mw >= 0.5 * cumsum_mw[-1, :]
        pred_idx = pred_idx.argmax(axis=0)
        # return weighted medians
        return sorted_y[pred_idx].diagonal()

    def get_params(self, deep: bool = False) -> Dict:
        return {'estimator': self.estimator,
                'n_estimators': self.n_estimators,
                'loss': self.loss_type}

    def set_params(self, **params):
        if not params:
            return self

        valid_params = self.get_params(deep=True)
        estimator_params = self.estimator.get_params()
        for key, value in params.items():
            split = key.split('__', 1)
            if len(split) > 1:
                # Nested parameter for estimator
                estimator_param, param_name = split
                if estimator_param == 'estimator' and param_name in estimator_params:
                    self.estimator.set_params(**{param_name: value})
            else:
                # Parameter for MyAdaBoostRegressor
                if key in valid_params:
                    setattr(self, key, value)

        return self

    def fit(self, X_train: np.array, y_train: np.array) -> None:
        # initialise sample weights, model weights, mean loss, & model array
        w = np.ones((y_train.shape[0]))
        self.f = []
        self.model_weights = []
        self.mean_loss = []
        np.random.seed(self.random_state)
        # loop through the specified number of iterations in the ensemble
        for i in range(self.n_estimators):
            # make a copy of the weak learner
            model = clone(self.estimator)
            # calculate probabilities for each sample
            p = w / np.sum(w)
            # sample training set according to p
            bootstrap_idx = np.random.choice(
                X_train.index,
                size=len(X_train),
                replace=True,
                p=p,
            )

            # Fit on the bootstrapped sample and obtain a prediction
            # for all samples in the training set
            X = X_train.loc[bootstrap_idx]
            y = y_train.loc[bootstrap_idx]
            # fit the current weak learner on the boostrapped dataset
            model.fit(X, y)
            # obtain predictions from the current weak learner on the entire dataset
            y_pred = model.predict(X_train)
            # compute the loss
            loss = self.__compute_loss(y_pred, y_train)
            # compute the adaptive weight
            beta, mean_loss = self.__compute_beta(p, loss)
            # check our mean loss
            if mean_loss >= 0.5:
                break
            # update sample weights
            w *= np.power(beta, 1 - loss)
            # append resulting model
            self.f.append(model)

    def get_loss(self) -> List:
        return self.mean_loss

    def predict(self, X_test: np.array) -> np.array:
        # obtain sample predictions for each memeber of the ensemble
        y_samp = np.array([model.predict(X_test) for model in self.f])
        # do weighted median over the samples
        print(y_samp.shape)
        y_pred = self.__weighted_median(y_samp)
        # input()
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)

    def __del__(self) -> None:
        del self.estimator
        del self.n_estimators
        del self.loss_type
        del self.f
        del self.model_weights
        del self.mean_loss

#
# import re
# import ast
# import pandas as pd
# from sklearn.model_selection import train_test_split, GridSearchCV
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
#
# X, Y = df.drop(['popularity'], axis=1), df['popularity']
# train, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=42)
#
# train['popularity'] = train_Y
# train_X, best_artists = preprocess_data(train, None)
# test_X, _ = preprocess_data(test_X, best_artists)
#
# from datetime import datetime
#
#
# def estimate_model(name, grid: GridSearchCV, train_X, train_Y, test_X, test_Y, df):
#     res = {'ensemble_name': name}
#
#     start_time = datetime.now()
#     grid.fit(train_X, train_Y)
#     end_time = datetime.now()
#     res['training_time'] = end_time - start_time
#     model = grid.best_estimator_
#
#     y_pred = model.predict(test_X)
#     res['r2_score'] = r2_score(test_Y, y_pred)
#     res['mean_squared_error'] = mean_squared_error(test_Y, y_pred)
#     res['mean_absolute_error'] = mean_absolute_error(test_Y, y_pred)
#     res['mean_absolute_percentage_error'] = mean_absolute_percentage_error(test_Y, y_pred)
#
#     df = pd.concat([df, pd.DataFrame(res, index=[0])])
#     return grid, model, df
#
#
# df = pd.DataFrame(
#     columns=['ensemble_name', 'r2_score', 'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error',
#              'training_time'])
#
# from sklearn.model_selection import StratifiedKFold
#
# random_state = 0
#
# param_grid = {
#     # 'n_estimators': [5, 10, 20],
#     'estimator__max_depth': [2, 5, 10],
#     # 'estimator__max_features': [2, 4, 8, len(train_X.columns)],
#     # 'estimator__criterion': ["squared_error", "friedman_mse", "absolute_error", "poisson"],
#     # 'estimator__splitter': ["best", "random"],
#     'estimator__min_samples_split': [2, 4, 6, 10],
#     # 'estimator__min_samples_leaf': [1, 2, 4, 6, 8, 10, 12],
#     'estimator__random_state': [random_state],
# }
#
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
#
# adaboost_dt = MyAdaBoostRegressor(estimator=DecisionTreeRegressor(), random_state=random_state)
#
# my_adaboost_grid_model = GridSearchCV(estimator=adaboost_dt, cv=cv,
#                                       param_grid=param_grid,
#                                       scoring='r2',
#                                       refit=True)
#
# my_adaboost_grid, my_adaboost_model, df = estimate_model('MyAdaBoostRegressor', my_adaboost_grid_model, train_X,
#                                                          train_Y,
#                                                          test_X.head(5000), test_Y.head(5000), df)
# #
# # if __name__ == '__main__':
# #     y_samp = np.array([[1, 2, 6], [5, 6, 0]])
# #     shape = y_samp.shape
# #     tmp = y_samp.ravel()
# #     tmp = np.sort(tmp)
# #     tmp = tmp.reshape(shape)
# #     samp_idx = np.argsort(y_samp, axis=0)
# #     k = samp_idx.T
# #     sorted_y = y_samp[samp_idx.T]
# #     sorted_y = np.array([sorted_y[i, :, i] for i in range(sorted_y.shape[0])]).T
# #     pp = 1
# #     # sort the model weights according to samp_idx
# #
# #     def sort_array_by_column(arr):
# #         # Apply sorting function to each row efficiently
# #         return np.apply_along_axis(lambda x: sorted(x), axis=0, arr=arr)
# #
# #     # Sample array
# #     data = np.array([[1, 2, 6], [5, 6, 0]])
# #
# #     # Sort the array
# #     sorted_data = sort_array_by_column(data)
# #
# #     # Print the sorted array
# #     print(sorted_data)
