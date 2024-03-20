import numpy as np
from typing import Any, List, Tuple, Dict

from sklearn.base import clone
from sklearn.metrics import r2_score


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
