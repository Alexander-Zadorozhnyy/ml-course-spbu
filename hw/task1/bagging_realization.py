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
