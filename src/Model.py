from abc import ABC, abstractmethod

import numpy as np
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC

RANDOM_STATE = 1
N_ESTIMATORS = 100


class Model(ABC):
    """klasa abstrakcyjna dla modeli, ze wspólnym interfejsem dla konkretnych
    realizacji algorytmu"""

    @abstractmethod
    def set_estimator(self, params):
        pass

    def fit(self, train_X, train_y):
        self.model.fit(train_X, train_y)

    def predict(self, X):
        return self.model.predict(X).astype(int)

    def score(self, X, y):
        preds = self.predict(X)
        result = (preds == y)
        unique, counts = np.unique(result, return_counts=True)
        return counts[1] / (counts[0] + counts[1])

    def save(self):
        dump(self.model, self.name)

    def load(self):
        self.model = load(self.name)

    def get_mae(self, val_X, val_y):
        preds = self.predict(val_X)
        return mean_absolute_error(val_y, preds)

    def get_mse(self, val_X, val_y):
        preds = self.predict(val_X)
        return mean_squared_error(val_y, preds)


class LinearRegressionModel(Model):
    def __init__(self):
        self.name = "./Models/LinearRegression"
        self.model = LinearRegression()

    def set_estimator(self, params):
        self.model = LinearRegression(**params)


class LogisticRegressionModel(Model):
    def __init__(self):
        self.name = "./Models/LogisticRegression"
        self.model = LogisticRegression(tol=1e-3, random_state=RANDOM_STATE,
                                        solver='liblinear', max_iter=1000,
                                        multi_class='auto')

    def set_estimator(self, params):
        self.model = LogisticRegression(**params)


class SVMModel(Model):
    def __init__(self):
        self.name = "./Models/SVM"
        self.model = SVC(gamma='scale')

    def set_estimator(self, params):
        self.model = SVC(**params)


class RandomForestModel(Model):
    def __init__(self):
        self.name = "./Models/RandomForest"
        self.model = RandomForestClassifier(n_estimators=N_ESTIMATORS,
                                            random_state=RANDOM_STATE)

    def set_estimator(self, params):
        self.model = RandomForestClassifier(**params)
