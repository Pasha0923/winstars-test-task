import numpy as np
from .interface import MnistClassifierInterface
from sklearn.ensemble import RandomForestClassifier

class RandomForestModel(MnistClassifierInterface):

    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)

    def train(self, X_train, y_train):
        X_train = X_train.reshape(X_train.shape[0], -1)
        self.model.fit(X_train, y_train)

    def predict(self, X):
        X = X.reshape(X.shape[0], -1)
        return self.model.predict(X)