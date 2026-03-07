from .rf_model import RandomForestModel
from .ffnn_model import NeuralNetworkModel
from .cnn_model import CNNModel
class MnistClassifier:

    def __init__(self, algorithm="rf"):

        if algorithm == "rf":
            self.model = RandomForestModel()

        elif algorithm == "nn":
            self.model = NeuralNetworkModel()

        elif algorithm == "cnn":
            self.model = CNNModel()

        else:
            raise ValueError("Unknown algorithm")

    def train(self, X_train, y_train):
        self.model.train(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)