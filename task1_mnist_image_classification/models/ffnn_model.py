import torch
import torch.nn as nn
import torch.optim as optim
from .interface import MnistClassifierInterface


class SimpleNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)
    
class NeuralNetworkModel(MnistClassifierInterface):

    def __init__(self, epochs=5):
        self.model = SimpleNN()
        self.epochs = epochs
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters())

    def train(self, X_train, y_train):
        self.model.train()
        X_train = torch.tensor(X_train).float()
        y_train = torch.tensor(y_train).long()

        for epoch in range(self.epochs):

            outputs = self.model(X_train)
            loss = self.loss_fn(outputs, y_train)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
    def predict(self, X):
        self.model.eval()
        X = torch.tensor(X).float()

        with torch.no_grad():
            outputs = self.model(X)
            predictions = torch.argmax(outputs, dim=1)

        return predictions.numpy()