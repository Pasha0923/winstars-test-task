import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
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

    def __init__(self, epochs=5, batch_size=64):
        self.model = SimpleNN()
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters())

    def train(self, X_train, y_train):
        self.model.train()
        X_train = torch.tensor(X_train).float()
        y_train = torch.tensor(y_train).long()

        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            for X_batch, y_batch in loader:
                outputs = self.model(X_batch)
                loss = self.loss_fn(outputs, y_batch)
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