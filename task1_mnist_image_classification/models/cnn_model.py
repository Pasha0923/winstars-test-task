import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from .interface import MnistClassifierInterface

class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
    
class CNNModel(MnistClassifierInterface):

    def __init__(self, epochs=5, batch_size=64):
        self.model = CNN()
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters())

    def train(self, X_train, y_train):

        self.model.train()

        X_train = torch.tensor(X_train).float().unsqueeze(1)
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
        X = torch.tensor(X).float().unsqueeze(1)

        with torch.no_grad():
            outputs = self.model(X)
            preds = torch.argmax(outputs, dim=1)

        return preds.numpy()