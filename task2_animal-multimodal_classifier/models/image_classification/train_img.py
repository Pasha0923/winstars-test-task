import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

data_dir = "data/raw-img"
classes_path = "trained_models/classes.json"
save_model_path = "trained_models/img_model.pth"

with open(classes_path, "r") as f:
    classes = json.load(f)
num_classes = len(classes)

# Define image transformations for ResNet
transform = transforms.Compose([
    transforms.Resize((160,160)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize pre-trained ResNet18 model and freeze all layers to speed up training
model = resnet18(weights=ResNet18_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False

# Replace ResNet's final layer head with new layer for num_classes
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
epochs = 2

model.train()
for epoch in range(epochs):
    total_loss = 0
    for i, (X_batch, y_batch) in enumerate(dataloader):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i % 10 == 0:
            print(f"Epoch {epoch+1} | Batch {i}/{len(dataloader)} | Loss {loss.item():.4f}")
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} finished | Avg Loss: {avg_loss:.4f}")

os.makedirs("trained_models", exist_ok=True)
torch.save(model.state_dict(), save_model_path)
print("Model saved to:", save_model_path)