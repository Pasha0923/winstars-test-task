# models/image_classification/img_model.py
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

class ImageClassifier:
    def __init__(self, num_classes=10, device="cuda"):
        self.device = device
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model.to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])

    def predict(self, image): 
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            output = self.model(img_tensor)
            pred_class = output.argmax(dim=1).item()
        return pred_class