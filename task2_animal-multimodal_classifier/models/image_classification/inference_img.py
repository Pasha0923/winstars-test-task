# models/image_classification/inference_img.py
import os
import json
import torch
from PIL import Image
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
import argparse

# -------------------------------
# 1️⃣ Аргументы
parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, required=True, help="Path to image")
parser.add_argument("--model_path", type=str, default="trained_models/img_model.pth")
parser.add_argument("--classes_path", type=str, default="trained_models/classes.json")
args = parser.parse_args()

# -------------------------------
# 2️⃣ Загружаем классы
if not os.path.exists(args.classes_path):
    raise FileNotFoundError(f"Classes file not found: {args.classes_path}")
with open(args.classes_path, "r") as f:
    classes = json.load(f)
num_classes = len(classes)

# -------------------------------
# 3️⃣ Трансформации
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ResNet нормализация
                         std=[0.229, 0.224, 0.225])
])

# -------------------------------
# 4️⃣ Загружаем модель
device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet18(weights=None)  # не грузим предобученные, так как тренировали сами
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(args.model_path, map_location=device))
model = model.to(device)
model.eval()

# -------------------------------
# 5️⃣ Загружаем изображение
if not os.path.exists(args.image):
    raise FileNotFoundError(f"Image file not found: {args.image}")
img = Image.open(args.image).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)

# -------------------------------
# 6️⃣ Предсказание
with torch.no_grad():
    output = model(img_tensor)
    pred_idx = output.argmax(dim=1).item()
    pred_class = classes[pred_idx]

print(f"Predicted class: {pred_class}")