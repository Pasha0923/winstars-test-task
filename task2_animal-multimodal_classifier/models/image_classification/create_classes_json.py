import os
import json

data_dir = "data/raw-img"
output_dir = "data"

print("Current working dir:", os.getcwd())
print("data_dir exists?", os.path.exists(data_dir))
print("data_dir content:", os.listdir(data_dir) if os.path.exists(data_dir) else "N/A")

os.makedirs(output_dir, exist_ok=True)

classes = sorted([c for c in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, c))]) if os.path.exists(data_dir) else []

if classes:
    with open(os.path.join(output_dir, "classes.json"), "w") as f:
        json.dump(classes, f)

print("Saved classes:", classes)