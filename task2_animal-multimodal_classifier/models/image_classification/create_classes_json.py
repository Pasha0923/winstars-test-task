import os
import json

data_dir = "data/raw-img"
output_dir = "data"

os.makedirs(output_dir, exist_ok=True)

classes = sorted([c for c in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, c))]) if os.path.exists(data_dir) else []

if classes:
    with open(os.path.join(output_dir, "classes.json"), "w") as f:
        json.dump(classes, f)
