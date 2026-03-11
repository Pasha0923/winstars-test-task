import json

animal_classes = [
    "butterfly", "cat", "chicken", "cow", "dog",
    "elephant", "horse", "sheep", "spider", "squirrel"
]
templates = [
    "There is a {} in the picture",
    "I think the image shows a {}",
    "The animal in this photo is a {}",
    "Looks like a {} in the image",
    "Can you see the {} here?"
]

ner_dataset = []

for animal in animal_classes:
    for template in templates:
        ner_dataset.append({
            "text": template.format(animal),
            "animal": animal
        })

with open("data/ner_dataset.json", "w") as f:
    json.dump(ner_dataset, f, indent=4)

print(f"Generated {len(ner_dataset)} NER examples")