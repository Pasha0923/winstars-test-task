# # models/ner/inference_ner.py
# import json
# import torch
# from transformers import AutoTokenizer, AutoModelForTokenClassification
# import argparse
# from pathlib import Path

# # -------------------------------
# # 1️⃣ Аргументы
# parser = argparse.ArgumentParser()
# parser.add_argument("--text", type=str, required=True, help="Text containing an animal")
# parser.add_argument("--model_path", type=str, default="../../trained_models/ner_model")
# parser.add_argument("--label_map_path", type=str, default="../../trained_models/ner_model/label2id.json")
# args = parser.parse_args()

# device = "cuda" if torch.cuda.is_available() else "cpu"

# # -------------------------------
# # 2️⃣ Пути через Path (относительно скрипта)
# base_path = Path(__file__).parent
# model_path = (base_path / args.model_path).resolve()
# tokenizer_path = model_path

# # -------------------------------
# # 3️⃣ Загружаем модель и токенизатор
# model = AutoModelForTokenClassification.from_pretrained(str(model_path))
# tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
# model.to(device)
# model.eval()

# # -------------------------------
# # 4️⃣ Загружаем label2id
# with open(base_path / args.label_map_path, "r") as f:
#     label2id = json.load(f)
# id2label = {v: k for k, v in label2id.items()}

# # -------------------------------
# # 5️⃣ Токенизация текста
# words = args.text.split()
# inputs = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True, padding=True).to(device)

# # -------------------------------
# # 6️⃣ Предсказание
# with torch.no_grad():
#     outputs = model(**inputs).logits

# pred_ids = torch.argmax(outputs, dim=2)[0].cpu().tolist()
# pred_labels = []
# word_ids = inputs.word_ids(batch_index=0)
# for idx, word_idx in enumerate(word_ids):
#     if word_idx is None:
#         continue
#     pred_labels.append(id2label[pred_ids[idx]])

# # -------------------------------
# # 7️⃣ Извлекаем первое животное
# pred_animal = None
# for label in pred_labels:
#     if label.startswith("B-"):
#         pred_animal = label[2:]
#         break

# # -------------------------------
# # 8️⃣ Вывод
# if pred_animal:
#     print(f"Animal found in text: {pred_animal}")
# else:
#     print("No animal found in text.")
# models/ner/inference_ner.py

import json
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import argparse
from pathlib import Path


def predict_text_animal(text, model_path, label_map_path):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.to(device)
    model.eval()

    with open(label_map_path, "r") as f:
        label2id = json.load(f)

    id2label = {v: k for k, v in label2id.items()}

    words = text.split()

    inputs = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs).logits

    pred_ids = torch.argmax(outputs, dim=2)[0].cpu().tolist()

    word_ids = inputs.word_ids(batch_index=0)

    for idx, word_idx in enumerate(word_ids):

        if word_idx is None:
            continue

        label = id2label[pred_ids[idx]]

        if label.startswith("B-"):
            return label[2:]

    return None


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--text", type=str, required=True, help="Text containing an animal")
    parser.add_argument("--model_path", type=str, default="../../trained_models/ner_model")
    parser.add_argument("--label_map_path", type=str, default="../../trained_models/ner_model/label2id.json")

    args = parser.parse_args()

    base_path = Path(__file__).parent

    model_path = (base_path / args.model_path).resolve()
    label_map_path = (base_path / args.label_map_path).resolve()

    pred_animal = predict_text_animal(
        args.text,
        str(model_path),
        str(label_map_path)
    )

    if pred_animal:
        print(f"Animal found in text: {pred_animal}")
    else:
        print("No animal found in text.")


if __name__ == "__main__":
    main()