# models/ner/train_ner.py
import json
from datasets import Dataset
from transformers import Trainer, TrainingArguments , TrainerCallback
from ner_model import NERModel

# -------------------------------
# 1️⃣ Загружаем свой датасет с животными
with open("data/ner_dataset.json") as f:
    data = json.load(f)

texts = [x["text"].split() for x in data]  # сплит по словам
animals = [x["animal"] for x in data]

# -------------------------------
# 2️⃣ Создаём метки на уровне токена
labels_list = ["O"] + [f"B-{a}" for a in set(animals)]
label2id = {label: i for i, label in enumerate(labels_list)}
id2label = {i: label for label, i in label2id.items()}

all_labels = []
for words, animal in zip(texts, animals):
    label_seq = ["O"] * len(words)
    for i, w in enumerate(words):
        if animal.lower() in w.lower():
            label_seq[i] = f"B-{animal}"
    all_labels.append(label_seq)

# -------------------------------
# 3️⃣ Загружаем pre-trained NER модель
num_labels = len(labels_list)
ner_model = NERModel(num_labels=num_labels)

# -------------------------------
# 4️⃣ Токенизация + выравнивание меток
tokenized_dataset = ner_model.tokenize_and_align_labels(texts, all_labels, label2id)

dataset = Dataset.from_dict({
    "input_ids": tokenized_dataset["input_ids"],
    "attention_mask": tokenized_dataset["attention_mask"],
    "labels": tokenized_dataset["labels"]
})

training_args = TrainingArguments(
    output_dir="trained_models/ner_model",
    num_train_epochs=10,
    per_device_train_batch_size=16,
    logging_dir="./logs",
    save_strategy="epoch", 
    logging_strategy="steps",
    logging_steps=5 
)
from transformers import TrainerCallback

class PrintLossCallback(TrainerCallback):
    def on_log(self, args, state, control, **kwargs):
        logs = kwargs.get("logs", None)
        if logs is not None and "loss" in logs:
            print(f"Step {state.global_step} | Epoch {state.epoch:.2f} | Loss: {logs['loss']:.4f}")
# -------------------------------
# 6️⃣ Trainer
trainer = Trainer(
    model=ner_model.model,
    args=training_args,
    train_dataset=dataset,
    callbacks=[PrintLossCallback]
)

# -------------------------------
# 7️⃣ Обучение
trainer.train()

# -------------------------------
# 8️⃣ Сохраняем модель и токенизатор
ner_model.model.save_pretrained("trained_models/ner_model")
ner_model.tokenizer.save_pretrained("trained_models/ner_model")
with open("trained_models/ner_model/label2id.json", "w") as f:
    json.dump(label2id, f)
print("NER модель дообучена и сохранена!")