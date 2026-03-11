# models/ner/ner_model.py
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

class NERModel:
    def __init__(self, model_name="distilbert-base-cased", num_labels=3):
        """
        num_labels = количество сущностей: O + животные
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )

        # # замораживаем все слои кроме классификатора (transfer learning)
        # for param in self.model.distilbert.parameters():
        #     param.requires_grad = False
        # 1️⃣ Разморозим последние 2 слоя DistilBERT
        for param in self.model.distilbert.transformer.layer[-2:].parameters():
            param.requires_grad = True

        # 2️⃣ Классификатор всегда учится
        for param in self.model.classifier.parameters():
            param.requires_grad = True
    def tokenize_and_align_labels(self, texts, labels, label2id):
        """
        texts: список предложений
        labels: список списков с метками токенов
        """
        tokenized_inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            is_split_into_words=True,
            return_tensors="pt"
        )

        label_ids = []
        for i, label in enumerate(labels):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_id = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_id.append(-100)  # игнорируем токены [CLS], [SEP]
                elif word_idx != previous_word_idx:
                    label_id.append(label2id[label[word_idx]])
                else:
                    label_id.append(-100)  # субтокены игнорируем
                previous_word_idx = word_idx
            label_ids.append(label_id)

        tokenized_inputs["labels"] = torch.tensor(label_ids)
        return tokenized_inputs