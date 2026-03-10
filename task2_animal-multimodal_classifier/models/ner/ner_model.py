# models/ner/ner_model.py
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
import torch

class NERModel:
    def __init__(self, model_name="dslim/bert-base-NER"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)

    def preprocess(self, texts):
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    def predict(self, text):
        tokens = self.preprocess([text])
        with torch.no_grad():
            outputs = self.model(**tokens).logits
        predictions = torch.argmax(outputs, dim=2)
        return self.decode(predictions, tokens)