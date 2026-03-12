from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

class NERModel:
    """
    NERModel wraps a DistilBERT model for token classification.
    """
    def __init__(self, model_name="distilbert-base-cased", num_labels=3):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        # Fine-tune last two transformer layers train classifier head
        for param in self.model.distilbert.transformer.layer[-2:].parameters():
            param.requires_grad = True

        for param in self.model.classifier.parameters():
            param.requires_grad = True
    def tokenize_and_align_labels(self, texts, labels, label2id):
        """
        Tokenize texts and align token-level labels for training
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
                    label_id.append(-100)
                elif word_idx != previous_word_idx:
                    label_id.append(label2id[label[word_idx]])
                else:
                    label_id.append(-100) 
                previous_word_idx = word_idx
            label_ids.append(label_id)

        tokenized_inputs["labels"] = torch.tensor(label_ids)
        return tokenized_inputs