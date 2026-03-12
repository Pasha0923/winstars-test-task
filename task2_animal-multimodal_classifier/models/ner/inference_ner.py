import json
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import argparse
from pathlib import Path

def predict_text_animal(text, model_path, label_map_path):
    """Predict the animal mentioned in the input text using a fine-tuned NER model"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.to(device)
    model.eval()
    
    # Load label mapping
    with open(label_map_path, "r") as f:
        label2id = json.load(f)
    id2label = {v: k for k, v in label2id.items()}

    # Tokenize input text
    words = text.split()
    inputs = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)

    # Get model predictions
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
    """CLI entry point for predicting animal in text"""
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