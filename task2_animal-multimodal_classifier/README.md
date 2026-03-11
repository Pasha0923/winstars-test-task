# Task 2: Animal Multimodal Classifier

## Overview

This project implements a multimodal animal classifier that combines:

1. **Text understanding (NER) to extract an animal name from a sentence**

2. **Image classification (CNN) to recognize an animal in an image**

3. **The pipeline:** 
- Extracts the animal from the text using the NER model
- Predicts the animal in the image using the CNN classifier
- Compares both predictions returns True / False depending on whether they refer to the same animal  

## Project Structure
```bash
task2_animal-multimodal_classifier/
│
├── data/
│   └── ner_dataset.json        # synthetic dataset used for NER fine-tuning
│   └── classes.json            # mapping between class indices and animal names
│
├── notebooks/
│   └── eda.ipynb               # exploratory data analysis of the Animals-10 dataset
│
├── models/                     # model implementations

│   ├── ner/
│   │   ├── ner_model.py        # NER model architecture (DistilBERT for token classification)
│   │   ├── train_ner.py        # script for fine-tuning the NER model
│   │   ├── inference_ner.py    # script for extracting animal names from text
│   │   └── generate_ner_dataset.py  # script to generate synthetic NER training data
│
│   ├── image_classification/
│   │   ├── img_model.py        # CNN image classifier (ResNet18)
│   │   ├── train_img.py        # script for training the image classifier
│   │   ├── inference_img.py    # script for predicting the animal from an image
│   │   └── create_classes_json.py   # utility script to generate class mapping file
│
├── pipeline.py                 # multimodal pipeline (text + image → True/False)
│
├── trained_models/
│   ├── img_model.pth           # trained weights of the image classification model
│   
│
├── examples/image                # example images for testing the pipeline
│  
├── pyproject.toml              # project dependencies (Poetry)
├── README.md                   # project documentation
└── .gitignore                  # ignored files
```

## Exploratory Data Analysis

Exploratory Data Analysis was performed in: notebooks/eda.ipynb

During EDA we:

- counted the number of images per class

- built a distribution plot of classes

- visualized sample images

- analyzed image sizes

- verified dataset consistency

This analysis helped understand the dataset before training the image classifier.

## Image Classification Model

The model was trained on the Animals-10 dataset from Kaggle:

https://www.kaggle.com/datasets/alessiocorrado99/animals10

The dataset contains 10 animal classes: "butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"

The trained weights are included in the repository:

trained_models/img_model.pth

## NER Model

The NER model extracts an animal name from a sentence : 

- Pretrained base: DistilBERT for token classification
- The model was fine-tuned on custom dataset to recognize only the animal classes used in the image classifier (data/ner_dataset.json)

The trained NER model weights exceed the GitHub file size limit.

Therefore they are provided via Google Drive.

Download link:

https://drive.google.com/drive/folders/1Hbwdq0-5YsCFl0RZlXCx3vqhoB3Z8FcI

After downloading:

1. Extract the folder
2. Copy it into: trained_models/

Final structure should look like:
```bash
trained_models/
├── img_model.pth
└── ner_model/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── label2id.json
```
## Requirements 

Python 3.11 – 3.13
Poetry (for dependency management)
 Libraries:
- torch
- torchvision
- transformers
- scikit-learn
- numpy
- pandas
- pillow
- matplotlib
- accelerate

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Pasha0923/winstars-test-task.git
cd task2_animal-multimodal_classifier
```
2. Install dependencies using Poetry:
```bash
poetry install
```

## Running 
1. Image Classification

Test image classification only:
```bash
 poetry run python models/image_classification/inference_img.py --image examples/image/1.jpeg
```
 2. NER (Text → Animal)
 Test text animal extraction:
```bash
poetry run python models/ner/inference_ner.py --text "Your text with animals from dataset , example cat"
```   
3. Multimodal Pipeline (Text + Image)
```bash
poetry run python pipeline.py --text "I have a feeling that the animal shown here might be a butterfly" --image examples/image/1.jpeg
```       

## Notes

This project demonstrates a multimodal AI pipeline that combines:

- NLP (NER) for extracting entities from text
- Computer Vision (CNN) for image classification
- Pipeline logic to compare predictions from both modalities
- The result is a simple but complete text + image verification system.