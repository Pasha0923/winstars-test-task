# Task 1: MNIST Image Classification

## Overview

This project demonstrates the implementation of three classification models for the MNIST dataset:

1. **Random Forest (RF)**  
2. **Feed-Forward Neural Network (NN)**  
3. **Convolutional Neural Network (CNN)**  

All models implement a unified interface `MnistClassifierInterface` with two methods:

- `train(X_train, y_train)` – train the model  
- `predict(X)` – make predictions  

The `MnistClassifier` class acts as a wrapper to select the algorithm (`rf`, `nn`, `cnn`) while keeping the **same input/output structure**.

## Project Structure
```bash
task1_mnist_image_classification/
├── models/
│ ├── interface.py # abstract classifier interface
│ ├── cnn_model.py # CNN
│ ├── nn_model.py # Feed-Forward NN
│ ├── rf_model.py # Random Forest
│ ├── mnist_classifier.py # main classifier wrapper
| ├── __init__.py
├── utils/
│ └── mnist_loader.py # function to load MNIST dataset
├── notebook/ 
| └── demo.ipynb # notebook demonstrating model training and evaluation
├── pyproject.toml # Poetry project file
|── .gitignore
|── requirements.txt
```

## Requirements 
 Python 3.13

 Libraries:
- numpy==2.4.2
- scipy==1.17.1
- scikit-learn==1.8.0
- torch==2.10.0
- torchvision==0.25.0

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your_username/winstars-test-task.git
cd task1_mnist_image_classification
```
2. Install dependencies (poetry or pip):

using poetry:
```bash
poetry install
```
using pip:
```bash
pip install -r requirements.txt
```
## Usage

### Demo Notebook

demo.ipynb demonstrates:
-Loading MNIST data
-Train the implemented models (RF, NN, CNN)
-Evaluating accuracy
-Edge case testing

The file `demo.ipynb` demonstrates how the models work.  
You can open it in Jupyter Notebook/VS Code or Google Colab and run the cells sequentially.

## Edge Cases
We test how the model handles:
- Correct single image
- Missing batch dimension
- Incorrect input shape

## Notes
- The MNIST dataset is automatically downloaded via torchvision.datasets.MNIST.
- Temporary files (like __pycache__/, .vscode/, and data/) are ignored via .gitignore.
- demo.ipynb contains examples and edge cases to demonstrate model functionality.


