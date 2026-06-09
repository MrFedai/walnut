# Fine-Grained Classification of Walnut Varieties Using Deep Learning

## Project Overview

This project investigates the application of deep learning techniques for the fine-grained visual classification of 18 visually similar walnut cultivars. Utilizing transfer learning with a ResNet18 backbone pretrained on ImageNet, the study explores challenges inherent to distinguishing subtle morphological differences across cultivars. A particular focus is placed on analyzing failure cases, notably the inability of the model to correctly classify the *maya1* cultivar, attributed to high intra-class visual similarity with *kaplan86*.

---

## Key Features

- **Model Architecture**: ResNet18 with ImageNet1K_V1 pretrained weights.
- **Transfer Learning Approach**: Feature extraction followed by fine-tuning on the walnut dataset.
- **Scientific Failure Case Analysis**: Detailed investigation of misclassification patterns, particularly focusing on the *maya1* cultivar.
- **Comprehensive Evaluation**: Includes accuracy, F1 scores, confusion matrices, and per-class performance metrics.

---

## Prerequisites

Ensure you have the following installed:

- Python >= 3.8
- PyTorch >= 1.10
- TorchVision >= 0.11
## Git Clone
```bash
git clone https://github.com/MrFedai/walnut.git
cd walnut
```
Install dependencies from `requirements.txt`:
```bash
python -m venv env
source env/bin/activate  # Linux/macOS
# for Windows : env\Scripts\activate
```
```bash
pip install -r requirements.txt
```

---

## Installation & Dataset

Due to size constraints, the dataset is hosted externally. Please download it using the link below and extract the contents into the `dataset/` folder.

\[ LINK TO DATASET \]

The dataset is organized in the following structure:

```
dataset/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
├── val/
│   ├── class1/
│   ├── class2/
│   └── ...
└── test/
    ├── class1/
    ├── class2/
    └── ...
```

---

## Usage

### Training

To train the model, run:

```bash
python train.py
```

### Evaluation

To evaluate the trained model on the test set, run:

```bash
python evaluate.py
```

---

## Reproducibility

All experiments were conducted with a fixed random seed (`42`) to ensure reproducible data splits and training outcomes. Modify or remove the seed in `train.py` for non-deterministic behavior.
---
## Results
![image alt](https://github.com/MrFedai/walnut/blob/03bc82e2aa29a4bbd30bc1bd9fbb52d1f69ab45e/confusion_matrix.png)
---

## Contact / Credits

**Author**: [Fatih Gok]  
**Course**: AI Lab, Sapienza University of Rome  
**Year**: 2025-2026

