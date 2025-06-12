# Emotion Recognition CNN

This repository contains the codebase for a facial emotion recognition project based on the FER-2013 dataset. The goal is to classify facial expressions into discrete emotion categories using a Convolutional Neural Network (CNN) built on top of a fine-tuned EfficientNet-B0 backbone.

## Overview

This project fine-tunes EfficientNet-B0 on the FER-2013 dataset to classify faces into one of seven emotion categories:
`Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral`,
based on a training set with the following categories:
`Angry, Disgust, Fear, Happy`.

The model leverages data augmentation and weighted loss to handle class imbalance, and training callbacks like early stopping and learning rate scheduling to optimize performance.

## Usage

To train the model:

```bash
python train.py --epochs 15 --batch-size 32
```

To evaluate on the test set:

```bash
python evaluate.py --model-path models/best_model.pth
```

## Prerequisites

- Python 3.8 or higher
- PyTorch >= 1.8
- torchvision
- numpy
- matplotlib

You can install all required packages with:

```bash
pip install -r requirements.txt


## Dataset

We used the [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) dataset, a publicly available dataset for facial expression classification.

## Results

| Metric   | Value |
| -------- | ----- |
| Accuracy | 68.4% |
| Loss     | 1.271 |

## Future Work

* Integrating AffectNet dataset for broader generalization
* Improving performance with ensemble methods
* Integrating with an object detection system for contextual emotion analysis, such as 'Driver Wellbeing Detection'

## License

This project is licensed under the MIT License.
