# Emotion Recognition CNN

This repository contains the codebase for a facial emotion recognition project based on the FER-2013 dataset. The goal is to classify facial expressions into discrete emotion categories using a Convolutional Neural Network (CNN) built on top of a fine-tuned EfficientNet-B0 backbone.

## Overview

This project fine-tunes EfficientNet-B0 on the FER-2013 dataset to classify faces into one of seven emotion categories:
`Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral`,
based on a training set with the following categories:
`Angry, Disgust, Fear, Happy`.

The model leverages data augmentation and weighted loss to handle class imbalance, and training callbacks like early stopping and learning rate scheduling to optimize performance.

---
*About the project*
  
## Objectives
-> Preprocess and augment the FER‑2013 dataset
-> Implement EfficientNet‑B0 (originally implemented ResNet), replace the classifier head, and fine‑tune on FER‑2013 with class‑weighted cross‑entropy and label smoothing
-> Evaluate the trained model on the held‑out test set and generate a confusion matrix  and per‑emotion confidence distributions

## Methods
Dataset: FER-2013 (35.9k images) with 7 emotion classes (Anger, Disgust, Fear, Happy, Sad, Surprise, Neutral) [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)

Model: Switched from ResNet to EfficientNet-B0 (as it's compact and a proved to be more suitable). The final classifier head was replaced to match the 7 emotion outputs.

Training Strategy:
- progressively training more layers of the pretrained backbone to leverage the learned features
- appling mixup augmentation and standard flips and rotations to add to the training set
- used label smoothing, dropout, and weight decay to prevent overfitting
- appling class weights in the loss function to compensate for underrepresented classes (for example, ‘disgust’ has only ~600 images)
- reduced from 64 to 32 to improve gradient stability
- tring differetn learning rates
- then repeating the process untill one of the combinations of the above mentione steps gives good results

Validation: Performance evaluated via confusion matrix and per-class confidence histograms.

## Results

| Metric   | Value |
| -------- | ----- |
| Accuracy | 82.2% |
| Loss     | 0.929 |

Visualizations: Classification report (precision/recall per emotion), training curves (accuracy & loss), and separate emotions confidence histograms.

![train-vs-val](https://github.com/user-attachments/assets/8a85c412-d062-49cf-93ff-8951f7436fc0)
![results1](https://github.com/user-attachments/assets/1327339e-90ea-4791-863e-49615bcbc6c5)

*Example output for 'happy':*
![confusion-matrix-happy](https://github.com/user-attachments/assets/80f76920-42e4-4f54-b9d4-a815bb89cc86)
![confidence_happy](https://github.com/user-attachments/assets/57afef25-f748-4668-b87f-134ddadc478b)


## Challenges Faced

*Early stages:*
At first, by using ResNet and and freezing many initial layers, outputs were very poor (barely reaching 30% accuracy and having high validation loss). These results would start plateauing, which triggered the early stopping at the 7th epoch. 
![early-results1](https://github.com/user-attachments/assets/a5b73366-4278-4d5a-801d-a842ad7363d4)
This was solved by gradually unfreezing layers (specifically 2, 3 and 4). This did help (do ovde si)

- Early Overfitting: frozen backbone led to high training accuracy but poor validation performance
- NaN Loss: instabilities that came from large class weights for rare classes

- Class Imbalance: ‘Disgust’ category is severely underrepresented (600 samples vs ~5000 for others), which limits recall
- Narrow Augmentation: initial augmentations were insufficient to cover data variability
- Fixes: unfreezing additional layers gradually; enhancing augmentations; appling sanity checks on class weights; using differnt learning rates and regularization

## Conclusions
Fine-tuned EfficientNet-B0 achieved better results than ResNet, even with more frozen layers.
Dropout, label smoothing, and expecially mixup stabilized training and improved generalization.
Frequent classes like ‘happy’ achieved higher confidence, while rare classes like ‘disgust’ remained challenging.
This shows the importance of architecture choice and data handling for this task.

---

## Usage

To train the model:

```
python train.py --epochs 15 --batch-size 32
```

To evaluate on the test set:

```
python evaluate.py --model-path models/best_model.pth
```

To test for a specific emotion from the training set:

```
python code/validation.py --data-dir data --batch-size 32 --model-path models/best.pth --emotion fear --num-samples 100
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
```

---

## Future Work

* Integrating AffectNet dataset for broader generalization
* Improving performance with ensemble methods
* Integrating with an object detection system for contextual emotion analysis, such as 'Driver Wellbeing Detection'

---

## License

This project is licensed under the MIT License.
