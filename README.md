# Computer-Vision

![UTA-DataScience-Logo](https://github.com/user-attachments/assets/fec1b411-bda5-437a-9eb8-08a018eb84ae)

# Facial Emotion Recognition using Transfer Learning

This repository contains a deep learning project focused on classifying human facial expressions into seven emotion categories using transfer learning on image data from a public Kaggle dataset.

---

## One-Sentence Summary

This repository applies transfer learning using MobileNetV2, EfficientNetB0, and ResNet50 to recognize facial emotions from images, based on the [Facial Expression Dataset](https://www.kaggle.com/datasets/msambare/fer2013) resized to 224×224.

---

## Overview

This project tackles the classification of facial expressions into 7 emotional states: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, and `surprise`. The goal is to train accurate image classification models using transfer learning to support emotion-aware systems and mental health applications.

The project uses pre-trained models from Keras (MobileNetV2, EfficientNetB0, ResNet50), combined with a clean subset of images for training. The approach emphasizes efficient model performance without data augmentation.

Our best model achieved **79% validation accuracy** on the train dataset, and  **74% validation accucary** on the test dataset.

---

## Summary of Work Done

### Data

- **Source**: Custom subset of the [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
- **Classes**: 7 emotions (angry, disgust, fear, happy, neutral, sad, surprise)
- **Resolution**: All images resized to 224x224
- **Class Imbalance**: Selected 224 random images per class totaling 1568 images to train the models

---

### Preprocessing / Clean-up

- Manual filtering to select **224 clean images per class**
- Resizing all images to **224×224** pixels
- Normalization applied via built-in Keras preprocessors
- Dataset loaded using `image_dataset_from_directory` with batch prefetching

---

### Data Visualization

<img width="796" height="789" alt="Image" src="https://github.com/user-attachments/assets/70b65b26-a4ee-47a2-9bae-4c544cfe6023" />


<img width="792" height="790" alt="Image" src="https://github.com/user-attachments/assets/4c75e706-3d26-4183-abfd-4a91499f984a" />

---

### Problem Formulation

- **Input**: 224×224 RGB face image
- **Output**: One of 7 emotion classes (softmax classification)

---

### 🧱 Models

Three transfer learning models were tested:
- **MobileNetV2**  
- **EfficientNetB0**
- **ResNet50**   

*  **Loss**: `SparseCategoricalCrossentropy`
*  **Optimizer**: Adam
* **Callbacks**: EarlyStopping, ReduceLROnPlateau

### Training

- **Hardware**: Local MacBook using CPU 
- **Software**: TensorFlow 2.16
- **Batch Size**: 10
- **Epochs**: 10 (laptop can only handle 10 epochs)
- **Stopping Criteria**: Val loss plateau

Training Curves:

- All models showed steadily improving accuracy over epochs.
- **RestNet50** peaked at **~79% val accuracy**
- **EfficientNetB0** and **MobileNetV2** reached **~77% validation accuracy**

---

### Performance Comparison

<img width="564" height="454" alt="Image" src="https://github.com/user-attachments/assets/e563f64b-6326-40be-b636-289197f6381c" />

<img width="568" height="452" alt="Image" src="https://github.com/user-attachments/assets/048040e2-f685-480a-adda-ab630f84d2b2" />

<img width="567" height="458" alt="Image" src="https://github.com/user-attachments/assets/9831fbdb-db3a-422c-bfa6-d568c38dc4d1" />

---

### Conclusions

- **RestNet50** outperformed other models both in validation and test settings.
- Avoiding augmentation didn’t hinder performance due to good-quality, balanced data.
- BatchNorm and learning rate scheduling improved stability.

---

## How to Reproduce Results

- Set up environment: Download the dataset [Facial Expression Dataset](https://www.kaggle.com/datasets/msambare/fer2013); on the CV_DataLoad-2.ipynb change the path to match the one on your computer.
- Install tensorflow, keras
- Run the notebooks in order:

| Notebook                                          | Description                                               |
| ------------------------------------------------- | --------------------------------------------------------- |
| `CV_DataLoad-2.ipynb`                             | Loads and visualizes data, splits into train/val datasets |
| `TrainBaseModel.ipynb`                            | Builds and trains baseline                                |
| `TrainBaseModelAugmentation.ipynb`                | Adds augmentation and trains again                        |
| `CompareAugmentation.ipynb`                       | Compares baseline vs augmented using ROC                  |
| `TrainTransferModels.ipynb`                       | Trains all three models from scratch                      |   
| `ModelsOnTestData.ipynb`                          | Tests all models on test dataset                          |

---


## Citations

- FER2013 Dataset on Kaggle: [Facial Expression Dataset](https://www.kaggle.com/datasets/msambare/fer2013)

- [TensorFlow Transfer Learning Docs](https://keras.io/api/applications/)



