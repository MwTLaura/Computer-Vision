# Computer-Vision

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



---

### Problem Formulation

- **Input**: 224×224 RGB face image
- **Output**: One of 7 emotion classes (softmax classification)

---
