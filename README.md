# 🦠 Malaria Parasite Detector from Blood Smear Images

A deep learning-based medical image classification project that detects whether a blood smear cell image is **Parasitized** or **Uninfected** using Convolutional Neural Networks (CNN) and Transfer Learning (MobileNetV2).

> **Disclaimer:** This project is an educational prototype and should **not** be used for clinical diagnosis.

---

## 📌 Project Overview

Malaria is a life-threatening disease caused by parasites transmitted through mosquito bites. Manual examination of blood smear images is time-consuming and requires skilled microscopists. This project demonstrates how Artificial Intelligence can assist in identifying malaria parasites from microscopic blood smear images.

The project implements:

- Custom CNN from scratch
- Transfer Learning using MobileNetV2
- Data preprocessing and augmentation
- Binary image classification
- Performance evaluation
- Model explainability (Grad-CAM)
- Robustness analysis

---

## 🎯 Problem Statement

Develop a binary image classifier that classifies segmented thin blood smear cell images into:

- **Parasitized**
- **Uninfected**

using Deep Learning techniques.

---

## 🚀 Features

- Binary Image Classification
- Custom CNN Architecture
- Transfer Learning with MobileNetV2
- Image Preprocessing
- Data Augmentation
- Stratified Train/Validation/Test Split
- Confusion Matrix
- ROC Curve
- Accuracy, Precision, Recall, F1-Score, Specificity, ROC-AUC
- Grad-CAM Visualization
- Robustness Testing

---

## 🏥 Business Use Cases

- AI-assisted malaria screening
- Educational medical imaging prototype
- Microscopy image quality control
- Demonstration of AI in infectious disease screening

---

## 📂 Dataset

**Dataset:** Malaria Cell Images

Source:
- NIH / LHNCBC
- Kaggle TensorFlow Mirror

Dataset contains:

- 27,558 Cell Images
- Balanced Dataset

Classes:

```
Parasitized
Uninfected
```

---

## 📁 Project Structure

```
malaria-parasites/
│
├── dataset/
│
├── models/
│
├── reports/
│
├── src/
│   ├── config.py
│   ├── preprocess.py
│   ├── gpu_setup.py
│   ├── train.py
│   ├── transfer_learning.py
│   ├── evaluate.py
│   ├── gradcam.py
│   └── robustness.py
│
├── main.py
├── streamlitapp.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit

---

## 🔄 Workflow

```
Dataset
      │
      ▼
Image Preprocessing
      │
      ▼
Train / Validation / Test Split
      │
      ▼
Image Augmentation
      │
      ▼
CNN Training
      │
      ▼
Transfer Learning (MobileNetV2)
      │
      ▼
Model Evaluation
      │
      ▼
Grad-CAM Visualization
      │
      ▼
Performance Report
```

---

## 🖼️ Data Preprocessing

The preprocessing pipeline includes:

- Image resizing
- Pixel normalization
- Data augmentation
- Stratified train-validation-test split

### Augmentation

- Random Horizontal Flip
- Random Rotation
- Random Zoom
- Random Contrast

---

## 🧠 Models Used

### 1. Custom CNN

A CNN built from scratch using TensorFlow/Keras.

### 2. Transfer Learning

MobileNetV2 pre-trained on ImageNet with a custom classification head.

---

## 📊 Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall (Sensitivity)
- Specificity
- F1-Score
- ROC-AUC
- Confusion Matrix

Example output:

```
Accuracy      : 98.4%
Precision     : 98.2%
Recall        : 98.6%
Specificity   : 98.3%
F1 Score      : 98.4%
ROC-AUC       : 99.4%
```

---

## 📈 Visualizations

The project generates:

- Confusion Matrix
- ROC Curve
- Training Accuracy
- Training Loss
- Validation Accuracy
- Validation Loss
- Grad-CAM Heatmaps

---

## 🔍 Robustness Analysis

Experiments performed:

- Blur Noise
- Gaussian Noise
- Reduced Training Dataset
- Performance Comparison

---

## 📷 Explainability

Grad-CAM is used to visualize which regions of the blood smear image influence the model's prediction.

This improves model interpretability and demonstrates that the network focuses on parasite regions rather than background artifacts.

---

## 💻 Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/malaria-parasites.git
```

Navigate into the project

```bash
cd malaria-parasites
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment

**macOS/Linux**

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

Train the CNN

```bash
python main.py
```

Evaluate the model

```bash
python src/evaluate.py
```

Launch the Streamlit application

```bash
streamlit run streamlitapp.py
```

---

## 📊 Results

The project produces:

- Trained CNN model
- MobileNetV2 model
- Evaluation metrics
- ROC Curve
- Confusion Matrix
- Grad-CAM Visualizations
- Robustness Analysis

---

## 📌 Future Improvements

- ResNet50 Transfer Learning
- EfficientNet
- Vision Transformers
- Hyperparameter Optimization
- Model Quantization
- Deployment on Mobile Devices
- Real-time Malaria Detection

---

## 📚 Learning Outcomes

This project demonstrates practical knowledge of:

- Computer Vision
- Binary Image Classification
- CNN Architecture
- Transfer Learning
- Medical Image Analysis
- TensorFlow
- Model Explainability
- Performance Evaluation

---

## 📜 License

This project is intended for educational and research purposes only.

---

## 👩‍💻 Author

**Jasleen Kaur**

GitHub: https://github.com/jasleen2162-source
