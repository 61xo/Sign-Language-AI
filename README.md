# 🤟 Sign Language Alphabet Recognition (DNN)

A high-performance, real-time sign language alphabet classifier built with **Python** and **Deep Learning**. This project transforms hand gestures into text using spatial landmark analysis.

---

## 🚀 Overview
This repository contains a complete pipeline for sign language recognition. Unlike traditional image classification, this project extracts **21 hand landmarks** (63 features) using **MediaPipe**, which are then processed by a **Deep Neural Network (DNN)** to predict characters with high precision and low latency.

---

## 🧠 Technical Architecture

The project is divided into three main phases:

### 1. Data Collection (`collect.py`)
* **Feature Extraction:** Uses `MediaPipe Hands` to detect hand landmarks from static images.
* **Data Formatting:** Records the `(x, y, z)` coordinates for each of the 21 landmarks.
* **Output:** Generates a structured `data1.csv` file containing the labels and their corresponding spatial features.

### 2. Model Training (`deepTrain.py`)
* **Network Type:** Multi-Layer Perceptron (MLP).
* **Layers:** * Input Layer (63 features).
    * Multiple **Dense** layers with **ReLU** activation.
    * **Dropout** layers (0.3 and 0.2) to mitigate overfitting.
    * Output Layer with **Softmax** activation for multi-class classification.
* **Optimization:** Uses the `Adam` optimizer and `Categorical Crossentropy` loss.
* **Preprocessing:** Implements `StandardScaler` for feature normalization and `LabelEncoder` for class management.

### 3. Real-time Inference (`deepPredict.py`)
* **Live Feed:** Captures video via OpenCV and processes frames in real-time.
* **Thresholding:** Implements a **80% confidence threshold** to ensure stable and accurate predictions.
* **Visualization:** Overlays predicted characters and probability percentages directly onto the video feed.

---

## 🛠️ Tech Stack
* **Language:** Python
* **AI/ML:** TensorFlow, Keras, Scikit-learn
* **Computer Vision:** OpenCV, MediaPipe
* **Data:** Pandas, NumPy, Joblib

---

## ⚙️ Setup & Usage

1. **Install Dependencies:**
   ```bash
   pip install tensorflow opencv-python mediapipe pandas scikit-learn joblib
