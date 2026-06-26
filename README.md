<div align="center">

# 🎤 Speech Emotion Recognition using Deep Learning

### AI-powered Speech Emotion Recognition using TensorFlow, Librosa and Streamlit

<p align="center">

<img src="https://img.shields.io/badge/Python-3.13-blue?logo=python">

<img src="https://img.shields.io/badge/TensorFlow-2.21-FF6F00?logo=tensorflow&logoColor=white">

<img src="https://img.shields.io/badge/Streamlit-1.58-FF4B4B?logo=streamlit&logoColor=white">

<img src="https://img.shields.io/badge/Deep%20Learning-AI-blueviolet">

<img src="https://img.shields.io/badge/Status-Completed-brightgreen">

<img src="https://img.shields.io/github/stars/Divyeshinturi/Speech-Emotion-Recognition?style=social">

</p>

---

### 🎯 Detect Human Emotions from Speech using Artificial Intelligence

Built using **TensorFlow**, **Librosa**, **Streamlit**, **Plotly**, and **Scikit-learn**.

</div>

---

# 📖 Table of Contents

- Project Overview
- Features
- Demo
- Dataset
- Project Workflow
- Model Architecture
- Model Performance
- Screenshots
- Installation
- Usage
- Folder Structure
- Technologies Used
- Future Improvements
- Author
- License

---

# 📌 Project Overview

Speech Emotion Recognition (SER) is an Artificial Intelligence application that identifies a speaker's emotional state from voice recordings.

This project combines the **RAVDESS** and **CREMA-D** datasets and applies a complete Machine Learning pipeline including preprocessing, feature extraction, model training, and deployment through a Streamlit web application.

Users can upload a `.wav` file and receive:

- 🎭 Predicted Emotion
- 🎯 Confidence Score
- 📊 Probability Distribution
- 📈 Interactive Visualization

---

# 🚀 Features

✅ Deep Learning-based Emotion Classification

✅ TensorFlow Neural Network

✅ Audio Feature Extraction using Librosa

✅ Audio Augmentation

✅ Interactive Streamlit Dashboard

✅ Plotly Visualization

✅ Upload WAV Audio

✅ Emotion Probability Graph

✅ Confidence Analysis

---

# 🎥 Demo

## Home Page

```
Add Screenshot Here
```

---

## Prediction Result

```
Add Screenshot Here
```

---

## Probability Distribution

```
Add Screenshot Here
```

---

# 📂 Dataset

| Dataset | Samples |
|----------|---------:|
| RAVDESS | 1,440 |
| CREMA-D | 7,442 |
| Total | 8,882 |

After Audio Augmentation

**35,528 Training Samples**

---

# 🎭 Emotion Classes

| Emotion |
|----------|
| Neutral |
| Calm |
| Happy |
| Sad |
| Angry |
| Fearful |
| Disgust |
| Surprised |

---

# ⚙️ Project Workflow

```
Speech Dataset
      │
      ▼
Audio Validation
      │
      ▼
Audio Augmentation
      │
      ▼
Feature Extraction
      │
      ▼
Feature Scaling
      │
      ▼
Model Training
      │
      ▼
Model Saving
      │
      ▼
Streamlit Deployment
```

---

# 📊 Feature Extraction

| Feature | Count |
|----------|-------:|
| MFCC | 40 |
| Chroma | 1 |
| Mel Spectrogram | 1 |
| Zero Crossing Rate | 1 |
| RMS Energy | 1 |

Total Features = **44**

---

# 🧠 Deep Learning Architecture

```
Input Layer (44)

↓

Dense (256)

↓

Batch Normalization

↓

Dropout

↓

Dense (128)

↓

Batch Normalization

↓

Dropout

↓

Dense (64)

↓

Dropout

↓

Softmax (8 Classes)
```

---

# 📈 Model Performance

| Metric | Value |
|----------|-------:|
| Test Accuracy | 58.13% |
| Emotion Classes | 8 |
| Training Samples | 35,528 |

---

# 📸 Screenshots

## 🏠 Home Page

```
assets/home.png
```

---

## 🎤 Prediction

```
assets/prediction.png
```

---

## 📊 Probability Distribution

```
assets/probability.png
```

---

# 📁 Folder Structure

```text
Speech-Emotion-Recognition/

app.py

README.md

requirements.txt

.gitignore

models/

outputs/

data/

assets/

src/
```

---

# 💻 Installation

Clone Repository

```bash
git clone https://github.com/Divyeshinturi/Speech-Emotion-Recognition.git
```

Move to Project

```bash
cd Speech-Emotion-Recognition
```

Install Dependencies

```bash
pip install -r requirements.txt
```

Run

```bash
python -m streamlit run app.py
```

---

# 🛠 Technologies Used

- Python
- TensorFlow
- Streamlit
- Librosa
- Plotly
- Pandas
- NumPy
- Scikit-learn
- Joblib

---

# 🔮 Future Improvements

- CNN + LSTM
- Wav2Vec2
- HuBERT
- Real-time Microphone Support
- Mobile Application
- REST API
- Cloud Deployment
- Higher Accuracy

---

# 👨‍💻 Author

## INTURI DIVYESH

Computer Science Engineering Student

Interested in

- Artificial Intelligence
- Machine Learning
- Deep Learning
- Data Science
- NLP
- Computer Vision

### GitHub

https://github.com/Divyeshinturi

### LinkedIn

(Add your LinkedIn URL)

---

# ⭐ Support

If you found this repository useful,

please consider giving it a ⭐ on GitHub.

---

# 📄 License

MIT License

---

<div align="center">

Made with ❤️ by **INTURI DIVYESH**

</div>