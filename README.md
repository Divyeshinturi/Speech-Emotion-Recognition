# 🎙️ Speech Emotion Recognition using Deep Learning

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)
![Deep Learning](https://img.shields.io/badge/AI-Deep%20Learning-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

</p>

---

## 📌 Project Overview

Speech Emotion Recognition (SER) is a Deep Learning application that predicts the emotional state of a speaker from their voice.

The system extracts meaningful audio features using **Librosa**, processes them through a trained **TensorFlow Deep Learning model**, and predicts one of several human emotions.

The project is deployed online using **Render** with an interactive **Streamlit** interface.

---

## 🚀 Live Demo

**🌐 Live Application**

https://speech-emotion-recognition-90x5.onrender.com

---

## 📂 GitHub Repository

https://github.com/Divyeshinturi/Speech-Emotion-Recognition

---

# ✨ Features

- 🎤 Upload speech/audio files
- 🎧 Automatic audio preprocessing
- 📊 MFCC Feature Extraction
- 🧠 Deep Learning based prediction
- 😊 Emotion Classification
- 📈 Interactive Streamlit Dashboard
- 📊 Emotion Probability Visualization
- 🌐 Online Deployment with Render

---

# 🧠 Emotions Supported

- Angry 😠
- Calm 😌
- Disgust 🤢
- Fear 😨
- Happy 😀
- Neutral 😐
- Sad 😢
- Surprise 😲

---

# 🏗️ Project Architecture

```
Audio Input
      │
      ▼
Audio Validation
      │
      ▼
Feature Extraction (MFCC)
      │
      ▼
Preprocessing
      │
      ▼
TensorFlow Deep Learning Model
      │
      ▼
Emotion Prediction
      │
      ▼
Streamlit Dashboard
```

---

# 🛠️ Tech Stack

### Programming Language

- Python 3.11

### Deep Learning

- TensorFlow
- Keras

### Machine Learning

- Scikit-learn

### Audio Processing

- Librosa
- SoundFile

### Data Processing

- NumPy
- Pandas

### Visualization

- Plotly
- Matplotlib

### Deployment

- Streamlit
- Render

### Version Control

- Git
- GitHub

---

# 📁 Project Structure

```
Speech-Emotion-Recognition
│
├── models/
│   ├── emotion_model.keras
│   ├── emotion_model.pkl
│   ├── history.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
├── src/
│   ├── feature_extraction.py
│   ├── preprocess.py
│   ├── predict.py
│   ├── dataset_loader.py
│   ├── train_model.py
│   ├── evaluate.py
│   └── utils.py
│
├── app.py
├── requirements.txt
├── runtime.txt
└── README.md
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/Divyeshinturi/Speech-Emotion-Recognition.git
```

Go to the project folder

```bash
cd Speech-Emotion-Recognition
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# 📊 Workflow

1. Upload Speech Audio
2. Validate Audio
3. Extract MFCC Features
4. Preprocess Data
5. Load Trained Model
6. Predict Emotion
7. Display Prediction
8. Visualize Results

---

# 🎯 Applications

- Mental Health Monitoring
- Customer Service Analytics
- Virtual Assistants
- Human-Computer Interaction
- Call Center Analysis
- Healthcare
- Smart Voice Systems
- Educational Research

---

# 📈 Future Improvements

- Live Microphone Recording
- Transformer-based Speech Models
- Multilingual Emotion Recognition
- Speech-to-Text Integration
- Real-time Emotion Detection
- Model Performance Dashboard
- Cloud Storage Support

---

# 👨‍💻 Author

**INTURI DIVYESH**

Computer Science Engineering Student

Sathyabama Institute of Science and Technology

GitHub:
https://github.com/Divyeshinturi

LinkedIn:
(Add your LinkedIn profile link here)

---

# ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.

It helps others discover the project and motivates future improvements.

---

# 📜 License

This project is intended for educational and research purposes.
