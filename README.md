# 🎙️ Speech Emotion Recognition using Deep Learning

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)
![Deep Learning](https://img.shields.io/badge/AI-Deep%20Learning-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

</p>

---

# 📌 Project Overview

Speech Emotion Recognition (SER) is a Deep Learning application that predicts the emotional state of a speaker from their voice.

The system extracts meaningful audio features using **Librosa**, processes them through a trained **TensorFlow/Keras Deep Learning model**, and predicts one of **eight human emotions**.

The application features an interactive **Streamlit** interface and is deployed online using **Render**.

---

# 🚀 Live Demo

### 🌐 Live Application

https://speech-emotion-recognition-90x5.onrender.com/

---

# 📂 GitHub Repository

https://github.com/Divyeshinturi/Speech-Emotion-Recognition

---

# ✨ Features

- 🎤 Upload WAV audio files
- 🎧 Automatic speech preprocessing
- 📊 MFCC Feature Extraction
- 🧠 TensorFlow Deep Learning Model
- 😊 8-Class Emotion Classification
- 📈 Emotion Probability Visualization
- 🎨 Interactive Streamlit Dashboard
- 🌐 Public Deployment using Render

---

# 🧠 Supported Emotions

- 😠 Angry
- 😌 Calm
- 🤢 Disgust
- 😨 Fearful
- 😀 Happy
- 😐 Neutral
- 😢 Sad
- 😲 Surprised

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
Feature Scaling
      │
      ▼
TensorFlow Deep Learning Model
      │
      ▼
Emotion Prediction
      │
      ▼
Probability Visualization
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
│   ├── history.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
├── src/
│   ├── feature_extraction.py
│   ├── preprocess.py
│   ├── predict.py
│   ├── train_model.py
│   ├── evaluate.py
│   ├── dataset_loader.py
│   ├── audio_validator.py
│   ├── augmentation.py
│   └── utils.py
│
├── app.py
├── requirements.txt
├── runtime.txt
├── .python-version
├── README.md
└── .gitignore
```

---

# ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/Divyeshinturi/Speech-Emotion-Recognition.git
```

### Navigate to the project

```bash
cd Speech-Emotion-Recognition
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

---

# 📊 Workflow

1. Upload a WAV audio file
2. Validate audio quality
3. Extract MFCC features
4. Scale extracted features
5. Load the trained TensorFlow model
6. Predict the emotion
7. Display confidence scores
8. Visualize emotion probabilities

---

# 📈 Results

- ✅ Emotion recognition using Deep Learning
- ✅ 8 emotion classes supported
- ✅ TensorFlow-based neural network
- ✅ Interactive probability visualization
- ✅ Publicly deployed using Render
- ✅ User-friendly Streamlit interface

---

# 🎯 Applications

- Mental Health Monitoring
- Human-Computer Interaction
- Customer Service Analytics
- Voice-based AI Systems
- Virtual Assistants
- Healthcare
- Call Center Analytics
- Educational Research

---

# 📈 Future Improvements

- 🎙️ Live microphone emotion recognition
- 🌍 Multilingual speech emotion recognition
- 🤖 Transformer-based speech models
- 📱 Mobile-friendly interface
- 📊 Model analytics dashboard
- ☁️ Cloud storage support
- 🔄 Real-time emotion detection

---

# 📸 Screenshots

> Add screenshots of your application here.

Example:

```
screenshots/
├── home.png
├── upload.png
├── prediction.png
└── chart.png
```

---

# 👨‍💻 Author

## INTURI DIVYESH

Computer Science Engineering Student

Sathyabama Institute of Science and Technology

### GitHub

https://github.com/Divyeshinturi

### Live Demo

https://speech-emotion-recognition-90x5.onrender.com/

---

# ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.

It motivates future improvements and helps others discover the project.

---

# 📜 License

This project is intended for educational and research purposes.

© 2026 INTURI DIVYESH
