import tempfile
import pandas as pd
import plotly.express as px
import streamlit as st

from src.predict import EmotionPredictor


st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

predictor = EmotionPredictor()


st.markdown("""
<style>

#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

header{
visibility:hidden;
}

.stApp{
background:#0f172a;
color:white;
}

.hero{
padding:35px;
border-radius:18px;
background:linear-gradient(135deg,#2563eb,#7c3aed);
text-align:center;
color:white;
margin-bottom:20px;
box-shadow:0 10px 30px rgba(0,0,0,.35);
}

.hero h1{
font-size:42px;
margin-bottom:10px;
}

.hero p{
font-size:18px;
opacity:.9;
}

.metric-card{
background:#1e293b;
padding:20px;
border-radius:15px;
text-align:center;
border:1px solid #334155;
box-shadow:0 5px 18px rgba(0,0,0,.25);
}

.metric-title{
font-size:15px;
color:#94a3b8;
}

.metric-value{
font-size:30px;
font-weight:bold;
margin-top:8px;
color:white;
}

.result-card{
background:#1e293b;
padding:25px;
border-radius:18px;
text-align:center;
border:1px solid #475569;
margin-top:15px;
}

.result-title{
font-size:18px;
color:#94a3b8;
}

.result-value{
font-size:42px;
font-weight:bold;
color:#22c55e;
}

.confidence{
font-size:32px;
font-weight:bold;
color:#38bdf8;
}

.upload-box{
padding:25px;
border-radius:18px;
background:#1e293b;
border:2px dashed #38bdf8;
margin-top:20px;
}

.sidebar-title{
font-size:24px;
font-weight:bold;
color:white;
}

.small-text{
color:#cbd5e1;
font-size:15px;
}

</style>
""", unsafe_allow_html=True)


with st.sidebar:

    st.markdown("# 🎤 Speech Emotion AI")

    st.markdown("---")

    st.markdown("### 📊 Project Information")

    st.write("**Model**")
    st.write("TensorFlow Deep Neural Network")

    st.write("**Dataset**")
    st.write("RAVDESS + CREMA-D")

    st.write("**Training Samples**")
    st.write("35,528")

    st.write("**Emotion Classes**")
    st.write("8")

    st.write("**Accuracy**")
    st.write("58.13%")

    st.markdown("---")

    st.success("Version 1.0")

    st.caption("Developed by Divyesh")


st.markdown("""
<div class="hero">

<h1>🎤 Speech Emotion Recognition</h1>

<p>
AI-powered Speech Emotion Analysis using Deep Learning
</p>

</div>
""", unsafe_allow_html=True)


c1, c2, c3, c4 = st.columns(4)

with c1:

    st.markdown("""
    <div class="metric-card">
    <div class="metric-title">Dataset</div>
    <div class="metric-value">35,528</div>
    </div>
    """, unsafe_allow_html=True)

with c2:

    st.markdown("""
    <div class="metric-card">
    <div class="metric-title">Classes</div>
    <div class="metric-value">8</div>
    </div>
    """, unsafe_allow_html=True)

with c3:

    st.markdown("""
    <div class="metric-card">
    <div class="metric-title">Accuracy</div>
    <div class="metric-value">58.13%</div>
    </div>
    """, unsafe_allow_html=True)

with c4:

    st.markdown("""
    <div class="metric-card">
    <div class="metric-title">Framework</div>
    <div class="metric-value">TensorFlow</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div class="upload-box">

<h3>📂 Upload Audio File</h3>

<p class="small-text">
Supported format: <b>.wav</b>
</p>

</div>
""", unsafe_allow_html=True)


uploaded_file = st.file_uploader(
    "",
    type=["wav"],
    help="Upload a WAV audio file for emotion prediction."
)


if uploaded_file is not None:

    left, right = st.columns([1, 1])

    with left:

        st.subheader("🎵 Uploaded Audio")

        st.audio(
            uploaded_file,
            format="audio/wav"
        )

        st.info(
            f"""
Filename: **{uploaded_file.name}**

Size: **{uploaded_file.size / 1024:.2f} KB**
"""
        )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as temp_file:

        temp_file.write(uploaded_file.read())

        temp_path = temp_file.name

    with right:

        st.subheader("🤖 AI Prediction")

        st.write(
            """
Click the button below to analyze the uploaded speech
and identify the speaker's emotion.
"""
        )

        predict = st.button(
            "🚀 Predict Emotion",
            use_container_width=True
        )

    if predict:

        with st.spinner(
            "🧠 AI is analyzing the speech..."
        ):

            result = predictor.predict(
                temp_path
            )

        emotion = result["emotion"].capitalize()

        confidence = result["confidence"] * 100

        probabilities = result["probabilities"]

        emotion_names = predictor.encoder.classes_

        st.success("Prediction completed successfully!")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(f"""
<div class="result-card">

<div class="result-title">
Predicted Emotion
</div>

<div class="result-value">
{emotion}
</div>

</div>
""", unsafe_allow_html=True)

        with col2:

            st.markdown(f"""
<div class="result-card">

<div class="result-title">
Confidence
</div>

<div class="confidence">
{confidence:.2f}%
</div>

</div>
""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        st.subheader("📊 Emotion Probability Distribution")

        probability_df = pd.DataFrame({
            "Emotion": [
                emotion.capitalize()
                for emotion in emotion_names
            ],
            "Probability": [
                round(prob * 100, 2)
                for prob in probabilities
            ]
        })

        probability_df = probability_df.sort_values(
            by="Probability",
            ascending=False
        )

        fig = px.bar(
            probability_df,
            x="Emotion",
            y="Probability",
            text="Probability",
            color="Probability",
            color_continuous_scale="Blues",
            height=450
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        fig.update_layout(
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font=dict(
                color="white",
                size=14
            ),
            xaxis_title="Emotion",
            yaxis_title="Confidence (%)",
            coloraxis_showscale=False,
            margin=dict(
                l=20,
                r=20,
                t=40,
                b=20
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.markdown("---")

        st.subheader("📋 Detailed Prediction")

        display_df = probability_df.copy()

        display_df["Probability"] = (
            display_df["Probability"]
            .astype(str)
            + "%"
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        st.subheader("🎯 Confidence Analysis")

        highest = probability_df.iloc[0]["Probability"]

        if highest >= 80:

            st.success(
                "The model is highly confident about this prediction."
            )

        elif highest >= 60:

            st.info(
                "The model is reasonably confident."
            )

        elif highest >= 40:

            st.warning(
                "The prediction has moderate confidence. Consider using a clearer recording."
            )

        else:

            st.error(
                "The model has low confidence. The audio may be noisy or ambiguous."
            ) 
        st.markdown("<br>", unsafe_allow_html=True)

        left, right = st.columns([2, 1])

        with left:

            st.subheader("🧠 About the AI Model")

            st.info("""
This application uses a **Deep Neural Network (TensorFlow/Keras)** trained on the
**RAVDESS** and **CREMA-D** speech emotion datasets.

The model analyzes acoustic speech features including:

• MFCC (40 Features)

• Chroma Feature

• Mel Spectrogram

• Zero Crossing Rate

• RMS Energy

Total Training Samples: **35,528**
""")

        with right:

            st.subheader("📌 Model Information")

            st.metric(
                "Framework",
                "TensorFlow"
            )

            st.metric(
                "Training Samples",
                "35,528"
            )

            st.metric(
                "Emotion Classes",
                "8"
            )

            st.metric(
                "Test Accuracy",
                "58.13%"
            )

        st.markdown("---")

        st.subheader("🚀 Future Improvements")

        st.markdown("""
- Improve model accuracy using CNN + LSTM architecture
- Fine-tune Wav2Vec2 / HuBERT models
- Support real-time microphone emotion detection
- Add multilingual speech emotion recognition
- Improve robustness for noisy environments
- Increase dataset diversity for better generalization
""")

        st.markdown("---")

        st.markdown("""
<div style="
background: linear-gradient(90deg,#2563eb,#7c3aed);
padding:20px;
border-radius:15px;
text-align:center;
color:white;
">

<h3>🎤 Speech Emotion Recognition</h3>

<p>
Built with ❤️ using <b>TensorFlow</b>, <b>Streamlit</b>, <b>Librosa</b>, and <b>Plotly</b>
</p>

<p>
Dataset: <b>RAVDESS + CREMA-D</b>
</p>

<p>
Version 1.0
</p>

</div>
""", unsafe_allow_html=True)           