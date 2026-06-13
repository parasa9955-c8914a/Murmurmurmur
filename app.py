import streamlit as st
import torch
from transformers import AutoProcessor, AutoModelForAudioClassification
import librosa

MODEL_ID = "ParasaPicha/heart-sound-classifier-astbase04"

@st.cache_resource
def load_model():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
    model.eval()
    return processor, model

processor, model = load_model()

st.set_page_config(page_title="Murmur 2026-3", layout="centered")

st.title("Heart Sound Classification version 3.0")
st.write("This app is a starter scaffold for deploying a Hugging Face heart sound classification model.")

uploaded_file = st.file_uploader(
    "Choose a WAV file",
    type=["wav"]
)

if uploaded_file is not None:

    st.subheader("Uploaded Audio")

    st.audio(uploaded_file)

    with st.spinner("Loading audio..."):

        y, sr = librosa.load(
            uploaded_file,
            sr=16000
        )

    st.write(f"Sampling Rate : {sr}")
    st.write(f"Duration : {len(y)/sr:.2f} seconds")

    with st.spinner("Running prediction..."):

        inputs = processor(
            y,
            sampling_rate=16000,
            return_tensors="pt"
        )

        with torch.no_grad():

            outputs = model(**inputs)

            pred_id = torch.argmax(
                outputs.logits,
                dim=-1
            ).item()

            probs = torch.softmax(
                outputs.logits,
                dim=-1
            )[0]

    label_map = {
    0: "Murmur",
    1: "Non-Murmur"}

    predicted_label = label_map[pred_id]

    st.success(
        f"Prediction : {predicted_label}"
    )

    st.subheader("Confidence Scores")

    for idx, score in enumerate(probs):

        st.write(
            f"{label_map[idx]}: {score.item()*100:.2f}%"
        )
