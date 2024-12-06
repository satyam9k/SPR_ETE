import streamlit as st
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
from speechbrain.pretrained import SepformerSeparation

# Load the SpeechBrain model
@st.cache_resource
def load_model():
    model = SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-wsj02mix"
    )
    return model

model = load_model()

# Function to plot waveform
def plot_waveform(waveform, sample_rate, title="Waveform"):
    plt.figure(figsize=(10, 4))
    plt.plot(np.linspace(0, len(waveform) / sample_rate, num=len(waveform)), waveform)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    st.pyplot(plt)

# Streamlit app layout
st.title("🎵 Speech Separator with Sepformer")
st.write("Upload an audio file, and this app will separate it into individual sources!")

# File uploader
uploaded_file = st.file_uploader("Upload an audio file (WAV format only):", type=["wav"])

if uploaded_file:
    # Save the uploaded file
    with open("uploaded_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.audio("uploaded_audio.wav", format="audio/wav", start_time=0)

    # Perform separation
    st.write("⏳ Processing... Please wait!")
    est_sources = model.separate_file(path="uploaded_audio.wav")

    # Save the separated audio files
    torchaudio.save("separated_01.wav", est_sources[:, :, 0].detach().cpu(), 8000)
    torchaudio.save("separated_02.wav", est_sources[:, :, 1].detach().cpu(), 8000)

    st.success("Speech separation completed! 🎉")

    # Display playback for separated audio
    st.subheader("Original Audio")
    st.audio("uploaded_audio.wav", format="audio/wav")

    st.subheader("Separated Sources")
    st.audio("separated_01.wav", format="audio/wav")
    st.audio("separated_02.wav", format="audio/wav")

    # Visualizations
    st.subheader("Waveform Visualizations")

    # Load audio and plot waveforms
    waveform_orig, sample_rate_orig = torchaudio.load("uploaded_audio.wav")
    waveform_1, sample_rate_1 = torchaudio.load("separated_01.wav")
    waveform_2, sample_rate_2 = torchaudio.load("separated_02.wav")

    st.write("**Original Audio**")
    plot_waveform(waveform_orig[0].numpy(), sample_rate_orig, title="Original Audio")

    st.write("**Separated Source 1**")
    plot_waveform(waveform_1[0].numpy(), sample_rate_1, title="Separated Source 1")

    st.write("**Separated Source 2**")
    plot_waveform(waveform_2[0].numpy(), sample_rate_2, title="Separated Source 2")

    # Option to download separated files
    st.subheader("Download Separated Audio")
    with open("separated_01.wav", "rb") as f:
        st.download_button(label="Download Source 1", data=f, file_name="separated_01.wav", mime="audio/wav")
    with open("separated_02.wav", "rb") as f:
        st.download_button(label="Download Source 2", data=f, file_name="separated_02.wav", mime="audio/wav")
