import streamlit as st
import torchaudio
from speechbrain.inference.separation import SepformerSeparation

# Load the Sepformer Separation model
@st.cache_resource
def load_model():
    return SepformerSeparation.from_hparams(source="speechbrain/sepformer-wsj02mix")

model = load_model()

# App Title
st.title("🎙️ Speech Separation App")

# Upload Audio File
uploaded_file = st.file_uploader("Upload an Audio File (WAV format)", type=["wav"])

if uploaded_file:
    # Save uploaded file locally
    with open("uploaded_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.audio("uploaded_audio.wav", format="audio/wav", start_time=0, caption="Original Audio")
    
    # Perform separation
    st.write("⏳ Processing... This may take some time.")
    try:
        est_sources = model.separate_file(path="uploaded_audio.wav")
        
        # Save separated audio files
        torchaudio.save("separated_01.wav", est_sources[:, :, 0].detach().cpu(), 8000)
        torchaudio.save("separated_02.wav", est_sources[:, :, 1].detach().cpu(), 8000)

        st.success("🎉 Separation completed!")

        # Playback separated audio
        st.subheader("Separated Audio Files:")
        st.audio("separated_01.wav", format="audio/wav", start_time=0, caption="Separated Source 1")
        st.audio("separated_02.wav", format="audio/wav", start_time=0, caption="Separated Source 2")

        # Download options
        with open("separated_01.wav", "rb") as f:
            st.download_button(label="Download Separated Source 1", data=f, file_name="separated_01.wav")
        with open("separated_02.wav", "rb") as f:
            st.download_button(label="Download Separated Source 2", data=f, file_name="separated_02.wav")

    except Exception as e:
        st.error(f"An error occurred: {e}")
