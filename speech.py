import streamlit as st
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import io
import os
import tempfile
import traceback
from speechbrain.pretrained import SepformerSeparation

# Set page configuration
st.set_page_config(page_title="Speech Separation App", layout="wide")

# Cache the model loading to improve performance
@st.cache_resource
def load_separation_model():
    return SepformerSeparation.from_hparams("speechbrain/sepformer-wsj02mix")

# Function to safely read audio file
def safe_read_audio(uploaded_file):
    try:
        # Create a temporary file to ensure proper file handling
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name
        
        # Read the audio file
        signal, sample_rate = sf.read(temp_file_path)
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        return signal, sample_rate
    except Exception as e:
        st.error(f"Error reading audio file: {str(e)}")
        st.error(f"Detailed traceback: {traceback.format_exc()}")
        return None, None

# Function to separate speech
def separate_speech(uploaded_file):
    # Load model
    model = load_separation_model()
    
    # Create a temporary file to ensure proper file handling
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name
    
    try:
        # Separate the file
        est_sources = model.separate_file(temp_file_path)
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        return est_sources
    except Exception as e:
        st.error(f"Error in speech separation: {str(e)}")
        st.error(f"Detailed traceback: {traceback.format_exc()}")
        # Clean up temporary file in case of error
        os.unlink(temp_file_path)
        return None

# Function to visualize waveforms
def plot_waveforms(original_signal, separated_signals):
    # Create a figure with subplots
    fig, axs = plt.subplots(len(separated_signals) + 1, 1, figsize=(10, 3*(len(separated_signals) + 1)))
    
    # Plot original signal
    axs[0].plot(original_signal)
    axs[0].set_title('Original Signal')
    axs[0].set_xlabel('Sample')
    axs[0].set_ylabel('Amplitude')
    
    # Plot separated signals
    for i, signal in enumerate(separated_signals, 1):
        axs[i].plot(signal)
        axs[i].set_title(f'Separated Signal {i}')
        axs[i].set_xlabel('Sample')
        axs[i].set_ylabel('Amplitude')
    
    plt.tight_layout()
    return fig

# Function to create spectrogram
def plot_spectrogram(signal, title):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        librosa.amplitude_to_db(np.abs(librosa.stft(signal)), ref=np.max), 
        sr=16000, 
        y_axis='hz', 
        x_axis='time'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    return plt.gcf()

# Streamlit App
def main():
    st.title("Speech Separation App 🎙️🔊")
    
    # Sidebar for navigation
    st.sidebar.header("Speech Separation Toolkit")
    
    # Diagnostic Information Section
    st.sidebar.subheader("Diagnostic Information")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a WAV file", 
        type=['wav'], 
        help="Upload a stereo or multi-speaker WAV file for separation"
    )
    
    if uploaded_file is not None:
        # Diagnostic Information
        st.sidebar.write("File Details:")
        st.sidebar.write(f"Filename: {uploaded_file.name}")
        st.sidebar.write(f"File Size: {uploaded_file.size} bytes")
        st.sidebar.write(f"File Type: {uploaded_file.type}")
        
        # Read the uploaded file
        original_signal, sample_rate = safe_read_audio(uploaded_file)
        
        if original_signal is not None:
            # Diagnostic Information
            st.sidebar.write("Audio File Info:")
            st.sidebar.write(f"Sample Rate: {sample_rate} Hz")
            st.sidebar.write(f"Signal Length: {len(original_signal)} samples")
            
            # Separate speech
            st.write("🔄 Separating Speech...")
            try:
                separated_signals = separate_speech(uploaded_file)
                
                if separated_signals is not None:
                    # Success message
                    st.success("Speech Separation Complete! 🎉")
                    
                    # Visualization Tabs
                    tab1, tab2, tab3 = st.tabs(
                        ["Waveform Comparison", "Spectrograms", "Signal Analysis"]
                    )
                    
                    with tab1:
                        st.header("Waveform Comparison")
                        
                        # Waveform plot
                        waveform_fig = plot_waveforms(
                            original_signal, 
                            [sig.numpy().flatten() for sig in separated_signals]
                        )
                        st.pyplot(waveform_fig)
                    
                    with tab2:
                        st.header("Spectrograms")
                        
                        # Create spectrograms for original and separated signals
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write("Original Signal Spectrogram")
                            orig_spec = plot_spectrogram(original_signal, "Original Signal")
                            st.pyplot(orig_spec)
                        
                        for i, sig in enumerate(separated_signals, 1):
                            with col1 if i == 1 else col2 if i == 2 else col3:
                                st.write(f"Separated Signal {i} Spectrogram")
                                sep_spec = plot_spectrogram(
                                    sig.numpy().flatten(), 
                                    f"Separated Signal {i}"
                                )
                                st.pyplot(sep_spec)
                    
                    with tab3:
                        st.header("Signal Analysis")
                        
                        # Signal characteristics
                        cols = st.columns(len(separated_signals) + 1)
                        
                        # Original signal analysis
                        with cols[0]:
                            st.subheader("Original Signal")
                            st.metric("Duration (s)", f"{len(original_signal)/sample_rate:.2f}")
                            st.metric("Peak Amplitude", f"{np.max(np.abs(original_signal)):.4f}")
                        
                        # Separated signals analysis
                        for i, sig in enumerate(separated_signals, 1):
                            signal = sig.numpy().flatten()
                            with cols[i]:
                                st.subheader(f"Separated Signal {i}")
                                st.metric("Duration (s)", f"{len(signal)/sample_rate:.2f}")
                                st.metric("Peak Amplitude", f"{np.max(np.abs(signal)):.4f}")
                    
                    # Playback and Download Section
                    st.header("Playback & Download")
                    
                    # Audio Playback
                    st.subheader("Original Signal")
                    st.audio(uploaded_file, format='audio/wav')
                    
                    # Separated Signal Playback
                    for i, sig in enumerate(separated_signals, 1):
                        st.subheader(f"Separated Signal {i}")
                        
                        # Convert tensor to numpy and write to temporary file
                        signal = sig.numpy().flatten()
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav', mode='wb') as temp_file:
                            sf.write(temp_file.name, signal, sample_rate)
                            
                            # Audio playback
                            st.audio(temp_file.name, format='audio/wav')
                            
                            # Download button
                            with open(temp_file.name, "rb") as file:
                                st.download_button(
                                    label=f"Download Separated Signal {i}",
                                    data=file,
                                    file_name=f"separated_signal_{i}.wav",
                                    mime="audio/wav"
                                )
                        
                        # Clean up temporary file
                        os.unlink(temp_file.name)
                else:
                    st.error("Failed to separate speech signals.")
            
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")
                st.error(f"Detailed traceback: {traceback.format_exc()}")
        else:
            st.error("Failed to read the audio file. Please check the file format and try again.")

if __name__ == "__main__":
    main()
