import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from scipy.io import wavfile
from scipy.fftpack import fft
from scipy.spatial.distance import cosine, euclidean, cdist
from librosa.sequence import dtw
import fastdtw
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import speech_recognition as sr
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Ngưỡng xác định âm thanh nào là actual sound
THRESHOLD = 0.02

# Load the audio files
def load_audio(file_path, duration=3):
    y, sr = librosa.load(file_path, duration=duration)
    return y, sr

# Detect the start and end of the actual sound
def detect_sound_boundaries(y, threshold=THRESHOLD):
    start_idx = np.argmax(np.abs(y) > threshold)
    end_idx = len(y) - np.argmax(np.abs(y[::-1]) > threshold)
    return start_idx, end_idx

# Center the audio by adding silence to the beginning and end as needed
def center_audio(y, sr, start_idx, end_idx, duration=3):
    sound = y[start_idx:end_idx]
    padding_len = int(sr * duration - len(sound))
    pad_before = padding_len // 2
    pad_after = padding_len - pad_before
    centered_audio = np.pad(sound, (pad_before, pad_after), 'constant')
    return centered_audio

# Normalize the amplitude of the audio signal
def normalize_amplitude(y):
    return y / np.max(np.abs(y))

# Divide the audio into bins
def divide_into_bins(y, sr, bin_size=0.5):
    bin_samples = int(bin_size * sr)
    num_bins = len(y) // bin_samples
    bins = [y[i*bin_samples:(i+1)*bin_samples] for i in range(num_bins)]
    return bins

# Extract MFCC features for each bin
def extract_mfcc_bins(bins, sr, n_mfcc=13):
    mfcc_bins = [librosa.feature.mfcc(y=bin, sr=sr, n_mfcc=n_mfcc).T for bin in bins]
    return mfcc_bins

# Compute DTW similarity between corresponding bins
def compute_dtw_similarity(mfcc_bins1, mfcc_bins2):
    distances = []
    for bin1, bin2 in zip(mfcc_bins1, mfcc_bins2):
        # Compute DTW distance
        distance, _ = fastdtw.fastdtw(bin1, bin2, dist=euclidean)
        distances.append(distance)
    return distances

# Plot similarity scores
def plot_similarity(similarities):
    fig = px.line(x=range(len(similarities)), y=similarities, labels={'x': 'Bin', 'y': 'DTW Distance'})
    fig.update_layout(title='DTW Distances Between Corresponding Bins')
    st.plotly_chart(fig)

# Recognize speech from audio file
def recognize_speech(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        text = ""
    except sr.RequestError:
        text = ""
    return text

# Main function
def similarity_calculate(file1, file2, bin_size=0.05):
    # Recognize speech from both audio files
    text1 = recognize_speech(file1)
    text2 = recognize_speech(file2)

    if text1 != text2:
        return 0.0
    
    audio1, sr1 = load_audio(file1)
    audio2, sr2 = load_audio(file2)

    start1, end1 = detect_sound_boundaries(audio1)
    start2, end2 = detect_sound_boundaries(audio2)

    centered_audio1 = center_audio(audio1, sr1, start1, end1)
    centered_audio2 = center_audio(audio2, sr2, start2, end2)

    norm_audio1 = normalize_amplitude(centered_audio1)
    norm_audio2 = normalize_amplitude(centered_audio2)

    bins1 = divide_into_bins(norm_audio1, sr1, bin_size)
    bins2 = divide_into_bins(norm_audio2, sr2, bin_size)

    mfcc_bins1 = extract_mfcc_bins(bins1, sr1)
    mfcc_bins2 = extract_mfcc_bins(bins2, sr2)

    dtw_distances = compute_dtw_similarity(mfcc_bins1, mfcc_bins2)
    mean_dtw_distance = np.mean(dtw_distances)
    return mean_dtw_distance

def remove_ones_and_compute_mean(arr):
    arr = np.array(arr)
    filtered_arr = arr[arr != 1]
    if len(filtered_arr) == 0:
        return np.nan
    mean_value = np.mean(filtered_arr)
    return mean_value
