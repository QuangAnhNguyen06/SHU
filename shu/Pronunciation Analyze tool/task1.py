import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from scipy.io import wavfile
from scipy.fftpack import fft
from scipy.spatial.distance import cosine, euclidean, cdist
#from librosa.feature import mfcc
from librosa.sequence import dtw
import fastdtw
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import speech_recognition as sr
import streamlit as st



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

# Compute cosine similarity between corresponding bins
def compute_bin_similarity(mfcc_bins1, mfcc_bins2):
    similarities = []
    for bin1, bin2 in zip(mfcc_bins1, mfcc_bins2):
        # Compute mean MFCC for each bin
        mean_mfcc1 = np.mean(bin1, axis=0).reshape(1, -1)
        mean_mfcc2 = np.mean(bin2, axis=0).reshape(1, -1)
        # Compute cosine similarity
        similarity = cosine_similarity(mean_mfcc1, mean_mfcc2)[0, 0]
        similarities.append(similarity)
    return similarities

# Plot similarity scores
def plot_similarity(similarities):
    # plt.figure(figsize=(10, 4))
    # plt.bar(range(len(similarities)), similarities)
    # plt.xlabel('Bin')
    # plt.ylabel('Cosine Similarity')
    # plt.title('Cosine Similarity Between Corresponding Bins')
    # plt.show()

    # Plot the peaks as a line chart
    plt.figure(figsize=(10, 4))
    plt.plot(similarities, label='Cosine Similarity')
    plt.xlabel('Bin')
    plt.ylabel('Cosine Similarity')
    plt.title('Peaks in Cosine Similarity')
    plt.legend()
    plt.show()
    return plt
 


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

    similarities = compute_bin_similarity(mfcc_bins1, mfcc_bins2)
    return similarities


def remove_ones_and_compute_mean(arr):
    # Convert the input to a numpy array if it is not already
    arr = np.array(arr)

    # Remove all occurrences of 1
    filtered_arr = arr[arr != 1]

    # Check if there are any values left after removing 1
    if len(filtered_arr) == 0:
        return np.nan  # Return NaN if no values are left

    # Compute the mean of the remaining values
    mean_value = np.mean(filtered_arr)
    return mean_value





# VOWEL AND CONSONANT

import streamlit as st
import allosaurus.app
import panphon
import pandas as pd
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def parse_timestamp_output(output):
    records = []
    lines = output.split('\n')
    for line in lines:
        if line.strip():
            tok = line.split(' ')
            start = float(tok[0])
            duration = float(tok[1])
            end = start + duration
            label = tok[2]
            records.append(dict(start=start, end=end, label=label))
    df = pd.DataFrame.from_records(records)
    return df

def is_consonant(ipa):
    ft = panphon.FeatureTable()
    cons = ft.word_fts(ipa)[0] >= {'cons': 1}
    return cons

def phone_recognize_file(path, emit=1.2, lang='eng'):
    model = allosaurus.app.read_recognizer()
    out = model.recognize(path, lang, timestamp=True, emit=emit)
    phones = parse_timestamp_output(out)
    phones['consonant'] = phones['label'].apply(is_consonant)
    phones['color'] = phones['consonant'].replace({True: 'green', False: 'red'})
    return phones

def load_spectrogram(path, hop_length=1024, sr=16000, n_mels=64, ref=np.max, **kwargs):
    y, sr = librosa.load(path, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, **kwargs)
    S_db = librosa.power_to_db(S, ref=ref)
    tt = np.arange(0, S_db.shape[1]) * (hop_length / sr)
    df = pd.DataFrame(S_db.T, index=pd.Series(tt, name='time'))
    return df



# def plot_events(ax, df, start='start', end='end', color=None, annotate=None,
#                 label=None, alpha=0.2, zorder=-1,
#                 text_kwargs={}, **kwargs):
#     import itertools
#     palette = itertools.cycle(plt.cm.tab10.colors)

#     def valid_time(dt):
#         return not pd.isnull(dt)

#     for idx, row in df.iterrows():
#         s = row[start]
#         e = row[end]

#         if color is None:
#             c = next(palette)
#         else:
#             c = row[color]

#         if valid_time(s) and valid_time(e):
#             ax.axvspan(s, e, label=label, color=c, alpha=alpha, zorder=zorder)
#         if valid_time(e):
#             ax.axvline(e, label=label, color=c, alpha=alpha, zorder=zorder)
#         if valid_time(s):
#             ax.axvline(s, label=label, color=c, alpha=alpha, zorder=zorder)

#         import matplotlib.transforms
#         trans = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
#         if annotate is not None:
#             ax.text(s, 1.05, row[annotate], transform=trans, **text_kwargs)

def plot_events(ax, df, start='start', end='end', color=None, annotate=None,
                label=None, alpha=0.2, zorder=-1,
                text_kwargs={}, **kwargs):
    import itertools
    palette = itertools.cycle(plt.cm.tab10.colors)

    def valid_time(dt):
        return not pd.isnull(dt)

    for idx, row in df.iterrows():
        s = row[start]
        e = row[end]

        if color is None:
            c = next(palette)
        else:
            c = row[color]

        if valid_time(s) and valid_time(e):
            ax.axvspan(s, e, label=label, color=c, alpha=alpha, zorder=zorder)
        if valid_time(e):
            ax.axvline(e, label=label, color=c, alpha=alpha, zorder=zorder)
        if valid_time(s):
            ax.axvline(s, label=label, color=c, alpha=alpha, zorder=zorder)

        import matplotlib.transforms
        trans = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
        if annotate is not None:
            ax.text(s, 1.05, row[annotate], transform=trans, **text_kwargs)