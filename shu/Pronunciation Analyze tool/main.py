import streamlit as st
import sounddevice as sd
import wavio
import os
import shutil
import task1
import task2
import numpy as np
import librosa.display
from matplotlib import pyplot as plt


# cd myenv/Scripts
# activate
# cd ..
# cd ..
# streamlit run main.py


RECORD_DURATION = 3
SAMPLE_RATE = 44100
AUDIO_DIR = "recordings"
AUDIO_PATH = os.path.join(AUDIO_DIR, "recording.wav")

# Tạo thư mục nếu chưa tồn tại
if not os.path.exists(AUDIO_DIR):
    os.makedirs(AUDIO_DIR)

# Hàm để ghi âm
def record_audio():
    st.write("Recording...")
    recording = sd.rec(int(RECORD_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=2)
    sd.wait()
    wavio.write(AUDIO_PATH, recording, SAMPLE_RATE, sampwidth=2)
    st.write("Complete!")
    return recording


# Hàm để xóa file âm thanh
def delete_audio():
    if os.path.exists(AUDIO_PATH):
        os.remove(AUDIO_PATH)
        st.success("Complete delete!")
    else:
        st.success("Not found!")




st.title("Task 1: Comparison")



# Nút chọn file âm thanh chuẩn
if 'show_file_uploader' not in st.session_state:
    st.session_state.show_file_uploader = False

if st.button("Choose audio file for standard"):
    st.session_state.show_file_uploader = not st.session_state.show_file_uploader

if st.session_state.show_file_uploader:
    uploaded_file1 = st.file_uploader("Select standard audio", type=["wav", "mp3", "aac", "flac", "ogg", "wma"])
    if uploaded_file1 is not None:
        st.session_state.uploaded_file1 = uploaded_file1
        st.write("Done:")
        st.audio(st.session_state.uploaded_file1, format='audio/wav')

        # Lưu file vào thư mục tạm thời và in ra đường dẫn
        with open(f"temp/{uploaded_file1.name}", "wb") as f:
            f.write(uploaded_file1.getbuffer())
        st.success(f"File uploaded successfully! File path: temp/{uploaded_file1.name}")    
        file1 = f"temp/{uploaded_file1.name}"






# Nút chính để hiển thị các lựa chọn cho file âm thanh thứ hai
if 'show_second_audio_buttons' not in st.session_state:
    st.session_state.show_second_audio_buttons = False

if st.button("Choose audio file to compare"):
    st.session_state.show_second_audio_buttons = not st.session_state.show_second_audio_buttons

if st.session_state.show_second_audio_buttons:
    uploaded_file2 = st.file_uploader("Select second audio", type=["wav", "mp3", "aac", "flac", "ogg", "wma"])
    if uploaded_file2 is not None:
        st.session_state.uploaded_file2 = uploaded_file2
        st.write("Done:")
        st.audio(st.session_state.uploaded_file2, format='audio/wav')
        
        # Lưu file vào thư mục tạm thời và in ra đường dẫn
        with open(f"temp/{uploaded_file2.name}", "wb") as f:
            f.write(uploaded_file2.getbuffer())
        st.success(f"File uploaded successfully! File path: temp/{uploaded_file2.name}")    
        file2 = f"temp/{uploaded_file2.name}"






# Nút chính để hiển thị các lựa chọn cho recording
if 'recording' not in st.session_state:
    st.session_state.recording = False

if st.button("Choose Recording Audio"):
    st.session_state.recording = not st.session_state.recording


if st.session_state.recording:
    col1, col2,_, = st.columns(3)
    with col2:
        # Nút ghi âm file âm thanh thứ hai
        if st.button("Recording"):
            uploaded_file2 = record_audio()
            st.write("Success")
            st.audio(AUDIO_PATH, format='audio/wav')
            file2 = f'recordings/recording.wav'
            

        # Nút xóa file âm thanh đã record
        if st.button("Delete"):
            delete_audio()




# Đường dẫn đến thư mục cần xóa các file
folder_path = 'temp/'

def delete_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            st.error(f'Failed to delete {file_path}. Reason: {e}')



if st.button('Clear Temp Folder'):
    delete_files_in_folder(folder_path)
    st.success('Clear temp successfully!')



def check_recording_file():
    folder_path = 'recordings'
    file_path = os.path.join(folder_path, 'recording.wav')
    
    if not os.path.exists(file_path):
        pass
    else:
        file2 = file_path
        return file2



if st.button('Compare'):
    #st.write(file1)
    #st.write(file2)
    file2 = check_recording_file()
    sim_score = task1.similarity_calculate(file1, file2, bin_size=0.05)
    fig = task1.plot_similarity(sim_score)
    st.pyplot(fig)
    a = task1.remove_ones_and_compute_mean(sim_score)
    a_percent = a*100
    st.write(f"Similarity score: {a_percent:.2f}%")


    # Thresholds
    high_threshold = 0.9
    low_threshold = 0.8

    # Kiểm tra và in kết quả
    if a > high_threshold:
        st.write("High similarity - Good Mimic")
    elif a < low_threshold:
        st.write("Low similarity")
    else:
        st.write("Normal similarity")







st.title("Task 2: Vowel Recognize")

# Nút chọn file âm thanh chuẩn
if 'task2' not in st.session_state:
    st.session_state.task2 = False

if st.button("Select Audio File"):
    st.session_state.task2 = not st.session_state.task2

if st.session_state.task2:
    uploaded_file1 = st.file_uploader("Select correct audio", type=["wav", "mp3", "aac", "flac", "ogg", "wma"])
    if uploaded_file1 is not None:
        st.session_state.uploaded_file1 = uploaded_file1
        st.write("Đã chọn file âm thanh:")
        st.audio(st.session_state.uploaded_file1, format='audio/wav')

        # Lưu file vào thư mục tạm thời và in ra đường dẫn
        with open(f"temp/{uploaded_file1.name}", "wb") as f:
            f.write(uploaded_file1.getbuffer())
        st.success(f"File uploaded successfully! File path: temp/{uploaded_file1.name}")    
        task2_file = f"temp/{uploaded_file1.name}"


if st.button('Detect Vowel'):
    path = task2_file
    phones = task2.phone_recognize_file(path)
    phones.head()
    hop_length = 512

    spectrogram = task2.load_spectrogram(path, hop_length=hop_length, fmax=4000)
    fig, (spec_ax, label_ax) = plt.subplots(2, figsize=(20, 8), sharex=True)
    librosa.display.specshow(spectrogram.T.values, ax=spec_ax, sr=16000, fmax=4000, hop_length=hop_length, x_axis='time')

    st.pyplot(fig)
    task2.plot_events(label_ax, phones, color='color', annotate='label')




# cd myenv\Scripts
# activate