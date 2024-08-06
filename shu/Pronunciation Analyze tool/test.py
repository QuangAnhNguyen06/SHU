import streamlit as st
import sounddevice as sd
import wavio
import os
import shutil
import task1
import task2
import test_dtw
import numpy as np
import librosa.display
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import time
import speech_recognition as sr
import plotly.express as px


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

# Đường dẫn đến thư mục cần xóa các file
folder_path = 'temp/'
recordings_folder_path = 'recordings/'


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

# Hàm để kiểm tra file ghi âm
def check_recording_file():
    folder_path = 'recordings'
    file_path = os.path.join(folder_path, 'recording.wav')
    
    if not os.path.exists(file_path):
        return None
    else:
        return file_path

# PLOT
# Hàm để hiển thị biểu đồ đồng hồ
def plot_score(result):
    st.title("Scoring")

    # Chuyển đổi kết quả thành phần trăm
    percentage = result * 100

    placeholder = st.empty()  # Tạo một placeholder để cập nhật biểu đồ

    # Vòng lặp để tăng dần giá trị hiển thị
    for value in range(0, int(percentage) + 1):
        # Tạo biểu đồ đồng hồ bằng Plotly
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = value,
            title = {'text': "Percentage"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 100], 'color': "gray"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 75}}))

        # Cập nhật biểu đồ trong placeholder
        with placeholder:
            st.plotly_chart(fig)

        # Chờ một chút trước khi cập nhật lại
        time.sleep(0.01)  # Thời gian chờ giữa các lần cập nhật

# Update the plot_similarity function to use Plotly
def plot_similarity(similarities):
    fig = px.line(x=range(len(similarities)), y=similarities, labels={'x': 'Bin', 'y': 'Cosine Similarity'})
    fig.update_layout(title='Peaks in Cosine Similarity')
    st.plotly_chart(fig)


# TASK 1
st.title("Pronunciation Assist")

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

# # Nút chính để hiển thị các lựa chọn cho file âm thanh thứ hai
# if 'show_second_audio_buttons' not in st.session_state:
#     st.session_state.show_second_audio_buttons = False

# if st.button("Choose audio file to compare"):
#     st.session_state.show_second_audio_buttons = not st.session_state.show_second_audio_buttons

# if st.session_state.show_second_audio_buttons:
#     uploaded_file2 = st.file_uploader("Select second audio", type=["wav", "mp3", "aac", "flac", "ogg", "wma"])
#     if uploaded_file2 is not None:
#         st.session_state.uploaded_file2 = uploaded_file2
#         st.write("Done:")
#         st.audio(st.session_state.uploaded_file2, format='audio/wav')
        
#         # Lưu file vào thư mục tạm thời và in ra đường dẫn
#         with open(f"temp/{uploaded_file2.name}", "wb") as f:
#             f.write(uploaded_file2.getbuffer())
#         st.success(f"File uploaded successfully! File path: temp/{uploaded_file2.name}")    
#         file2 = f"temp/{uploaded_file2.name}"

# Nút chính để hiển thị các lựa chọn cho recording
if 'recording' not in st.session_state:
    st.session_state.recording = False

if st.button("Recording Audio"):
    st.session_state.recording = not st.session_state.recording

if st.session_state.recording:
    col1, col2, _ = st.columns(3)
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

# if st.button('Clear Temp'):
#     delete_files_in_folder(folder_path)
#     delete_files_in_folder(recordings_folder_path)
#     st.success('Clear temp successfully!')

if st.button('Compare'):
    
    # # DTW
    # file2 = check_recording_file()
    # sim_score = test_dtw.similarity_calculate(file1, file2, bin_size=0.05)
    # #a = test_dtw.remove_ones_and_compute_mean(sim_score)
    # a_percent = sim_score * 100
    # a = a_percent


    # COSINE
    file2 = check_recording_file()
    sim_score = task1.similarity_calculate(file1, file2, bin_size=0.05)
    a = task1.remove_ones_and_compute_mean(sim_score)
    a_percent = a * 100



    # Chuyển đến trang mới và hiển thị biểu đồ đồng hồ
    plot_score(a)
    if a_percent == 0:
        st.subheader("Maybe wrong word?") 
    elif a_percent < 50:
        st.subheader("Need a lot of improvement!") 
    elif a_percent >= 50 and a_percent < 75:
        st.subheader("Understandable!") 
    else:
        st.subheader("Well Done!") 

    with st.expander("Analyze Results"):
        text1 = recognize_speech(file1)
        text2 = recognize_speech(file2)
        if text1 == text2:
            st.write(f"Standard Word: {text1}")
            st.write(f"Your Word: {text2}")
            st.write(f"Similarity score: {a_percent:.2f}%")
            plot_similarity(sim_score)
        else:
            st.write(f"Standard Word: {text1}")
            st.write(f"Your Word: {text2}")
            st.write(f"Try Again!")
    # Thêm nút Reset
    if st.button('DELETE ALL'):
        st.session_state.show_file_uploader = False
        st.session_state.show_second_audio_buttons = False
        st.session_state.recording = False
        delete_files_in_folder(folder_path)
        delete_files_in_folder(recordings_folder_path)
        st.experimental_rerun()



# st.title("Vowel and Consonant Analyze")

# uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

# if uploaded_file is not None:
#     with open("temp_audio_file.wav", "wb") as f:
#         f.write(uploaded_file.getbuffer())
    
#     path = "temp_audio_file.wav"
#     phones = task1.phone_recognize_file(path)

#     hop_length = 512
#     spectrogram = task1.load_spectrogram(path, hop_length=hop_length, fmax=4000)

#     fig, (spec_ax, label_ax) = plt.subplots(2, figsize=(20, 8), sharex=True)
#     #librosa.display.specshow(spectrogram.T.values, ax=spec_ax, sr=16000, fmax=4000, hop_length=hop_length, x_axis='time')
#     task1.plot_events(label_ax, phones, color='color', annotate='label')

#     st.pyplot(fig)
    
#     st.write("### Vowel and Consonant Timestamps")
#     st.dataframe(phones)

#     @st.cache_data
#     def get_phones():
#         return phones

#     phones_data = get_phones()










def calculate_cosine_similarity(y1, y2, sr, timestamps):
    similarities = []
    for _, row in timestamps.iterrows():
        start = int(row['start'] * sr)
        end = int(row['end'] * sr)
        segment1 = y1[start:end]
        segment2 = y2[start:end]
        
        if len(segment1) > 0 and len(segment2) > 0:
            mfcc1 = librosa.feature.mfcc(y=segment1, sr=sr, n_mfcc=13).T
            mfcc2 = librosa.feature.mfcc(y=segment2, sr=sr, n_mfcc=13).T
            mean_mfcc1 = np.mean(mfcc1, axis=0).reshape(1, -1)
            mean_mfcc2 = np.mean(mfcc2, axis=0).reshape(1, -1)
            sim = task1.cosine_similarity(mean_mfcc1, mean_mfcc2)[0, 0]
            similarities.append(sim * 100)
        else:
            similarities.append(None)
    return similarities



st.title("Vowel and Consonant Analyze")

if st.button('Vowel and Consonant Analyze'):
    # if 'uploaded_file1' in st.session_state and 'uploaded_file2' in st.session_state:
    #     file1 = st.session_state.uploaded_file1.name
    #     file2 = check_recording_file()
    file2 = "F:\gr_project\\recordings\\recording.wav"
    if file1 and file2:
            # Load and process the first audio file
            y1, sr1 = task1.load_audio(file1)
            start1, end1 = task1.detect_sound_boundaries(y1)
            centered_audio1 = task1.center_audio(y1, sr1, start1, end1)
            norm_audio1 = task1.normalize_amplitude(centered_audio1)
            
            # Load and process the second audio file
            y2, sr2 = task1.load_audio(file2)
            start2, end2 = task1.detect_sound_boundaries(y2)
            centered_audio2 = task1.center_audio(y2, sr2, start2, end2)
            norm_audio2 = task1.normalize_amplitude(centered_audio2)

            if sr1 != sr2:
                st.error("Sample rates of the two audio files do not match.")
            else:
                # Recognize vowels and consonants in both files
                phones1 = task1.phone_recognize_file(file1)
                phones2 = task1.phone_recognize_file(file2)

                # Calculate cosine similarity based on timestamps from the first file
                similarities = calculate_cosine_similarity(norm_audio1, norm_audio2, sr2, phones2)
                phones2['similarity'] = similarities

                
                st.write("### Vowel and Consonant Timestamps for First Audio File")
                st.dataframe(phones1)

                st.write("### Vowel and Consonant Timestamps for Second Audio File with Similarity ")
                st.dataframe(phones2)


                phones1_norm = phones1.copy()
                # Lấy giá trị start ở hàng đầu tiên
                start_0 = phones1.loc[0, 'start']

                # Trừ giá trị start_0 cho các cột start và end
                phones1_norm['start'] = phones1_norm['start'] - start_0
                phones1_norm['end'] = phones1_norm['end'] - start_0
                

                phones2_norm = phones2.copy()
                # Lấy giá trị start ở hàng đầu tiên
                start_0 = phones2.loc[0, 'start']

                # Trừ giá trị start_0 cho các cột start và end
                phones2_norm['start'] = phones2_norm['start'] - start_0
                phones2_norm['end'] = phones2_norm['end'] - start_0
                

                # Plot events for both files in the same figure
                fig, (label_ax1, label_ax2) = plt.subplots(2, figsize=(20, 8), sharex=True)
                task1.plot_events(label_ax1, phones1_norm, color='color', annotate='label')
                label_ax1.set_title('Vowel and Consonant for First Audio File')
                task1.plot_events(label_ax2, phones2_norm, color='color', annotate='label')
                label_ax2.set_title('Vowel and Consonant for Second Audio File')
                st.pyplot(fig)

                # @st.cache_data
                # def get_phones():
                #     return phones1, phones2
                # phones_data1, phones_data2 = get_phones()
    else:
            st.warning("Please complete the steps in the Pronunciation Assist section first.")