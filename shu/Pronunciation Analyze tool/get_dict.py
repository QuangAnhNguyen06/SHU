import os
import requests
from bs4 import BeautifulSoup
import streamlit as st
import shutil
from pydub import AudioSegment
import imageio_ffmpeg as ffmpeg
from pathlib import Path


# Hàm tạo URL từ từ nhập vào
def input_word(word):
    base_url = "https://dictionary.cambridge.org/dictionary/english/"
    full_url = base_url + word
    return full_url

# Định nghĩa headers với user agent
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}


# Hàm chuyển đổi file MP3 sang WAV
import librosa
import numpy as np
import scipy.io.wavfile as wavfile

def convert_mp3_to_wav(mp3_path, wav_path):
    # Đọc file mp3
    audio_data, sample_rate = librosa.load(mp3_path, sr=None)
    
    # normalize to int16
    audio_data = np.int16(audio_data * 32767)
    
    # Ghi file wav
    wavfile.write(wav_path, sample_rate, audio_data)
    #print(f"Converted {mp3_path} to {wav_path}")




# Hàm để tải file âm thanh
def download_audio_file(audio_url, save_path):
    response = requests.get(audio_url, headers=headers, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        print(f"Audio downloaded successfully: {save_path}")
    else:
        print(f"Failed to download audio file. Status code: {response.status_code}")



# Hàm chính để scraping trang và tải âm thanh
def main(word):
    url = input_word(word)
    response = requests.get(url, headers=headers)

    # Kiểm tra nếu yêu cầu thành công
    if response.status_code == 200:
        # Phân tích nội dung HTML
        soup = BeautifulSoup(response.content, 'html.parser')

        # Tìm thẻ span với class chứa "uk"
        uk_span = soup.find('span', class_='uk dpron-i')

        if uk_span:
            # Tìm nguồn âm thanh trong thẻ span này
            audio_source = uk_span.find('source', type='audio/mpeg')

            if audio_source:
                # Lấy thuộc tính src
                audio_src = audio_source['src']
                #print(f"Found audio src: {audio_src}")

                # Tạo URL đầy đủ
                audio_url = f"https://dictionary.cambridge.org{audio_src}"
                #print(f"Full audio URL: {audio_url}")

                # Định nghĩa đường dẫn để lưu file âm thanh
                save_directory = "F:\\gr_project\\cam_audio"
                if not os.path.exists(save_directory):
                    os.makedirs(save_directory)
                save_path = os.path.join(save_directory, f'{word}.mp3')
                #mp3_path = os.path.join(save_directory, f'{word}.mp3')
                #wav_path = os.path.join(save_directory, f'{word}.wav')
                
                # Tải file âm thanh
                download_audio_file(audio_url, save_path)

                #download_audio_file(audio_url, mp3_path)
                #convert_mp3_to_wav(mp3_path, wav_path)
                #return wav_path
                
                return save_path
            else:
                print("No audio source found within the 'uk' span.")
        else:
            print("No span with class containing 'uk' found.")
    else:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")
    return None




# Hàm để xóa tất cả các file trong thư mục
def delete_all_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')











# Tạo giao diện Streamlit
st.title('Dictionary Audio Downloader')

# Tạo search bar để nhập từ tiếng Anh
word = st.text_input('Input a word:')

save_directory = "F:\\gr_project\\cam_audio"

if st.button('Download a word'):
    if not word:
        st.error("Error.")
    else:
        save_path = main(word)
        if save_path:
            st.success(f'Success downloading "{word}"!')
            st.audio(save_path)



st.title("MP3 to WAV Converter")

# Giao diện Streamlit
uploaded_file = st.file_uploader("Choose MP3 file", type=["mp3"])


if uploaded_file is not None:
    save_folder = Path("F:/gr_project/cam_audio")
    mp3_path = save_folder / uploaded_file.name

    with open(mp3_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Upload Done!")

    wav_path = save_folder / f"{mp3_path.stem}.wav"

    convert_mp3_to_wav(mp3_path, wav_path)
    st.success(f"Converted to WAV and save at {wav_path}")



if st.button('Delete All'):
    delete_all_files_in_folder(save_directory)
    st.success('Delete Success')