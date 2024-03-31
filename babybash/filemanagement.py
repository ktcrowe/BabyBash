# Tools for file management

import os
import librosa  # For audio feature extraction

from audioprocessing import normalize_audio_length  # Tools for audio processing
from pydub import AudioSegment  # For converting M4A to WAV format


# load files from a given folder and return the features (MFCCs) and labels (normalizing audio length if specified)
def load_data(folder, label, sample_rate, n_mfcc, normalize_to_length=None):
    features = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith('.wav'):
            try:
                path = os.path.join(folder, filename)
                audio = normalize_audio_length(path, normalize_to_length, sample_rate) if normalize_to_length else librosa.load(path, sr=sample_rate)[0]
                mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
                mfcc = mfcc.mean(axis=1)
                features.append(mfcc)
                labels.append(label)

            except Exception as e:
                print(f'Error encountered while parsing {filename}: {e}')
    return features, labels


# Count the amount of files in a given directory (used for automatically naming models when loading/saving)
def count_files_in_folder(folder_path):
    if not os.path.exists(folder_path):
        return "Folder not found."

    file_count = 0
    for entry in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, entry)):
            file_count += 1

    return file_count


# Convert M4A to WAV format
def m4a_to_wav():
    for filename in os.listdir('../data/positive'):
        if filename.endswith('.m4a'):
            sound = AudioSegment.from_file(f'../data/positive/{filename}', format='m4a')
            file_handle = sound.export(f'../data/positive/{filename[:-4]}.wav', format='wav')
            os.remove(f'../data/positive/{filename}')
    for filename in os.listdir('../data/negative'):
        if filename.endswith('.m4a'):
            sound = AudioSegment.from_file(f'../data/negative/{filename}', format='m4a')
            file_handle = sound.export(f'../data/negative/{filename[:-4]}.wav', format='wav')
            os.remove(f'../data/negative/{filename}')

