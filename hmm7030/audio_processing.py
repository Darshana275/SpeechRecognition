# audio_processing.py

import librosa
import os
import numpy as np

# Function to extract MFCC features from a .wav file
def extract_mfcc(audio_file):
    audio, sr = librosa.load(audio_file, sr=None)  # Load audio
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)  # Extract MFCC features
    mfcc = mfcc.T  # Transpose to have time frames as rows
    return mfcc

# Function to load audio data from directories and extract MFCC features
def load_data_from_directories(dataset_path):
    data = []
    labels = []

    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        
        if os.path.isdir(label_path):  # Check if it's a directory (bed, bird, cat)
            for filename in os.listdir(label_path):
                if filename.endswith('.wav'):  # Check if the file is a .wav file
                    file_path = os.path.join(label_path, filename)
                    mfcc = extract_mfcc(file_path)
                    data.append(mfcc)
                    labels.append(label)  # The folder name will be the label (bed, bird, cat)

    return data, labels