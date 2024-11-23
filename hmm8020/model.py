# model.py

import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder
import logging

# Suppress hmmlearn convergence messages
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

# Function to train separate HMM models for each word
def train_hmms(data, labels, n_states=2, n_iter=100, tol=1e-2):
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # Dictionary to hold models for each word
    models = {}
    for class_label in np.unique(y):
        word = le.inverse_transform([class_label])[0]
        
        # Gather all sequences for this word
        word_data = [data[i] for i in range(len(data)) if y[i] == class_label]
        
        # Concatenate sequences to create a large training set for this word
        X = np.concatenate(word_data, axis=0)
        lengths = [len(seq) for seq in word_data]
        
        # Initialize and train an HMM model for this word with optimized parameters
        model = hmm.GaussianHMM(
            n_components=n_states, 
            covariance_type="diag",  # Using diagonal covariance for faster training
            n_iter=n_iter, 
            tol=tol
        )
        model.fit(X, lengths)
        models[word] = model

    return models, le

# Function to recognize speech using the trained HMM models
def recognize_speech(models, le, audio_file):
    from audio_processing import extract_mfcc

    mfcc = extract_mfcc(audio_file)

    # Calculate the likelihood of the MFCC sequence for each word model
    scores = {word: model.score(mfcc) for word, model in models.items()}
    
    # Get the word with the highest likelihood
    recognized_word = max(scores, key=scores.get)
    return recognized_word
