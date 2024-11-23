# evaluate_accuracy.py

from audio_processing import load_data_from_directories, extract_mfcc
from model import train_hmms, recognize_speech
from sklearn.metrics import accuracy_score
import os

# Paths to your training and test data folders
TRAIN_DATASET_PATH = 'ds'
TEST_DATASET_PATH = 'test_data'

def load_test_data(test_path, le):
    test_data = []
    true_labels = []

    # Iterate through each subfolder in the test dataset
    for folder in os.listdir(test_path):
        folder_path = os.path.join(test_path, folder)
        
        if os.path.isdir(folder_path):  # Process only folders (which are categories)
            for filename in os.listdir(folder_path):
                if filename.endswith('.wav'):  # Process only .wav files
                    file_path = os.path.join(folder_path, filename)
                    # The folder name itself is the label (cat, bed, bird)
                    label = folder
                    mfcc = extract_mfcc(file_path)
                    test_data.append(mfcc)
                    true_labels.append(label)

    # Convert labels to encoded format to match model output
    encoded_labels = le.transform(true_labels)
    return test_data, encoded_labels

def main():
    print("Loading training data...")
    # Load training data and labels from your dataset
    train_data, train_labels = load_data_from_directories(TRAIN_DATASET_PATH)
    print(f"Loaded {len(train_data)} training samples.")

    print("Training HMM models...")
    # Train separate HMM models for each word
    models, le = train_hmms(train_data, train_labels)
    print("Models trained successfully!")

    print("Loading and evaluating test data...")
    # Load test data
    test_data, true_labels = load_test_data(TEST_DATASET_PATH, le)

    # Make predictions for each test sample
    predictions = []
    for mfcc in test_data:
        scores = {word: model.score(mfcc) for word, model in models.items()}
        predicted_word = max(scores, key=scores.get)
        predicted_label = le.transform([predicted_word])[0]
        predictions.append(predicted_label)

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Model Accuracy on test data: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
