from audio_processing import load_data_from_directories, extract_mfcc
from model import train_hmms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

TRAIN_DATASET_PATH = 'ds'
TEST_DATASET_PATH = 'test_data'

def load_test_data(test_path, le):
    test_data = []
    true_labels = []

    for folder in os.listdir(test_path):
        folder_path = os.path.join(test_path, folder)
        
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith('.wav'):
                    file_path = os.path.join(folder_path, filename)
                    mfcc = extract_mfcc(file_path)
                    test_data.append(mfcc)
                    true_labels.append(folder)

    if not test_data:
        print("No test data found. Check the TEST_DATASET_PATH and subfolder structure.")
        return None, None

    encoded_labels = le.transform(true_labels)
    return test_data, encoded_labels

def main():
    print("Loading training data...")
    train_data, train_labels = load_data_from_directories(TRAIN_DATASET_PATH)
    print(f"Loaded {len(train_data)} training samples.")

    print("Training HMM models...")
    models, le = train_hmms(train_data, train_labels)
    print("Models trained successfully!")

    print("Loading and evaluating test data...")
    test_data, true_labels = load_test_data(TEST_DATASET_PATH, le)
    if test_data is None or true_labels is None:
        print("Error: No test data available. Please check your test data path and files.")
        return

    predictions = []
    for mfcc in test_data:
        scores = {word: model.score(mfcc) for word, model in models.items()}
        predicted_word = max(scores, key=scores.get)
        predicted_label = le.transform([predicted_word])[0]
        predictions.append(predicted_label)

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")

    cm = confusion_matrix(true_labels, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == "__main__":
    main()
