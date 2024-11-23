# main.py

from audio_processing import load_data_from_directories
from model import train_hmms, recognize_speech
import warnings

# Suppress all UserWarnings (including convergence warnings from hmmlearn)
warnings.filterwarnings("ignore", category=UserWarning)

# Path to your dataset folder
DATASET_PATH = 'ds'

def main():
    print("Loading data...")
    # Load data and labels from your dataset
    data, labels = load_data_from_directories(DATASET_PATH)
    print(f"Loaded {len(data)} samples.")
    
    print("Training HMM models...")
    # Train separate HMM models for each word with MFCC data
    models, le = train_hmms(data, labels)
    print("Models trained successfully!")

    # Test the model on a new audio file (replace with your own path)
    test_audio_file = 'test_data/cat/43.wav'  # Update with a path to your test audio file
    print("Recognizing speech...")
    recognized_word = recognize_speech(models, le, test_audio_file)
    print(f"Predicted word: {recognized_word}")
if __name__ == "__main__":
    main()
