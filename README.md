# SpeechRecognition

This project investigates the performance of Hidden Markov Models (HMM) for speech recognition using datasets with varying train-test split ratios: 70:30, 80:20, and 90:10. The dataset comprises three categories of audio data—bird, bed, and cat. This study evaluates how different data splits impact the accuracy of HMM models.

Dataset Description
The dataset contains three subfolders:
bird: Contains audio files labeled as "bird."
bed: Contains audio files labeled as "bed."
cat: Contains audio files labeled as "cat."

Each subfolder contains a total number of files, which are split into training and testing datasets according to the specified ratios.

Methodology
Model Selection: Hidden Markov Models (HMMs) were employed for speech recognition tasks.
Evaluation: The models were evaluated based on accuracy over 10 test runs for each train-test split ratio. The average accuracy for each split was computed to provide a clear comparison.
Performance Metrics: Accuracy (%) was used as the primary metric, measured as the average of 10 evaluations for each split ratio.

To run the project
1. Clone the repository:
   "git clone https://github.com/Darshana275/Split-Ratio-Impact-on-HMMs_SpeechRecognitionModel"
   
   "cd hmm7030 or cd hmm8020 or cd hmm9010"  (based on the train-test split ratio you want to work on)
   
3. Install the dependencies
   Ensure you have the required dependencies installed. These may include libraries such as `numpy`, `scipy`, and `hmmlearn`. Install them using:
  "pip install numpy scipy hmmlearn"

4. Run Prediction:
    Use main.py to predict a word based on the input audio file. Execute the following command:
    "python foldername/main.py"
    where foldername is either hmm7030 or hmm8020 or hmm9010
    Follow the prompts to input the audio file for prediction.

5. Evaluate the Model:
   Use evaluate.py to compute accuracy, precision, recall, F1 score, and generate ROC AUC curves. Run the script with:
   "python foldername/evaluate.py"
   where foldername is either hmm7030 or hmm8020 or hmm9010
   This script will display evaluation metrics and generate visualizations for model performance.

Results

Split Ratio      Accuracy (%) (Average of 10 Evaluations)
70:30             79.63%
80:20             79.05%
90:10             79.85%

70:30 Split: Achieved the highest average accuracy of 79.63%, indicating a balanced dataset size for training and testing.
80:20 Split: Slightly lower accuracy (79.05%), likely due to reduced testing data.
90:10 Split: Achieved a comparable accuracy of 79.85%, with reduced testing data but increased training data.

Conclusion

The 70:30 split provides a good balance between training and testing data, resulting in the best overall accuracy. However, both 80:20 and 90:10 splits also demonstrate comparable performance, suggesting that HMM models are robust to variations in train-test split ratios within this range.
