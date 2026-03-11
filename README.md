ICU False Alarm Detection using 1D-CNN

Project Overview

In Intensive Care Units (ICUs), up to 80% of alarms are false, leading to "alarm fatigue" among medical staff and putting patients at risk. This project implements a Deep Learning solution to classify ICU alarms as True or False by analyzing multi-channel physiological signals (ECG, PPG, and ABP).

Developed during my Erasmus+ at the University of Augsburg, Germany.

Technical Architecture

The core is a custom 1D Convolutional Neural Network (CNN) designed to extract features from temporal waveforms:

- Feature Extractor: 4 blocks of Conv1d + BatchNorm + ReLU + MaxPool1d.
- Adaptive Pooling: Collapses the time dimension to a fixed-length vector, allowing variable input lengths.
- Classifier: Dense layers with Dropout (0.3) to prevent overfitting and a final logit output for binary classification.

Data Pipeline:
- Z-Normalization: Per-channel normalization (mean=0, std=1) for stable training.
- Signal Preprocessing: Handling NaN/Inf values and cropping the last 20 seconds of each recording to focus on the event trigger.


Performance & Optimization

- Threshold Tuning: Instead of a default 0.5, the model uses an optimized threshold (default 0.35) to maximize the Weighted F1-Score, prioritizing the reduction of critical false negatives.
- Class Imbalance: Handled using BCEWithLogitsLoss with a pos_weight of 2.8.
- Early Stopping: Implemented to halt training when validation loss stops improving, ensuring the best model weights are saved.


How to Run

Environment Setup:
```
conda activate condaenv-dl4biosig  
pip install -r requirements.txt 
```

Training:
Place your trainSet.h5 in the data/ folder and run:
```
python train_model.py
```
Inference:
```
python run_model.py --test_path data/testSet.h5
```

Repository Structure

- model.py: PyTorch implementation of the 1D-CNN.
- dataset.py: Custom H5 dataset loader and preprocessing logic.
- train_utils.py: Evaluation metrics, Early Stopping, and Threshold Optimization.
- constants.py: Centralized hyperparameters (Learning Rate: 6e-4, Batch Size: 32).
