# Single-Trial EEG Classification: Decoding Timbre Perception

This repository contains scripts for classifying EEG data on a single-trial basis to decode timbre perception. The pipeline is designed for use with 3D preprocessed EEG data (trials × channels × time), stored in MATLAB `.mat` format.

## 🧠 Objective
To investigate whether timbre-related auditory information can be decoded from early EEG responses using supervised machine learning techniques.

---

## 🗃️ `classification_Groupcv.py`

- Loads and prepares EEG data for classification.
- Labels each trial based on stimulus condition (based on information already stored in .mat data matrix)
- Uses **GroupKFold** to split trials into training/testing sets while ensuring block-wise trial grouping (trials were presented in blocks of 5 per participant).
- Includes:
  - PCA-based dimensionality reduction
  - Multiple classifiers: LDA, SVM, k-NN, Gradient Boosting
  - GridSearchCV for hyperparameter tuning

---

## 🧪 `feature_extraction.py`

Includes several domain-informed EEG feature engineering methods:
1. `compute_psd`: Power spectral density across canonical EEG bands
2. `erp_features`: Peak/latency extraction from time-domain ERPs
3. `compute_periodicity`: Harmonic frequency power (e.g., 55 Hz, 110 Hz)
4. `offsets_features`: Post-stimulus slope and mean amplitude
5. `peak_power`: 

---

## 📦 Data Format

- Input: 3D NumPy arrays or MATLAB `.mat` files  
  Format: `(n_trials, n_channels, n_timepoints)`
