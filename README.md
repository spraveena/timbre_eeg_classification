# Single Trial EEG Classification Decoding Timbre Perception

This script takes in 3-dimensional preprocessed EEG data stored in a MATLAB variable.

### classification_Groupcv.py
Loads data, and assigns labels according the the class each trial belongs to. The data is split into training and test sets. The format of the data colelcted was such that stimulus was presented to participants in blocks of 5, hence the use of GroupKFold to prevent leakage into the test set when the data is split.

Classification pipeline includes PCA dimensionality reduction before classification using one of four possible classifiers. The script performs grid search CV to obtain best model parameters.


### feature_extraction.py
Contains several domain-informed feature extraction functions such as :
1. compute_psd
2. erp_features
3. compute_periodicty
4. offsets_features
5. peak_power



