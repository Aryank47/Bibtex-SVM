# Multi-label Classification with SVM

This project demonstrates the implementation of a Support Vector Machine (SVM) for multi-label classification on the Bibtex dataset. The model predicts the top-5 labels for each test sample based on the highest confidence scores and reports average precision@5 for the complete test set. Multi-label classification is crucial for scenarios like tagging systems where multiple categories may apply to a single item.

## Table of Contents

- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Running the Script](#running-the-script)
- [Code Overview](#code-overview)
- [Outputs](#outputs)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Project Structure

- `Bibtex/`: Folder containing the dataset files.
  - `Bibtex_data.txt`: Main data file in a custom text format.
  - `bibtex_trSplit.txt`: Indices for training samples.
  - `bibtex_tstSplit.txt`: Indices for testing samples.
- `svm_multi_label.py`: Main Python script containing the SVM model, data preprocessing, and evaluation functions.

## Requirements

- Python 3.8 or above
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy

## Setup

You can install the necessary libraries using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy plotly
```

## Running the Script

To run the script, navigate to the directory containing `svm_multi_label.py` and execute the following command in the terminal:

```bash
python svm_multi_label.py
```

Ensure that the Bibtex dataset files are in the correct directory as mentioned in the project structure.

## Code Overview

### Data Preprocessing

- **convert_txt_data_to_df**: Converts the raw text data into DataFrame format, suitable for processing. It uses `MultiLabelBinarizer` to encode sets of labels into binary form, facilitating SVM training.
- **load_indices**: Fetches indices for training and testing from specified files, ensuring data is correctly partitioned.
- **preprocess_data**: Applies feature scaling and dimensionality reduction via `VarianceThreshold` and `PCA`.

### Model Training and Evaluation

- **train_and_evaluate_svm**: Trains the SVM model using `OneVsRestClassifier`. It computes predictions for the test set and extracts the top-5 labels based on confidence scores. Precision@5 is calculated using `label_ranking_average_precision_score`.

### Analysis Functions

- **analyze_data**: Visualizes data characteristics like label distribution and feature variance.

### Polynomial Feature Transformation

- **polynomial_2_transformation**: Applies a polynomial feature transformation of degree 2 to the training and testing data.

## Outputs

- Histograms and plots illustrating data characteristics.
- Performance metrics of the SVM model for each configuration of regularization parameter C.
- Top-5 predicted labels for each test sample.
- Console outputs indicating the progress and results of model training and evaluation.

## Troubleshooting

- **File Not Found**: Ensure all data files are placed in the `Bibtex/` directory as specified. Check paths in the script if errors persist.
