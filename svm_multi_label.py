import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    label_ranking_average_precision_score,
    precision_score,
    recall_score,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import (
    MultiLabelBinarizer,
    PolynomialFeatures,
    StandardScaler,
    normalize,
)
from sklearn.svm import SVC

# Constants
DATA_FILE_PATH = "./Bibtex/Bibtex_data.txt"
TRAIN_DATA_FILE_PATH = "./Bibtex/bibtex_trSplit.txt"
TEST_DATA_FILE_PATH = "./Bibtex/bibtex_tstSplit.txt"


def convert_txt_data_to_df(txt_file):
    """Converts a specified txt file to a DataFrame."""
    try:
        with open(txt_file, "r") as file:
            lines = file.readlines()
    except IOError as e:
        print(f"Error opening file: {e}")
        return None, None, None

    # Extract the number of rows, columns, and labels from the first line
    num_rows, num_features, num_classes = map(int, lines[0].split())

    data = np.full((num_rows, num_features), -1, dtype=int)
    labels_data = []

    for idx, line in enumerate(lines[1:]):
        label_part, features_part = line.split(" ", 1)
        labels = set(map(int, label_part.split(",")))
        labels_data.append(labels)

        for feature in features_part.strip().split():
            index, _ = feature.split(":")
            data[idx, int(index) - 1] = 1

    mlb = MultiLabelBinarizer(classes=range(num_classes))
    labels_matrix = mlb.fit_transform(labels_data)
    class_labels = mlb.classes_

    df = pd.DataFrame(data, columns=[f"feature_{i}" for i in range(num_features)])
    labels_df = pd.DataFrame(
        labels_matrix, columns=[f"label_{i}" for i in range(num_classes)]
    )
    df = pd.concat([df, labels_df], axis=1)

    return df, labels_matrix, class_labels


def load_indices(file_path):
    """Load indices from a file."""
    try:
        with open(file_path, "r") as file:
            indices = sorted([int(line.split(" ")[0].strip()) - 1 for line in file])
    except IOError as e:
        print(f"Error reading indices from file: {e}")
        return None
    return np.array(indices)


def analyze_data(data, labels_df):
    """Perform data analysis including plotting label distribution and feature sparsity."""
    variances = np.var(data, axis=0)
    plt.hist(variances, bins=50)
    plt.title("Histogram of Feature Variances")
    plt.xlabel("Variance")
    plt.ylabel("Number of Features")
    plt.grid(True)
    plt.show()

    # Plot sorted variances
    sorted_variances = np.sort(variances)
    plt.plot(sorted_variances)
    plt.title("Sorted Feature Variances")
    plt.xlabel("Feature Index")
    plt.ylabel("Variance")
    plt.grid(True)
    plt.show()

    label_counts = labels_df.sum().sort_values(ascending=False)
    d = [
        go.Bar(
            x=label_counts.sort_index().values,
            y=label_counts.sort_index().index,
            orientation="h",
        )
    ]
    layout = go.Layout(height=1000)
    fig = go.Figure(data=d, layout=layout)
    pyo.plot(fig)

    feature_counts = np.count_nonzero(data.to_numpy(), axis=0)
    plt.hist(feature_counts, bins=30, color="blue", alpha=0.7)
    plt.title("Histogram of Non-Zero Feature Counts")
    plt.xlabel("Number of Non-Zero Entries")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()


def preprocess_data(data, threshold=0.40):
    """Preprocess data by scaling and normalization."""
    selector = VarianceThreshold(0.05)
    data_reduced = selector.fit_transform(data)

    # Scale data before applying PCA
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_reduced)

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=threshold)
    pca.fit(data_scaled)
    reduced_data = pca.transform(data_scaled)

    print(
        f"Number of components that make up {threshold*100}% of variance:"
        f" {pca.n_components_}"
    )

    return reduced_data


def train_and_evaluate_svm(
    x_train,
    x_test,
    y_train,
    y_test,
    C,
    class_labels,
    poly_features=False,
    pegasos=False,
):

    results = []
    if pegasos:
        base_clf = SGDClassifier(
            alpha=1 / C, max_iter=10000, random_state=42, verbose=1
        )
        estimator = CalibratedClassifierCV(base_clf, method="sigmoid")
    else:
        estimator = SVC(
            kernel="linear",
            C=C,
            probability=True,
            class_weight="balanced",
            cache_size=1000,
        )

    model = OneVsRestClassifier(
        estimator=estimator,
        n_jobs=-1,
        verbose=60,
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)  # Predict probabilities

    # Get the indices of the top 5 predictions for each sample
    top_5_indices = np.argsort(-y_pred_proba, axis=1)[:, :5]

    # Convert indices to class labels
    top_5_labels = {
        f"test_sample_{i}": [class_labels[i] for i in indices]
        for i, indices in enumerate(top_5_indices)
    }

    # Create a binary matrix of predictions based on top 5 indices
    top_5_pred = np.zeros_like(y_test, dtype=bool)
    for i, indices in enumerate(top_5_indices):
        top_5_pred[i, indices] = True

    precision_at_5 = label_ranking_average_precision_score(
        y_test,
        y_pred_proba,
    )

    accuracy = accuracy_score(y_test, y_pred, normalize=True) * 100
    precision = (
        precision_score(y_test, y_pred, average="weighted", zero_division=1) * 100
    )
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0) * 100
    loss = hamming_loss(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=1) * 100

    results.append((C, accuracy, precision, recall, f1, precision_at_5, loss))
    print(
        f"Completed {'POLY' if poly_features else 'Linear'} SVM calculation with C={C} ...."
    )
    print(
        f"Accuracy of {'POLY' if poly_features else 'Linear'} SVM with C={C}: {accuracy}"
    )
    print(
        f"Hamming loss of {'POLY' if poly_features else 'Linear'} SVM with C={C}: {loss}"
    )
    print(
        f"Precision of {'POLY' if poly_features else 'Linear'} SVM with C={C}: {precision}"
    )
    print(f"Recall of {'POLY' if poly_features else 'Linear'} SVM with C={C}: {recall}")
    print(f"F1 Score of {'POLY' if poly_features else 'Linear'} SVM with C={C}: {f1}")
    print(f"Precision@5 with C={C}: {precision_at_5}")
    print(f"Top 5 labels with C={C}: {top_5_labels}")

    return results


def plot_results(results, title):
    _, axes = plt.subplots(nrows=6, ncols=1, figsize=(8, 15))
    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "Precision@5",
        "Hamming Loss",
    ]
    for i, ax in enumerate(axes.flat):
        ax.plot(
            [result[0] for result in results],
            [result[i + 1] for result in results],
            marker="o",
            label=metrics[i],
        )
        ax.set_title(f"{metrics[i]} for {title} Features")
        ax.set_xlabel("C Value")
        ax.set_ylabel(metrics[i])
        ax.legend()
    plt.tight_layout()
    plt.show()


def polynomial_2_transformation(X_train, X_test):
    poly_transformer = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly_transformer.fit_transform(X_train)
    X_test_poly = poly_transformer.transform(X_test)
    return X_train_poly, X_test_poly


def main():
    data, labels, class_labels = convert_txt_data_to_df(DATA_FILE_PATH)
    if data is None or labels is None:
        print("Failed to load data.")
        return

    train_indices = load_indices(TRAIN_DATA_FILE_PATH)
    test_indices = load_indices(TEST_DATA_FILE_PATH)
    if train_indices is None or test_indices is None:
        print("Failed to load indices.")
        return

    labels_df = pd.DataFrame(
        labels, columns=[f"label_{i}" for i in range(labels.shape[1])]
    )
    analyze_data(data, labels_df)

    pre_processed_data = preprocess_data(data, 0.40)
    X_train = normalize(pre_processed_data[train_indices], "l2")
    X_test = normalize(pre_processed_data[test_indices])
    y_train = labels[train_indices]
    y_test = labels[test_indices]
    C_values = [0.1, 1, 10, 100, 500]
    for C in C_values:
        print("Evaluating Linear SVM...")
        linear_results = train_and_evaluate_svm(
            X_train,
            X_test,
            y_train,
            y_test,
            C,
            class_labels,
        )
        print("Linear SVM Results:", linear_results)
        plot_results(linear_results, "Linear SVM")

        # Transforming data using polynomial features
        print("Applying Polynomial Transformation...")
        X_train_poly, X_test_poly = polynomial_2_transformation(
            X_train,
            X_test,
        )

        # Using SVM on polynomial features
        print("Evaluating Polynomial SVM...")
        poly_results = train_and_evaluate_svm(
            X_train_poly,
            X_test_poly,
            y_train,
            y_test,
            C,
            class_labels,
            poly_features=True,
        )
        print("Polynomial SVM Results:", poly_results)
        plot_results(poly_results, "Polynomial SVM")


if __name__ == "__main__":
    main()
