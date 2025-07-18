import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

def find_best_ratio_synthetic():
    X_synthetic, y_synthetic = make_classification(
        n_samples=500,       # Total number of data points
        n_features=15,       # Total number of features
        n_informative=8,     # Number of useful features
        n_redundant=3,       # Number of redundant features
        n_classes=4,         # Number of distinct classes/categories
        random_state=42      # For reproducibility
    )
    print("A synthetic dataset has been created with 500 samples, 15 features, and 4 classes.")

    # Use a fixed K value for the classifier. A common heuristic is sqrt(n_samples),
    # but we will choose a fixed value like 9.
    k_value = 9
    print(f"Running experiments with a fixed K={k_value}.\n")

    # Define the range of test set sizes to evaluate
    test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Dictionary to store the average accuracy for each ratio
    ratio_accuracies = {}

    # Loop over each test size
    for size in test_sizes:
        accuracies = []
        # Repeat the experiment 10 times for each ratio for stable results
        for i in range(10):
            # Split the synthetic data
            X_train, X_test, y_train, y_test = train_test_split(X_synthetic, y_synthetic, test_size=size, random_state=i)

            # Scale the features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Initialize and train the KNN classifier
            knn = KNeighborsClassifier(n_neighbors=k_value)
            knn.fit(X_train_scaled, y_train)

            # Predict and calculate accuracy
            y_pred = knn.predict(X_test_scaled)
            accuracies.append(metrics.accuracy_score(y_test, y_pred))

        # Calculate the average accuracy for the current ratio
        train_percent = int((1 - size) * 100)
        test_percent = int(size * 100)
        ratio_key = f"{train_percent}/{test_percent}"
        ratio_accuracies[ratio_key] = np.mean(accuracies)

    print("Average accuracy for different train/test ratios on the Synthetic Dataset:")
    for ratio, acc in ratio_accuracies.items():
        print(f"Ratio (Train/Test) = {ratio}: Average Accuracy = {acc:.4f}")

    best_ratio = max(ratio_accuracies, key=ratio_accuracies.get)
    print(f"\nThe best train/test ratio for the synthetic dataset is {best_ratio} with an average accuracy of {ratio_accuracies[best_ratio]:.4f}.\n")

    plt.figure(figsize=(10, 6))
    plt.plot(list(ratio_accuracies.keys()), list(ratio_accuracies.values()), marker='o', linestyle='-', color='green')
    plt.title('Accuracy vs. Train/Test Split Ratio for Synthetic Dataset')
    plt.xlabel('Train/Test Split Ratio')
    plt.ylabel('Average Accuracy')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    find_best_ratio_synthetic()