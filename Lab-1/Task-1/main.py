import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

def find_best_k():
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Define the range of K values to test
    k_range = range(1, 26)
    
    # Dictionary to store the average accuracy for each value of K
    k_avg_scores = {}

    # Loop through each value of K in the defined range
    for k in k_range:
        accuracies = []
        # For each K, run the experiment 10 times to get a stable average
        for i in range(10):
            # Split data into 70% training and 30% testing sets
            # Use a different random_state (i) for each run to ensure different splits
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=i)

            # Scale the features to have a mean of 0 and variance of 1
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Initialize and train the KNN classifier with the current K
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train_scaled, y_train)

            # Make predictions and calculate the accuracy for this run
            y_pred = knn.predict(X_test_scaled)
            accuracies.append(metrics.accuracy_score(y_test, y_pred))

        # Calculate the average accuracy across the 10 runs for the current K
        k_avg_scores[k] = np.mean(accuracies)

    print("Average accuracy for each value of K (averaged over 10 runs):")
    for k, score in k_avg_scores.items():
        print(f"K = {k:2d}: Average Accuracy = {score:.4f}")

    best_k = max(k_avg_scores, key=k_avg_scores.get)
    print(f"\nThe best K is {best_k} with an average accuracy of {k_avg_scores[best_k]:.4f}.\n")

    plt.figure(figsize=(12, 7))
    plt.plot(list(k_avg_scores.keys()), list(k_avg_scores.values()), marker='o', linestyle='--', color='dodgerblue')
    plt.title('Average Accuracy vs. K Value for Iris Dataset')
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Average Accuracy')
    plt.xticks(k_range)
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    find_best_k()