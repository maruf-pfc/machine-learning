import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# STEP 1: DEFINE THE LOGISTIC REGRESSION CLASSIFIER
# This class contains all the logic for our model.
# ==============================================================================

class LogisticRegressionScratch:
    """
    A from-scratch implementation of Logistic Regression using Gradient Descent.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Constructor to set up our model's hyperparameters.
        
        Args:
            learning_rate (float): How big of a step to take on each iteration.
            n_iterations (int): How many times to loop through the training data.
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        """
        The private sigmoid function. This is the heart of logistic regression.
        Formula: g(z) = 1 / (1 + e^-z)
        """
        # This function squishes any number into a probability between 0 and 1.
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        This method trains the model by learning the best weights and bias.
        """
        # Get the number of samples and features from the input data
        n_samples, n_features = X.shape

        # A. Initialize parameters: Start weights and bias at zero.
        self.weights = np.zeros(n_features)
        self.bias = 0

        # B. Gradient Descent: Loop many times to slowly improve the parameters.
        for _ in range(self.n_iterations):
            # Calculate the model's current predictions (as a probability)
            # z = (weights * features) + bias
            linear_model_output = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model_output)

            # C. Calculate the gradients (the direction to update our parameters)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # D. Update the parameters to improve the model
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """
        This method uses the learned weights and bias to make a final prediction.
        """
        # Calculate the final probabilities using the trained parameters
        linear_model_output = np.dot(X, self.weights) + self.bias
        y_predicted_proba = self._sigmoid(linear_model_output)

        # Convert probabilities to a final class prediction (0 or 1)
        y_predicted_class = [1 if i > 0.5 else 0 for i in y_predicted_proba]
        return np.array(y_predicted_class)


# ==============================================================================
# STEP 2: LOAD AND PREPARE THE DATA
# We will use the famous Iris dataset for this example.
# ==============================================================================

# Load the dataset
iris = load_iris()
X = iris.data

# We will simplify the problem: predict if a flower is 'setosa' (class 1) or not (class 0).
# This makes it a binary classification problem, which is perfect for this model.
y = (iris.target == 0).astype(int)  # `y` is now `[1, 1, ..., 0, 0, ...]`

# ==============================================================================
# STEP 3: SPLIT THE DATA INTO TRAINING AND TESTING SETS
# We train the model on one part of the data and test it on another, unseen part.
# ==============================================================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==============================================================================
# STEP 4: SCALE THE FEATURES
# This step is VERY important for gradient descent to work well.
# It puts all our features on the same scale.
# ==============================================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================================================================
# STEP 5: TRAIN THE LOGISTIC REGRESSION MODEL
# Now we create an instance of our class and train it with our data.
# ==============================================================================

print("Training the logistic regression model...")
# Create an instance of our classifier
model = LogisticRegressionScratch(learning_rate=0.1, n_iterations=1000)

# Train it using the scaled training data
model.fit(X_train_scaled, y_train)
print("Model training complete!")

# ==============================================================================
# STEP 6: MAKE PREDICTIONS AND EVALUATE THE MODEL
# Let's see how well our model performs on the unseen test data.
# ==============================================================================

# Get the model's predictions on the test set
predictions = model.predict(X_test_scaled)

# Calculate the accuracy: (Number of correct predictions) / (Total predictions)
accuracy = np.mean(predictions == y_test)

print("\n--- Model Evaluation ---")
print(f"The model's predictions on the test set: {predictions}")
print(f"The actual labels of the test set:      {y_test}")
print(f"\nThe accuracy of our model is: {accuracy * 100:.2f}%")

# The Iris 'setosa' class is very easy to separate, so we expect 100% accuracy.
if accuracy == 1.0:
    print("\nThe model achieved perfect accuracy, successfully learning to classify the data!")