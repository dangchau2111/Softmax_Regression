# softmax_regression.py

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def standard_scaler(X):
    """Normalize data using mean and standard deviation."""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1  # Avoid division by zero
    return (X - mean) / std

class SoftmaxRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def softmax(self, z):
        """Compute softmax probabilities."""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def compute_loss(self, X, y):
        """Compute the Categorical Cross-Entropy Loss."""
        m = X.shape[0]
        predictions = self.softmax(np.dot(X, self.theta))
        y_one_hot = np.eye(self.num_classes)[y]
        loss = -np.sum(y_one_hot * np.log(predictions + 1e-8)) / m
        return loss

    def fit(self, X, y):
        """Train the Softmax Regression model."""
        m, n = X.shape
        self.num_classes = len(np.unique(y))
        self.theta = np.zeros((n, self.num_classes))

        # One-hot encode labels
        y_one_hot = np.eye(self.num_classes)[y]

        for i in range(self.num_iterations):
            # Compute linear model
            linear_model = np.dot(X, self.theta)
            # Compute predictions using softmax
            predictions = self.softmax(linear_model)

            # Compute gradients
            gradient = np.dot(X.T, (predictions - y_one_hot)) / m
            # Update weights
            self.theta -= self.learning_rate * gradient

            # Print loss every 100 iterations
            if i % 100 == 0:
                loss = self.compute_loss(X, y)
                print(f'Iteration {i}: Loss = {loss}')

    def predict(self, X):
        """Predict the class labels."""
        linear_model = np.dot(X, self.theta)
        predictions = self.softmax(linear_model)
        return np.argmax(predictions, axis=1)

    def score(self, X, y):
        """Calculate the accuracy of the model."""
        predictions = self.predict(X)
        return np.mean(predictions == y)

def main():
    # Load MNIST dataset
    mnist = fetch_openml('mnist_784')

    # Data and labels
    X = mnist.data
    y = mnist.target.astype(int)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize data using NumPy
    X_train = standard_scaler(X_train)
    X_test = standard_scaler(X_test)

    # Initialize and train the model
    model = SoftmaxRegression(learning_rate=0.1, num_iterations=1000)
    model.fit(X_train, y_train)

    # Evaluate the model
    accuracy = model.score(X_test, y_test)
    print(f'Accuracy: {accuracy * 100:.2f}%')

if __name__ == "__main__":
    main()
