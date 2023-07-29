import numpy as np
from scipy.spatial.distance import cdist

class RBFNetwork:
    def __init__(self, k):
        self.k = k
        self.centroids = None
        self.weights = None

    def fit(self, X, y):
        # Step 1: Initialize the centroids randomly
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]

        # Step 2: Calculate the distance between each training example and the k centroids
        distances = cdist(X, self.centroids)

        # Step 3: Compute the RBF outputs
        rbf_outputs = np.exp(-(distances ** 2))

        # Step 4: Train the output layer weights using linear regression
        self.weights = np.linalg.lstsq(rbf_outputs, y, rcond=None)[0]

    def predict(self, X):
        # Step 2: Calculate the distance between each test example and the k centroids
        distances = cdist(X, self.centroids)

        # Step 3: Compute the RBF outputs
        rbf_outputs = np.exp(-(distances ** 2))

        # Step 5: Predict the output
        return np.dot(rbf_outputs, self.weights)

# Sample usage
# Load a dataset (e.g. iris dataset)


from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RBF network on the training set
rbf = RBFNetwork(k=50)
rbf.fit(X_train, y_train)

# Predict the output on the test set
y_pred = rbf.predict(X_test)

# Evaluate the performance of the model using accuracy score
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred.round())
print(f"Accuracy score: {acc:.3f}")
