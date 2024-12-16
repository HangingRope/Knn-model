import numpy as np
from collections import Counter

class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X_train, y_train):
        """Store the training data."""
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        """Predict the class for each point in X_test."""
        predictions = [self._predict_single(x) for x in X_test]
        return np.array(predictions)
    
    def _predict_single(self, x):
        """Helper method to predict the class of a single data point."""
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)[0][0]
        return most_common
    
    def _euclidean_distance(self, point1, point2):
        """Calculate Euclidean distance between two points."""
        return np.sqrt(np.sum((point1 - point2) ** 2))


