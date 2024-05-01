import numpy as np

# Making class that implements KNN algo
class KNNClassifier:
    # Takes the value of K, decides if classifier is NN (K=1) or KNN (K>1)
    def __init__(self, K = 1):
        self.K = K
    
    # Saves training testX and labels 
    def fit(self, trainX, trainY):
        self.testX = trainX
        self.labels = trainY

        self.N = len(self.labels)

    def predictSample(self, testX):
        # Initialize arrays to store nearest distances and indices
        nearest_distances = np.full(self.K, np.inf)
        nearest_indices = np.full(self.K, -1, dtype=int)

        # Iterate through each sample in the training data
        for i, train_data in enumerate(self.testX):
            # Calculate Euclidean distance between current training data sample and testX
            distance = np.linalg.norm(train_data - testX)

            # Update nearest neighbors if current distance is smaller than the maximum in nearest_distances
            if distance < np.max(nearest_distances):
                max_index = np.argmax(nearest_distances)
                nearest_distances[max_index] = distance
                nearest_indices[max_index] = i

        # Extract labels of nearest neighbors
        nearest_labels = self.labels[nearest_indices]

        # Count occurrences of each unique label among nearest neighbors
        unique_labels, label_counts = np.unique(nearest_labels, return_counts=True)

        # Return the label that occurs most frequently among the nearest neighbors
        return unique_labels[np.argmax(label_counts)]

    # Predicts class for each data sample in testX
    def predict(self, testX):
        N = len(testX)
        preds = np.zeros(N)

        for i in range(N):
            preds[i] = self.predictSample(testX[i])

        return preds.astype(int)

    
    # Failed predictions divided by all predictions
    def errorRate(self, testX, testY):
        N = len(testY)
        preds = self.predict(testX)
        numFailed = np.count_nonzero(testY - preds)
        return numFailed/N

    # Finds diffierence between matrices, returns second order norm
    def euclideanDistance(x, y):
        return np.linalg.norm(x - y, 2)
