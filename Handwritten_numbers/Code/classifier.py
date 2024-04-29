import numpy as np

# Making class that implements KNN algo
class NNClassifier:
    def __init__(self, K = 1):

        if K%2 != 1:
            raise ValueError("K is not odd, votes may tie!")
        
        self.K = K

    def fit(self, trainX, trainY):
        """
        Saves all images, along with their class.
        """
        self.data = trainX
        self.labels = trainY

        self.N = len(self.labels)

    def predict(self, data):
        """Returns the predicted class for each sample in data
        """
        N = len(data)
        preds = np.zeros(N)

        for i in range(N):
            preds[i] = self.predictSample(data[i])

        return preds.astype(int)

    def predictSample(self, data):
        """Predicts the class of a single sample by comparing its distance to all samples in the training set,
        and deciding based an a majority vote between the K Nearest Neighbors.
        """
        best_val = np.ones(self.K)*np.inf
        best_idx = -np.ones(self.K).astype(int)

        for i in range(self.N):
            d = self.euclideanDistance(self.data[i], data)
            if d < np.max(best_val):
                ix = np.argmax(best_val) #replace highest value
                best_val[ix] = d
                best_idx[ix] = i

        labs = self.labels[best_idx]
        uniqueLabels, counts = np.unique(labs, return_counts=True)

        return uniqueLabels[np.argmax(counts)]
    

    def errorRate(self, testX, testY):
        """
        Returns proportion of correct predictions on the test data and labels given.        
        """
        N = len(testY)
        preds = self.predict(testX)
        return np.count_nonzero(testY - preds)/N

    def euclideanDistance(self, a, b):
        return np.linalg.norm(a - b, 2)