import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
import scipy.io
import time

# Not necessary due to loadmat() reading given data
""" 
from keras.datasets import mnist
(trainX, trainY), (testX, testY) = mnist.load_data()

trainX = np.array(trainX).astype(float)
trainY = np.ravel(trainY)
testX = np.array(testX).astype(float)
testY = np.ravel(testY)

"""

mat = scipy.io.loadmat("./MNist_ttt4275/data_all.mat")
#load
trainX = np.array(mat['trainv']).astype(float)
trainY = np.array(mat['trainlab']).ravel() # flattened reference to array
testX = np.array(mat['testv']).astype(float)
testY = np.array(mat['testlab']).ravel() # flattened reference to array

correctTestLabels = set(testY)
indices = np.arange(len(trainX))


#       NON-SHUFFLED
#   Subsets of data
chunkSize = 1000
subsetTrainX = trainX[:chunkSize]
subsetTrainY = trainY[:chunkSize]
subsetTestX = testX[:chunkSize]
subsetTestY = testY[:chunkSize]


# Normalising data
def normalise(data):
    """
    min-max normalize the elements of data.

    data: tensor with values to be normalized.
    """
    value_range = np.max(data) - np.min(data)
    normalized_data = (data - np.min(data))/value_range
     
    return normalized_data

#normalize training and test data
subsetTrainX = normalise(subsetTrainX)
subsetTestX = normalise(subsetTestX)
trainX = normalise(trainX)
testX = normalise(testX)

# plotting confusion matrix
def plot_confusion_matrix(conf_matrix, numK, errorRate = None, plot = True, filename = None):    
    plt.figure(figsize=(8, 8)) #size in inches
    sns.heatmap(conf_matrix, annot=True, fmt='g', linewidths=.5, square = True, cmap = 'Spectral_r', xticklabels=correctTestLabels, yticklabels=correctTestLabels)
    plt.ylabel('Actual label', size=12)
    plt.xlabel('Predicted label', size=12)
    if errorRate is not None:
        #plt.title('(K = {fnumKf}%)NN - Accuracy errorRate: {ferrorRate:.{precision}f}%'.format(ferrorRate = errorRate*100, precision = 1), size = 20)
        plt.title(f'Error rate: {round(errorRate*100, 2)}%', size = 18)
    if filename is not None:
        plt.savefig(filename)
    if plot:
        plt.show()
    plt.close()

# Making class that implements KNN algo
class NNClassifier:
    def __init__(self, K = 1):

        if K%2 != 1:
            raise ValueError("K should be odd, so votes don't tie")
        
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

def get_total_predictions():
    total_predictions = np.array([])
    for n in range(10):
        prediction_n = np.loadtxt(f'Task1_data/Predictions/preds{n}.txt', dtype=int)
        total_predictions = np.append(total_predictions, prediction_n)
        print(f'File number {n}, Prediction list length = {total_predictions.shape}')
    return total_predictions

total_preds = get_total_predictions()
total_errorRate = 1-0.9691
conf_matrix = confusion_matrix(testY, total_preds)
plot_confusion_matrix(conf_matrix, 1, total_errorRate, plot=False, filename='NN_confmatr.png')

n = 5
fig, axs = plt.subplots(1,n)
fig.set_size_inches(20, 5)
error_idx = np.nonzero(total_preds - testY)[0]

for i in range(n):
    ix = error_idx[i+6]
    axs[i].imshow(testX[ix].reshape(28, 28))
    axs[i].set_title(f'Class {testY[ix]}, Predicted {int(total_preds[ix])}')
    axs[i].axis('off')
plt.savefig('misclassified_img_NN.png')

# Testing algo for K = 1
start = time.time()
classifier = NNClassifier(K = 1)


#####preds = classifier.predict(testX[chunks*1000:(chunks+1)*1000])
""" startChunk = 8
numChunksTesting = 2
total_errorRate = np.array([])
total_preds = np.array([])
testXChunks = np.array_split(testX, 10)
testXChunks = np.array(testXChunks)

for chunks in range(numChunksTesting):
    classifier.fit(trainX, trainY)
    print(f'Predicting chunk {chunks+startChunk}')
    preds = classifier.predict(testXChunks[chunks+startChunk])
    np.savetxt(f'preds{chunks+startChunk}.txt', preds)
    total_preds = np.append(total_preds, preds)
    print(f'Calculating errorRate for chunk {chunks + startChunk}')
    errorRate = classifier.errorRate(testXChunks[chunks + startChunk], testY[(chunks + startChunk)*1000:(chunks+1 + startChunk)*1000])
    total_errorRate = np.append(total_errorRate, errorRate)
    np.savetxt(f'errorRate{startChunk}{startChunk+numChunksTesting-1}.txt', total_errorRate)
    
 """
#np.savetxt('preds.txt', total_preds) # use np.loadtxt() to extract preds --> this will interpret each row as float element... YAY!


conf_matrix = confusion_matrix(testY, total_preds)
end = time.time()
print(f'Time used to classify KNN and generate confusion matrix for no clustering is {end - start}')





############         Include when all data is gathered
""" 
plot_confusion_matrix(conf_matrix, 1, errorRate, plot=False, filename='NN_confmatr.png')


# Plotting misclassified digits
n = 4
fig, axs = plt.subplots(1, n)
fig.set_size_inches(20, 5)
print(f'Shape of total preds: {total_preds.shape}, shape of test chunks: {testX[0:numChunksTesting * 1000].shape}')
error_idx = np.nonzero(total_preds - testY)[0]

for i in range(n):
    ix = error_idx[i]
    axs[i].imshow(testX[ix].reshape(28, 28))
    axs[i].set_title(f'Class {testY[ix]}, Predicted {round(total_preds[ix], 1)}')
    axs[i].axis('off')
plt.savefig('misclassified_img_NN.png')
plt.show() 

 """







################################################
#                  PART TWO
#           CLustering M = 64 per class

M = 64 #num clusters, per class

clusters = np.zeros((10, M, 28*28))

for i in range(10):
    clusters[i] = KMeans(n_clusters = M, n_init='auto').fit(trainX[trainY == i]).cluster_centers_


def plot_temp0_of_digits():
    n = 0 #some number between 0 and M

    fig, axs = plt.subplots(2, 5)
    fig.set_size_inches(20, 8)

    for i in range(2):
        for j in range(5):
            axs[i, j].imshow(clusters[i*5 + j, n].reshape(28, 28))
            axs[i, j].set_title(f'Template {n} for digit {i*5 + j}')
            axs[i, j].axis('off')

    plt.show()


def plot_temps_digit(digit):
    m = np.sqrt(M).astype(int)

    fig, axs = plt.subplots(m, m)
    fig.set_size_inches(8, 8)
    n= 0
    for i in range(m):
        for j in range(m):
            axs[i, j].imshow(clusters[digit, n].reshape(28, 28))
            axs[i, j].axis('off')
            n = n+1

    plt.show()


clusters = clusters.reshape(10*M, 28*28)
clusterLabels = np.tile(np.arange(10), M).reshape(M, 10).T.flatten() #should be a better way to do this, but this works

indices = np.arange(10*M)
np.random.shuffle(indices)

clusters = clusters[indices]
clusterLabels = clusterLabels[indices]

""" 

############## K=1 with clusters (clusters)

startK1 = time.time()
classifier = NNClassifier(K = 1)

classifier.fit(clusters, clusterLabels)
errorRate = classifier.errorRate(testX, testY)
preds = classifier.predict(testX)
conf_matrix = confusion_matrix(testY, preds)
endK1 = time.time()

print(f'Time for cluster KNN, for K = 1; {endK1-startK1}')

print(f'errorRate for clustered KNN, K = 1: {errorRate}')
plot_confusion_matrix(conf_matrix, 1, errorRate)



################# K=3
numNeighbors = 3
start = time.time()
classifier = NNClassifier(K = numNeighbors)

classifier.fit(clusters, clusterLabels)
errorRate = classifier.errorRate(testX, testY)
preds = classifier.predict(testX)
conf_matrix = confusion_matrix(testY, preds)
end = time.time()

print(f'Time for cluster KNN, for K = {numNeighbors}; {end-start}')

print(f'errorRate for clustered KNN, K = {numNeighbors}: {errorRate}')
plot_confusion_matrix(conf_matrix, 3, errorRate)


################# K=5
numNeighbors = 5
start = time.time()
classifier = NNClassifier(K = numNeighbors)

classifier.fit(clusters, clusterLabels)
errorRate = classifier.errorRate(testX, testY)
preds = classifier.predict(testX)
conf_matrix = confusion_matrix(testY, preds)
end = time.time()

print(f'Time for cluster KNN, for K = {numNeighbors}; {end-start}')

print(f'errorRate for clustered KNN, K = {numNeighbors}: {errorRate}')
plot_confusion_matrix(conf_matrix, 5, errorRate)


################ K=7 NN with clusters (clusters)

startK7 = time.time()
classifier = NNClassifier(K = 7)

classifier.fit(clusters, clusterLabels)
errorRate = classifier.errorRate(testX, testY)
preds = classifier.predict(testX)
conf_matrix = confusion_matrix(testY, preds)
endK7 = time.time()
print(f'Time for cluster KNN, for K = 7; {endK7-startK7}')
print(f'errorRate for clustered KNN, K = 7: {errorRate}')

plot_confusion_matrix(conf_matrix, 7, errorRate)

 """

""" 
def plot_failed_classifications():
    n = 5
    fig, axs = plt.subplots(1, n)
    fig.set_size_inches(20, 5)
    plt.title('Failed classifications for NN classifier with K = 7 and M = 64')

    error_idx = np.nonzero(preds - testY)[0]

    for i in range(n):
        ix = error_idx[i]
        axs[i].imshow(testX[ix].reshape(28, 28))
        axs[i].set_title(f'Class {testY[ix]}, Predicted {preds[ix]}')

    plt.show()

plot_failed_classifications()

 """