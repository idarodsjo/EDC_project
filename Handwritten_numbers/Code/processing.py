import numpy as np
import scipy.io

dataSet = scipy.io.loadmat("./MNist_ttt4275/data_all.mat")

trainX = np.array(dataSet['trainv']).astype(float)
trainY = np.array(dataSet['trainlab']).ravel() # flattened reference to array
testX = np.array(dataSet['testv']).astype(float)
testY = np.array(dataSet['testlab']).ravel() # flattened reference to array

correctTestLabels = set(testY)
indices = np.arange(len(trainX))

# Defining data subsets
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