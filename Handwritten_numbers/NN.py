# Link to read script: https://www.kaggle.com/code/hojjatk/read-mnist-dataset/notebook
# Link to data files: https://www.kaggle.com/datasets/hojjatk/mnist-dataset?select=t10k-images-idx3-ubyte
# KNN classifier tutorial: https://www.kaggle.com/code/prashant111/knn-classifier-tutorial

from keras.datasets import mnist
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization purposes
import seaborn as sns # for data visualization
import time
from sklearn.cluster import KMeans

(training_data, training_labels), (test_data, test_labels) = mnist.load_data()

# Reshaping the data to fit the model

training_data = training_data.reshape(60000, 28, 28, 1)
training_labels = training_labels.reshape(60000)
test_data = test_data.reshape(10000, 28, 28, 1)
test_labels = test_labels.reshape(10000)

# Some relevant parameters for alter use

classes = np.unique(training_labels)
numClasses = len(classes)
trainingSetSize = len(training_data)


# Seperate data into each class

def split_data(training_data, training_labels):
    classList = np.empty_like(training_data)

    # Making classList into a nested list (eash element is a list containing one class, element index [0-9] = class)
    """ for i in range(numClasses):
        #single_class = []
        #classList.append(single_class)
        classList[i] = training_data[training_labels[i]] """

    # Sorting data into each class in classList
    """ for i in range(trainingSetSize):
        classList[training_labels[i]].append(training_data[i]) """
    for i in range(numClasses):
        classList[i] = training_data[training_labels[i]]
    
    return classList


# Counting samples in each class
def count_samples(classList):
    sample_count = []
    for class_i in range(len(classList)):
        sample_count.append(len(classList[class_i]))

    return sample_count


#---------------------------------------------------------
#   **************************************************
#   |                                                |
#   |                 CLUSTER DATA                   |
#   |                                                |
#   **************************************************
#---------------------------------------------------------

def cluster_data(num_clusters, training_data, training_labels):
    start = time.time()

    classList = split_data(training_data, training_labels)
    classListFlattened = classList.flatten().reshape(60000, 784)

    clusterMatrix = np.empty((numClasses, num_clusters, 784))

    before = 0   # ?
    after = 0   # ?

    for class_i, i in enumerate(classes):
        after += i
        cluster = KMeans(n_clusters=num_clusters, random_state=0).fit(classListFlattened).cluster_centers_
        before = after
        clusterMatrix[class_i] = cluster
        print(class_i)

    end = time.time()
    clusterMatrixFlattened = clusterMatrix.flatten().reshape(numClasses * num_clusters, 784)
    return clusterMatrixFlattened

clusterMatrix = cluster_data(64, training_data, training_labels)

print('Cluster matrix size :')
print(clusterMatrix.size)

# POTENTIAL PROBLEM: Only finds 8 clusters in each class

