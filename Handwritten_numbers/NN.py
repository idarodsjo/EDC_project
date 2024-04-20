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
from scipy.spatial import distance
import datetime

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
    sorted_labels = np.argsort(training_labels)
    sorted_data = np.empty_like(training_data)

    for i in range(len(training_labels)):
        sorted_data[i] = training_data[sorted_labels[i]]

    return sorted_data 


# Counting samples in each class
def count_samples(classList):
    sample_count = []
    for class_i in range(len(classList)):
        sample_count.append(len(classList[class_i]))

    return sample_count


""" plt.subplot(2,1,1)
plt.imshow(class_list[0])
plt.subplot(2,1,2)
plt.imshow(class_list[57000])
plt.show() """

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




#---------------------------------------------------------
#   **************************************************
#   |                                                |
#   |                       NN                       |
#   |                                                |
#   **************************************************
#---------------------------------------------------------

class NN():
    def __init__(self, K=5):
        self.K = K
    def fit(self, training_data, training_labels):
        self.training_data = training_data
        self.training_labels = training_labels

# First NN without clustering
def NN_classifier(self, test_data):
    classifications = []
    successful_classifications = []
    failed_classifications = []

    for i in range(len(test_data)):
        eucledian_distance = []

        for j in range(len(self.training_data)):
            eucledian_distance.append(distance.euclidean(test_data[i], self.training_data[j]))
        
        NN_index = np.argmin(eucledian_distance)

        if test_labels[i] != training_labels[NN_index]:
            failed_classifications.append([test_data[i], training_data[NN_index]])
        else:
            successful_classifications.append([test_data[i], training_data[NN_index]])

        classifications.append(self.training_labels[NN_index])

    return classifications, successful_classifications, failed_classifications



#---------------------------------------------------------
#   **************************************************
#   |                                                |
#   |              CONFUSION MATRIX                  |
#   |                                                |
#   **************************************************
#---------------------------------------------------------

def get_confusion_matrix(classifications):
    confusion_matrix = np.zeros((numClasses, numClasses))

    for i, x in enumerate(classifications):
        confusion_matrix[test_labels[i], x] += 1
    
    return confusion_matrix

def get_normalised_confusion_matrix(classifications):
    confusion_matrix = np.zeros((numClasses, numClasses))

    for i, x in enumerate(classifications):
        confusion_matrix[test_labels[i], x] += 1

    return confusion_matrix / np.amax(confusion_matrix)