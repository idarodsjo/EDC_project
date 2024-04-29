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
import scipy
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

class_list = split_data(training_data, training_labels)

plt.subplot(2,1,1)
plt.imshow(training_data[0])
plt.subplot(2,1,2)
plt.imshow(training_data[57000])
plt.show()

#---------------------------------------------------------
#   **************************************************
#   |                                                |
#   |                 CLUSTER DATA                   |
#   |                                                |
#   **************************************************
#---------------------------------------------------------

# trains the data

def cluster_data(num_clusters, training_data, training_labels):
    start = time.time()

    classList = split_data(training_data, training_labels)
    classListFlattened = classList.flatten().reshape(60000, 784)

    clusterMatrix = np.empty((numClasses, num_clusters, 784))

   # before = 0   # ?
    #after = 0   # ?

    for class_i, i in enumerate(classes):
        #after += i
        cluster = KMeans(n_clusters=num_clusters, random_state=0).fit(classListFlattened).cluster_centers_
        #before = after
        clusterMatrix[class_i] = cluster
        print(class_i)

    end = time.time()
    clusterMatrixFlattened = clusterMatrix.flatten().reshape(numClasses * num_clusters, 784)
    return clusterMatrixFlattened

clusterMatrix = cluster_data(64, training_data, training_labels)


#---------------------------------------------------------
#   **************************************************
#   |                                                |
#   |               EUCLIDIAN DISTANCE               |
#   |                                                |
#   **************************************************
#---------------------------------------------------------

def get_euclidian_distance(training_chunk, test_chunk):
    return scipy.spatial.distance_matrix(training_chunk, test_chunk)



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
def KNN_classifier(test_data_chunk, train_data_chunk, train_labels_chunk, K):
    distance = get_euclidian_distance(train_data_chunk, test_data_chunk)

    nearest_indices = np.argpartition(distance, K, axis=0)[:K]
    nearest_labels = train_labels_chunk[nearest_indices]

    return scipy.stats.mode(nearest_labels, keepdims=False).mode

#---------------------------------------------------------
#   **************************************************
#   |                                                |
#   |                   testING                      |
#   |                                                |
#   **************************************************
#---------------------------------------------------------

def test_KNN_classifier(self, num_chunks = 60, K = 1):
    print('testing KNN classifier for K = ' + str(K))

    start_time = time.time()

    self.num_chunks = num_chunks
    self.K = K


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

def get_error_rate(confusion_matrix):
    error = np.trace(confusion_matrix)
    error_rate = 1 - (error / np.sum(confusion_matrix))
    
    return round(error_rate, 4)


#---------------------------------------------------------
#   **************************************************
#   |                                                |
#   |                   PLOTTING                     |
#   |                                                |
#   **************************************************
#---------------------------------------------------------

def plot_digit_image(data, index):
    plt.imshow(data[index], cmpap=plt.get_cmap('gray'))


def plot_confusion_matrix(title, confusion_matrix, error_rate, visualise):
    if visualise:
        plt.figure(figsize = (10.5, 8))
        plt.title(f'Confusion matrix for {title} \n Error rate: {str(error_rate*100)}')
        sns.heatmaå(confusion_matrix, annot=True, fmt='.0f')
        plt.xlabel('Classified label')
        plt.ylabel('True label')
        plt.show()