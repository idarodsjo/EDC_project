import Delete_later.data_processing as Processing
import numpy as np
import time
import Delete_later.help_func as Utilities
from sklearn.cluster import KMeans

# This file contains the KNN classifier

def calculateEuclidianDistances(trainX, testX):
    print('Calculating Euclidian distances ...')
    euclidianDistances = np.array([])

    start = time.time()

    for testImage in testX:
        distances = [np.linalg.norm(testImage - trainImage) for trainImage in trainX]
        #euclidianDistances.append(distances)
        euclidianDistances = np.append(euclidianDistances, distances)
        print(f'Status: {len(euclidianDistances)} / 10 000 have been calculated')

    end = time.time()
    print(f'Calculation time: {end-start}')

    return euclidianDistances
    
def cluster_data(num_clusters, trainX, training_labels, num_classes):
    start = time.time()

    class_list = Utilities.split_data(trainX, training_labels)
    class_list_flattened = class_list.flatten().reshape(60000, 784)

    cluster_matrix = np.empty((num_classes, num_clusters, 784))
    print(f'Cluster matrix shape: {cluster_matrix.shape}')

    classes = np.unique(training_labels)
    for class_i in enumerate(classes):
        cluster = KMeans(n_clusters=num_clusters, random_state=0).fit(class_list_flattened).cluster_centers_
        print(f'Cluster shape: {cluster.shape}')
        cluster_matrix[class_i] = cluster
        print(class_i)
    
    end = time.time()
    cluster_matrix_flattened = cluster_matrix.flatten().reshape(num_classes * num_clusters, 784)

    clusterTime = end - start

    return cluster_matrix_flattened, clusterTime


def get_clusters(dataset, labels, classes, n_clusters=64):
    label_and_clusters = [] # [ (label, clusters=[]) ]
    for curr_class in classes:
        indices = np.where(labels == curr_class)[0]
        data_of_curr_class = dataset[indices]

        clusters = KMeans(n_clusters=n_clusters).fit(data_of_curr_class).cluster_centers_

        label_and_clusters.append( (curr_class, clusters) )

    return label_and_clusters

def get_cluster_dataset(dataset, labels, n_clusters=64):
    classes = range(10)
    n_classes = len(classes)
    label_and_clusters = get_clusters(dataset, labels, classes, n_clusters)

    train_data = []
    train_labels = [ [curr_class]*n_clusters for curr_class in classes ]
    train_labels = np.array(train_labels).reshape(n_clusters*n_classes)
    for (label, clusters) in label_and_clusters:
        train_data.extend(clusters)

    return (train_data, train_labels)



def KNNClassifier(euclidianDistances, trainY, testY, K=1):
    # MAKE INTO NP ARRAYS!!! MUCH FASTER COMPUTAION
    classifiedLabels = np.array([])
    classifiedSuccessIndices = np.array([])
    classifiedFailedIndices = []

    print('About to iterate through distances')

    for i, testDistances in enumerate(euclidianDistances):
        closestTrainIndex = np.argmin(testDistances)
        label = trainY[closestTrainIndex]
        classifiedLabels = np.append(classifiedLabels, label)

        if label == testY[i]:   # ERROR: Iterates until index = 10000, but list only goes to 9999
            classifiedSuccessIndices = np.append(classifiedSuccessIndices, i)
            print(f'Label = {label}, Predicted label = {testY[i]}, for i = {i}')
        else:
            classifiedFailedIndices = np.append(classifiedFailedIndices, i)
    
    return classifiedLabels, classifiedSuccessIndices, classifiedFailedIndices
