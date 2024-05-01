import numpy as np
from sklearn.cluster import KMeans

def make_clusters(trainX, trainY):
    numClustersInClass = 64
    flattenedPixture = 28 * 28
    numClasses = 10

    # Initialize an array to store cluster centers
    clusters = np.zeros((numClasses, numClustersInClass, flattenedPixture))
    
    # Perform clustering for each class
    for i in range(numClasses):
        # Select samples belonging to the current class
        classSamples = trainX[trainY == i]
        
        # Apply KMeans clustering to the class samples
        kmeans = KMeans(n_clusters=numClustersInClass, n_init='auto')
        clusters[i] = kmeans.fit(classSamples).cluster_centers_

    # Reshape the clusters array for easier indexing
    clusters = clusters.reshape(numClasses * numClustersInClass, flattenedPixture)

    # Generate cluster labels
    clusterLabels = np.repeat(np.arange(numClasses), numClustersInClass)

    # Shuffle the clusters and labels in the same order
    indices = np.arange(numClasses * numClustersInClass)
    np.random.shuffle(indices)
    clusters = clusters[indices]
    clusterLabels = clusterLabels[indices]

    return clusters, clusterLabels
