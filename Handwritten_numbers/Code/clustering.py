import numpy as np
from sklearn.cluster import KMeans

def make_clusters(trainX, trainY):
    M = 64 #num clusters, per class

    clusters = np.zeros((10, M, 28*28))
    for i in range(10):
        clusters[i] = KMeans(n_clusters = M, n_init='auto').fit(trainX[trainY == i]).cluster_centers_

    clusters = clusters.reshape(10*M, 28*28)
    clusterLabels = np.tile(np.arange(10), M).reshape(M, 10).T.flatten()

    indices = np.arange(10*M)
    np.random.shuffle(indices)

    clusters = clusters[indices]
    clusterLabels = clusterLabels[indices]
    
    return clusters, clusterLabels
