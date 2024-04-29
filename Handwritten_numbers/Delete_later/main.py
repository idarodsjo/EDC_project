import Delete_later.classifier as Classifier
import Delete_later.data_processing as Processing
import numpy as np
from multiprocessing import Pool, Process
import Delete_later.help_func as Utilities
import Delete_later.plotting_funcs as Plotting

def calculateChunkDistances(chunk):
    return Classifier.calculateEuclidianDistances(Processing.trainX, chunk)

def main():
    numChunks = 10 
    chunkedTestX = np.array_split(Processing.testX, numChunks)
    chunkedTestX = np.array(chunkedTestX)
    numClusters = 64


    #clusterMatrix, _ = Classifier.get_clusters(numClusters, Processing.trainX, Processing.trainY, 10)
    train_data, train_label = Classifier.get_cluster_dataset(Processing.trainX, Processing.trainY, n_clusters=64)


    distances = Classifier.calculateEuclidianDistances(train_data, Processing.testX)

    print('Distances calculated')
    
    # USE THESE LATER; After Euclidian distances are saved
    classifiedLabels, successIndices, failIndices = Classifier.KNNClassifier(distances, Processing.trainY, Processing.testY, K=1)

    print('Labels are classified')

    confusionMatrix = Utilities.calculate_confusion_matrix(classifiedLabels, Processing.testY, 10)
    print('Confusion matrix found')
    errorRate = Utilities.calculate_error_rate(confusionMatrix)
    print('Error rate found')

    print('Plotting confusion matrix')
    Plotting.confusion_matrix('Confusion Matrix for KNN, where K = 1', confusionMatrix, errorRate, True)



"""     print(Processing.testX.shape)

    processes = np.array([])
    for i in range(numChunks):
        chunk = chunkedTestX[i].reshape(1000, 784)
        print(chunkedTestX.shape)
        print(chunk.shape)
        p = Process(target=calculateChunkDistances, args=(chunk,))
        processes = np.append(processes, p)
        p.start()

    # Joining processes after calculations
    for p in processes:
        p.join() """



    


if __name__ == '__main__':
    main()