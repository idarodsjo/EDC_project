from classifier import KNNClassifier
import plotting
import processing
import numpy as np
import time
import clustering
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import chunking



def main():
    #  Defining data set parameters
    testX = processing.testX
    testY = processing.testY
    trainX = processing.trainX
    trainY = processing.trainY

    print(trainX.shape)
    print(testX.shape)
    print(testY.shape)

    clusters, clusterLabels = clustering.make_clusters(trainX, trainY)
    correctTestLabels = set(testY)


    plot = 'cNN'

    match plot:
        case 'NN':
            predictions = chunking.get_all_predictions()

            startK1 = time.time()
            classifier = KNNClassifier(K = 1)
            print('Fitting training set')
            classifier.fit(trainX, trainY)
            print('Finding errorRate')
            N = len(testY)
            errorRate = np.count_nonzero(testY - predictions)/N
            print(f'Error rate : {errorRate}')
            confusionMatrix = confusion_matrix(testY, predictions)
            endK1 = time.time()

            plotting.plot_successful_predictions(predictions, testX, testY, filename='NN_s_preds.png')
            plotting.plot_failed_predictions(predictions, testX, testY, filename='NN_f_preds.png')
            plotting.plot_confusion_matrix(confusionMatrix, correctTestLabels, errorRate, filename='NN_confmatr.png', plot = True)
        case 'cNN':
            startK1 = time.time()
            classifier = KNNClassifier(K = 1)

            classifier.fit(clusters, clusterLabels)
            errorRate = classifier.errorRate(testX, testY)
            preds = classifier.predict(testX)
            confusionMatrix = confusion_matrix(testY, preds)
            endK1 = time.time()

            print(f'Time for cluster KNN, for K = 1; {endK1-startK1}')

            print(f'errorRate for clustered KNN, K = 1: {errorRate}')
            plotting.plot_confusion_matrix(confusionMatrix, correctTestLabels, errorRate, filename='cNN_confmatr.png', plot=True)
            plotting.plot_successful_predictions(preds, testX, testY, filename='cNN_s_preds.png')
            plotting.plot_failed_predictions(preds, testX, testY, filename='cNN_f_preds.png')
        case 'c7NN':
            startK1 = time.time()
            classifier = KNNClassifier(K = 7)

            classifier.fit(clusters, clusterLabels)
            errorRate = classifier.errorRate(testX, testY)
            preds = classifier.predict(testX)
            confusionMatrix = confusion_matrix(testY, preds)
            endK1 = time.time()

            print(f'Time for cluster KNN, for K = 1; {endK1-startK1}')

            #print(f'errorRate for clustered KNN, K = 1: {errorRate}')
            plotting.plot_confusion_matrix(confusionMatrix, correctTestLabels, errorRate, filename='c7NN_confmatr_correct.png', plot=True)
            plotting.plot_successful_predictions(preds, testX, testY, filename='c7NN_s_preds.png')
            plotting.plot_failed_predictions(preds, testX, testY, filename='c7NN_f_preds.png')
        case 'errors_KNN':
            startK1 = time.time()
            classifier = KNNClassifier(K = 7)

            classifier.fit(clusters, clusterLabels)
            #errorRate = classifier.errorRate(testX, testY)
            preds = classifier.predict(testX)
            #confusionMatrix = confusion_matrix(testY, preds)
            endK1 = time.time()

            print(f'Time for cluster KNN, for K = 1; {endK1-startK1}')

            #print(f'errorRate for clustered KNN, K = 1: {errorRate}')
            #plotting.plot_confusion_matrix(confusionMatrix, correctTestLabels, errorRate, filename='c7NN_confmatr_correct.png', plot=True)
            plotting.plot_successful_predictions(preds, testX, testY, filename='c7NN_s_preds.png')
            plotting.plot_failed_predictions(preds, testX, testY, filename='c7NN_f_preds.png')
        case 'temps':
            plotting.plot_cluster_n_all_numbers(clustering, 5, filename='temp_5_all_nums.png')
        case _:
            print('Non-acceptable input in match-case.')



if __name__ == '__main__':
    main()