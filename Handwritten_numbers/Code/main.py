from classifier import NNClassifier
import plotting
import processing
import numpy as np
import time
import clustering
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import utilities



def main():
    #  Defining data set parameters
    testX = processing.testX
    testY = processing.testY
    trainX = processing.trainX
    trainY = processing.trainY

    clusters, clusterLabels = clustering.make_clusters(trainX, trainY)
    correctTestLabels = set(testY)

    plot = 'cNN'

    match plot:
        case 'NN':
            predictions = utilities.get_total_predictions()

            startK1 = time.time()
            classifier = NNClassifier(K = 1)
            print('Fitting training set')
            classifier.fit(trainX, trainY)
            print('Finding errorRate')
            N = len(testY)
            #errorRate = np.count_nonzero(testY - predictions)/N
            print('Plotting confusion matrix')
            #conf_matrix = confusion_matrix(testY, predictions)
            endK1 = time.time()

            plotting.plot_successful_predictions(predictions, testX, testY, filename='NN_s_preds.png')
            plotting.plot_failed_predictions(predictions, testX, testY, filename='NN_f_preds.png')
            #plotting.plot_confusion_matrix(conf_matrix, correctTestLabels, errorRate, filename='NN_confmatr.png', plot = True)
        case 'cNN':
            startK1 = time.time()
            classifier = NNClassifier(K = 1)

            classifier.fit(clusters, clusterLabels)
            errorRate = classifier.errorRate(testX, testY)
            preds = classifier.predict(testX)
            conf_matrix = confusion_matrix(testY, preds)
            endK1 = time.time()

            print(f'Time for cluster KNN, for K = 1; {endK1-startK1}')

            print(f'errorRate for clustered KNN, K = 1: {errorRate}')
            plotting.plot_confusion_matrix(conf_matrix, correctTestLabels, errorRate, filename='cNN_confmatr.png', plot=True)
            plotting.plot_successful_predictions(preds, testX, testY, filename='cNN_s_preds.png')
            plotting.plot_failed_predictions(preds, testX, testY, filename='cNN_f_preds.png')
        case 'c7NN':
            startK1 = time.time()
            classifier = NNClassifier(K = 7)

            classifier.fit(clusters, clusterLabels)
            #errorRate = classifier.errorRate(testX, testY)
            preds = classifier.predict(testX)
            #conf_matrix = confusion_matrix(testY, preds)
            endK1 = time.time()

            print(f'Time for cluster KNN, for K = 1; {endK1-startK1}')

            #print(f'errorRate for clustered KNN, K = 1: {errorRate}')
            #plotting.plot_confusion_matrix(conf_matrix, correctTestLabels, errorRate, filename='c7NN_confmatr_correct.png', plot=True)
            plotting.plot_successful_predictions(preds, testX, testY, filename='c7NN_s_preds.png')
            plotting.plot_failed_predictions(preds, testX, testY, filename='c7NN_f_preds.png')
        case _:
            print('Non-acceptable input in match-case.')



if __name__ == '__main__':
    main()