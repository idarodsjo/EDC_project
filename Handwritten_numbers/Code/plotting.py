import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix(conf_matrix, correctTestLabels, errorRate = None, plot = True, filename = None):    
    plt.figure(figsize=(8, 8)) #size in inches
    sns.heatmap(conf_matrix, annot=True, fmt='g', linewidths=.5, square = True, cmap = 'Spectral_r', xticklabels=correctTestLabels, yticklabels=correctTestLabels)
    plt.ylabel('Actual label', size=12)
    plt.xlabel('Predicted label', size=12)
    if errorRate is not None:
        #plt.title('(K = {fnumKf}%)NN - Accuracy errorRate: {ferrorRate:.{precision}f}%'.format(ferrorRate = errorRate*100, precision = 1), size = 20)
        plt.title(f'Error rate: {round(errorRate*100, 2)}%', size = 18)
    if filename is not None:
        plt.savefig(filename)
    if plot:
        plt.show()
    plt.close()

def plot_temp0_of_digits(clusters):
    n = 0 #some number between 0 and M

    fig, axs = plt.subplots(2, 5)
    fig.set_size_inches(20, 8)

    for i in range(2):
        for j in range(5):
            axs[i, j].imshow(clusters[i*5 + j, n].reshape(28, 28))
            axs[i, j].set_title(f'Template {n} for digit {i*5 + j}')
            axs[i, j].axis('off')

    plt.show()

def plot_temps_digit(M, clusters, digit):
    m = np.sqrt(M).astype(int)

    fig, axs = plt.subplots(m, m)
    fig.set_size_inches(8, 8)
    n= 0
    for i in range(m):
        for j in range(m):
            axs[i, j].imshow(clusters[digit, n].reshape(28, 28))
            axs[i, j].axis('off')
            n = n+1

    plt.show()

def plot_failed_predictions(predictions, testX, testY, filename = None):
    n = 4
    fig, axs = plt.subplots(1,n)
    fig.set_size_inches(20, 5)
    error_idx = np.nonzero(predictions - testY)[0]
    corr_idx = np.delete(predictions, error_idx)
    print(corr_idx)

    for i in range(n):
        ix = error_idx[i+5]
        axs[i].imshow(testX[ix].reshape(28, 28))
        axs[i].set_title(f'Class {testY[ix]}, Predicted {int(predictions[ix])}')
        axs[i].axis('off')
    if filename is not None:
        plt.savefig(filename)

def plot_successful_predictions(predictions, testX, testY, filename = None):
    n = 4
    fig, axs = plt.subplots(1,n)
    fig.set_size_inches(20, 5)
    correct_idx = np.where(predictions - testY == 0)[0]

    for i in range(n):
        ix = correct_idx[i+5]
        axs[i].imshow(testX[ix].reshape(28, 28))
        axs[i].set_title(f'Class {testY[ix]}, Predicted {int(predictions[ix])}')
        axs[i].axis('off')
    if filename is not None:
        plt.savefig(filename)
