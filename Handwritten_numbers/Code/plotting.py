import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np

def plot_confusion_matrix(confusionMatrix, correctTestLabels, errorRate = None, plot = True, filename = None):    
    plt.figure(figsize=(8, 8)) #size in inches
    sns.heatmap(confusionMatrix, annot=True, fmt='g', linewidths=.5, square = True, cmap = 'flare', norm=matplotlib.colors.LogNorm(), xticklabels=correctTestLabels, yticklabels=correctTestLabels)
    plt.ylabel('Actual label', size=12)
    plt.xlabel('Predicted label', size=12)
    if errorRate is not None:
        plt.title(f'Error rate: {round(errorRate*100, 2)}%', size = 18)
    if filename is not None:
        plt.savefig(filename)
    if plot:
        plt.show()
    plt.close()

def plot_cluster_n_all_numbers(clusters, n, filename = None, plot = False):
    if n < 0 or n > 64:
        print('Invalid input for cluster n, should be number between 0 and 63')
    fig, axs = plt.subplots(2, 5)
    fig.set_size_inches(20, 8)

    for i in range(2):
        for j in range(5):
            axs[i, j].imshow(clusters[i*5 + j, n].reshape(28, 28))
            axs[i, j].set_title(f'Template {n} for digit {i*5 + j}')
            axs[i, j].axis('off')
    if filename is not None:
        plt.savefig(filename)
    if plot:
        plt.show()

def plot_temps_digit(M, clusters, digit, filename = None, plot = False):
    m = np.sqrt(M).astype(int)

    fig, axs = plt.subplots(m, m)
    fig.set_size_inches(8, 8)
    n= 0
    for i in range(m):
        for j in range(m):
            axs[i, j].imshow(clusters[digit, n].reshape(28, 28))
            axs[i, j].axis('off')
            n = n+1
    if filename is not None:
        plt.savefig(filename)
    if plot:
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
        axs[i].set_title(f'Class {testY[ix]}, Predicted {int(predictions[ix])}', size=20)
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
        axs[i].set_title(f'Class {testY[ix]}, Predicted {int(predictions[ix])}', size=20)
        axs[i].axis('off')
    if filename is not None:
        plt.savefig(filename)
