from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


###############################################################
##                                                           ##
##     READING DATA & PREPARING FOR TRAINING AND TESTING     ##
##                                                           ##
###############################################################

classes = ['setosa', 'versicolor', 'virginica']
features = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']

Classes = len(classes)              #3
Feature = len(features)             #4
alpha = 0.01                        #Step factor
N = 1000                            #Number of iterations
W = np.zeros([Classes, Feature])    #Matrix of classes and features


def read_data():
    """Read data and seperate into dataset without species, x, and with only species, t"""
    global iris, x, t
    iris = pd.read_csv('iris.csv')
    #iris_head = iris.head()
    #iris_describe = iris.describe()
    #iris_info = iris.info()
    #iris_sum = iris.isnull().sum()
    iris = iris.drop(columns=['Id'])
    #iris = iris.drop(columns=['Id', 'SepalWidthCm'])
    #iris = iris.drop(columns=['Id', 'SepalWidthCm', 'PetalWidthCm'])
    #iris = iris.drop(columns=['Id', 'SepalWidthCm', 'PetalWidthCm', 'PetalLengthCm'])
    le = LabelEncoder()     #Label Encoding for converting the labels into numeric form
    iris['Species'] = le.fit_transform(iris['Species'])
    x = iris.drop('Species', axis=1)
    t = iris.Species

def iris_training():
    """Training set with 30 first samples for both x and t.
       Training set with 30 last samples for both x and t."""
    #30 first samples for training
    global iris_x_training_first, iris_t_training_first, iris_x_training_last, iris_t_training_last, iris_t_training
    zeros_columns = np.zeros([90,2])
    iris_x_setosa_training_first, iris_t_setosa_training_first = x[:30], t[:30]
    iris_x_versicolor_training_first, iris_t_versicolor_training_first = x[50:80], t[50:80]
    iris_x_virginica_training_first, iris_t_virginica_training_first = x[100:130], t[100:130]
    iris_x_training_first = np.concatenate([iris_x_setosa_training_first, iris_x_versicolor_training_first, iris_x_virginica_training_first])
    iris_t_training_first = np.concatenate([iris_t_setosa_training_first, iris_t_versicolor_training_first, iris_t_virginica_training_first])
    iris_t_training_first = np.concatenate([iris_t_training_first[:, np.newaxis], zeros_columns], axis=1)
    iris_t_training = np.zeros((90,3))

    iris_t_training[:30, 0] = 1
    iris_t_training[30:60, 1] = 1
    iris_t_training[60:, 2] = 1

    #30 last samples for training
    iris_x_setosa_training_last, iris_t_setosa_training_last = x[20:50], t[20:50]
    iris_x_versicolor_training_last, iris_t_versicolor_training_last = x[70:100], t[70:100]
    iris_x_virginica_training_last, iris_t_virginica_training_last = x[-30:], t[-30:]
    iris_x_training_last = np.concatenate([iris_x_setosa_training_last, iris_x_versicolor_training_last, iris_x_virginica_training_last])
    iris_t_training_last = np.concatenate([iris_t_setosa_training_last, iris_t_versicolor_training_last, iris_t_virginica_training_last])
    iris_t_training_last = np.concatenate([iris_t_training_last[:, np.newaxis], zeros_columns], axis=1)

    return iris_x_training_first, iris_x_training_last, iris_t_training, iris_t_training_last, iris_t_training_first

def iris_test():
    """Test set with 20 last  samples for both x and t.
       Test set with 20 first samples for both x and t."""
    #20 last samples for testing
    global iris_x_testing_last, iris_t_testing_last, iris_x_testing_first, iris_t_testing_first
    zeros_columns = np.zeros([60,2])
    iris_x_setosa_testing_last, iris_t_setosa_testing_last = x[30:50], t[30:50]
    iris_x_versicolor_testing_last, iris_t_versicolor_testing_last = x[80:100], t[80:100]
    iris_x_virginica_testing_last, iris_t_virginica_testing_last = x[-20:], t[-20:]
    iris_x_testing_last = np.concatenate([iris_x_setosa_testing_last, iris_x_versicolor_testing_last, iris_x_virginica_testing_last])
    iris_t_testing_last = np.concatenate([iris_t_setosa_testing_last, iris_t_versicolor_testing_last, iris_t_virginica_testing_last])
    iris_t_testing_last = np.concatenate([iris_t_testing_last[:, np.newaxis], zeros_columns], axis=1)

    #20 first samples for testing
    iris_x_setosa_testing_first, iris_t_setosa_testing_first = x[30:50], t[30:50]
    iris_x_versicolor_testing_first, iris_t_versicolor_testing_first = x[80:100], t[80:100]
    iris_x_virginica_testing_first, iris_t_virginica_testing_first = x[-20:], t[-20:]
    iris_x_testing_first = np.concatenate([iris_x_setosa_testing_first, iris_x_versicolor_testing_first, iris_x_virginica_testing_first])
    iris_t_testing_first = np.concatenate([iris_t_setosa_testing_first, iris_t_versicolor_testing_first, iris_t_virginica_testing_first])
    iris_t_testing_first = np.concatenate([iris_t_testing_first[:, np.newaxis], zeros_columns], axis=1)

    return iris_x_testing_last, iris_t_testing_last, iris_x_testing_first, iris_t_testing_first

read_data()
iris_training()
iris_test()

###################################################################
##                                                               ##
##     EQUATIONS, FUNCTIONS & PLOTS FOR TRAINING AND TESTING     ##
##                                                               ##
###################################################################

def mse(g_k, t_k):
    """Eq:19 from compendium"""
    return 0.5*np.matmul(np.matrix.transpose(g_k-t_k), (g_k-t_k))

def sigmoid(z):
    """Eq:20 from compendium"""
    g_ik = np.array(1 / (1 + np.exp(-z)))
    return g_ik

def grad_mse_test(g, t, x):
    """Eq:21 and 22 from compendium"""
    delta_mse = g - t
    delta_g = np.multiply(g, 1 - g)
    delta_z = np.transpose(x)
    return np.dot(delta_z, np.multiply(delta_mse, delta_g))

def predicted_labels(g):
    return list(map(np.argmax, g))

def error_rate(predicted_labels, actual_labels):
    number_error = 0
    for i in range(len(actual_labels)):
        if not np.array_equal(actual_labels[i], predicted_labels[i]):
            number_error += 1
    error = number_error/len(actual_labels)
    return error

def iris_confusion_matrix(actual_labels, predictions, title):
    disp = ConfusionMatrixDisplay.from_predictions(actual_labels, predictions, display_labels=classes, cmap=sns.cubehelix_palette(as_cmap=True))
    disp.ax_.set_title(title)
    plt.show()

def iris_histogram(x, n, title):    
    for j in range (Classes):
        rows = x[n*j: n + n*j, :]
        for i in range(Feature):
            plt.hist(rows[:,i], color='indigo', edgecolor='black', bins = 17, alpha = 0.6)
            plt.xlabel('[cm]')
            plt.ylabel('Frequency')
            plt.title(features[i] + ' for ' + classes[j] + ', ' + title)
            plt.show()

def iris_histogram_feature(x, n, title):
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    colors = ['plum', 'darkgreen', 'indigo']

    for i in range(Feature):
        row = i // 2            #Determine the row index of the subplot
        col = i % 2             #Determine the column index of the subplot
        
        for j in range(Classes):
            rows = x[n*j: n + n*j, :]
            axs[row, col].hist(rows[:, i], color=colors[j % len(colors)], bins=17, alpha=0.5, label=classes[j])
        axs[row, col].set_xlabel('[cm]')
        axs[row, col].set_ylabel('Frequency')
        axs[row, col].set_title(features[i] + ', ' + title)
        axs[row, col].legend()

    plt.tight_layout()
    plt.show()

def training(W, X, T, n, a):
    for i in range(n):
        z = np.dot(X, np.transpose(W))
        g = sigmoid(z)
        mse_test = grad_mse_test(g, T, X)
        W = W - a*np.transpose(mse_test)
    return W, g

def testing(W, x):
    z = np.dot(x, np.transpose(W))
    g = sigmoid(z)
    return g

def find_best_alpha(alphas):
    best_alpha = None
    best_error_rate = float('inf')
    
    for alpha in alphas:
        w_training, g_training = training(W, iris_x_training_first, iris_t_training, N, alpha)
        
        g_validation = testing(w_training, iris_x_testing_first)
        predicted_labels_validation = predicted_labels(g_validation)
        error_rate_validation = error_rate(predicted_labels_validation, iris_t_testing_first)
        
        if error_rate_validation < best_error_rate:
            best_error_rate = error_rate_validation
            best_alpha = alpha
    
    return best_alpha, best_error_rate

#########################
##                     ##
##         MAIN        ##
##                     ##
#########################
def main():
    global iris_t_training_first, iris_t_training_last, iris_t_testing_last, iris_t_testing_first, iris_t_training
    w_training, g_training = training(W, iris_x_training_first, iris_t_training, N, alpha)
    g_testing = testing(w_training, iris_x_testing_last)
    predicted_labels_training = predicted_labels(g_training)
    predicted_labels_testing = predicted_labels(g_testing)

    iris_t_training_first = np.delete(iris_t_training_first, [1, 2], 1)
    iris_t_training_first = iris_t_training_first.flatten().astype(int).tolist()
    iris_t_training_last = np.delete(iris_t_training_last, [1, 2], 1)
    iris_t_training_last = iris_t_training_last.flatten().astype(int).tolist()
    iris_t_testing_last = np.delete(iris_t_testing_last, [1, 2], 1)
    iris_t_testing_last = iris_t_testing_last.flatten().astype(int).tolist()
    iris_t_testing_first = np.delete(iris_t_testing_first, [1, 2], 1)
    iris_t_testing_first = iris_t_testing_first.flatten().astype(int).tolist()


    #PLOTTING CONFUSION MATRIX AND HISTOGRAM
    #FIRST SET
    iris_confusion_matrix(iris_t_training_first, predicted_labels_training, 'Confusion matrix for training set')
    iris_confusion_matrix(iris_t_testing_last, predicted_labels_testing, 'Confusion matrix for testing set')
    #SECOND SET
    #iris_confusion_matrix(iris_t_training_last, predicted_labels_training, 'Confusion matrix for training set')
    #iris_confusion_matrix(iris_t_testing_first, predicted_labels_testing, 'Confusion matrix for testing set')
    #HISTOGRAM
    #iris_histogram(iris_x_training_first, 30, 'first training set')
    #iris_histogram(iris_x_testing_last, 20, 'first testing set')
    iris_histogram_feature(iris_x_training_first, 30, 'first training set')
    iris_histogram_feature(iris_x_testing_last, 20, 'first testing set')

    #PRINT ERROR RATE FOR TRAINING- AND TESTING
    #FIRST SET
    error_rate_training = error_rate(predicted_labels_training, iris_t_training_last)
    error_rate_test = error_rate(predicted_labels_testing, iris_t_testing_first)
    print('Error rate for training set is' ,error_rate_training)
    print('Error rate for test set is', error_rate_test)
    #SECOND SET
    #error_rate_training = error_rate(predicted_labels_training, iris_t_training_first)
    #error_rate_test = error_rate(predicted_labels_testing, iris_t_testing_last)
    #print('Error rate for training set is' ,error_rate_training)
    #print('Error rate for test set is', error_rate_test)

    #FINDING THE BEST ALPHA
    #alphas = [0.001, 0.01, 0.1, 0.5, 1.0]
    #best_alpha, best_error_rate = find_best_alpha(alphas)
    #print('Best alpha:', best_alpha)
    #print('Lowest validation error rate:', best_error_rate)


##############################
##                          ##
##     RUNNING THE CODE     ##
##                          ##
##############################
main()
