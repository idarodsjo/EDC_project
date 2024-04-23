from sklearn import metrics
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.linear_model import Perceptron, LinearRegression, Ridge, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

########################
##     FIRST PART     ##
########################

def read_data():
    """Read data and seperate into dataset without species, x, and with only species, t"""
    global iris, x, t
    iris = pd.read_csv('iris.csv')
    iris = iris.drop(columns=['Id'])
    #iris_describe = iris.describe()
    #iris_info = iris.info()
    #iris_species = iris.iloc[:,-1:].value_counts()
    #iris_sum = iris.isnull().sum()
    le = LabelEncoder()     #Label Encoding for converting the labels into numeric form
    iris['Species'] = le.fit_transform(iris['Species'])
    x = iris.drop('Species', axis=1)
    t = iris.Species

def iris_plots_hist():
    fig, axes = plt.subplots(nrows=2, ncols=2, sharey=True)

    species = ['Sepal Length [cm]', 'Sepal Width [cm]', 'Petal Length [cm]', 'Petal Width [cm]']
    colors = ['indigo', 'sienna', 'pink', 'darkgreen']
    xlim_values = [(0.9, 8), (0, 4.5), (0.9, 8), (0, 4.5)]

    for ax, column, xlim, color, title in zip(axes.flat, iris.columns, xlim_values, colors, species):
        iris[column].plot.hist(ax=ax, xlim=xlim, title=title, color=color, alpha=0.8)
        ax.set(xlabel='[cm]', ylabel='Number')
        ax.grid()

    plt.tight_layout()
    plt.show()

def iris_scatter():
    colors = ['plum', 'darkgreen', 'indigo']
    species = ['Iris-virginica', 'Iris-versicolor', 'Iris-setosa']

    fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)

    for i in range(3):
        x = iris[iris['Species'] == species[i]]
        axs[0, 0].scatter(x['SepalLengthCm'], x['SepalWidthCm'], c=colors[i], label=species[i])
        axs[0, 1].scatter(x['PetalLengthCm'], x['PetalWidthCm'], c=colors[i], label=species[i])
        axs[1, 0].scatter(x['SepalLengthCm'], x['PetalLengthCm'], c=colors[i], label=species[i])
        axs[1, 1].scatter(x['SepalWidthCm'], x['PetalWidthCm'], c=colors[i], label=species[i])

    axs[0, 0].set_title('Sepal Length vs Sepal Width')
    axs[0, 1].set_title('Petal Length vs Petal Width')
    axs[1, 0].set_title('Sepal Length vs Petal Length')
    axs[1, 1].set_title('Sepal Width vs Petal Width')

    for ax in axs.flat:
        ax.set(xlabel='Length (cm)', ylabel='Width (cm)')
        ax.legend()
        ax.grid()

    plt.tight_layout()
    plt.show()

def iris_training():
    """Training set with 30 first samples for both x and t.
       Training set with 30 last samples for both x and t."""
    #30 first samples for training
    global iris_x_training_first, iris_t_training_first, iris_x_training_last, iris_t_training_last
    iris_x_setosa_training_first, iris_t_setosa_training_first = x[:30], t[:30]
    iris_x_versicolor_training_first, iris_t_versicolor_training_first = x[50:80], t[50:80]
    iris_x_virginica_training_first, iris_t_virginica_training_first = x[100:130], t[100:130]
    iris_x_training_first = pd.concat([iris_x_setosa_training_first, iris_x_versicolor_training_first, iris_x_virginica_training_first])
    iris_t_training_first = pd.concat([iris_t_setosa_training_first, iris_t_versicolor_training_first, iris_t_virginica_training_first])

    #30 last samples for training
    iris_x_setosa_training_last, iris_t_setosa_training_last = x[20:50], t[20:50]
    iris_x_versicolor_training_last, iris_t_versicolor_training_last = x[70:100], t[70:100]
    iris_x_virginica_training_last, iris_t_virginica_training_last = x[-30:], t[-30:]
    iris_x_training_last = pd.concat([iris_x_setosa_training_last, iris_x_versicolor_training_last, iris_x_virginica_training_last])
    iris_t_training_last = pd.concat([iris_t_setosa_training_last, iris_t_versicolor_training_last, iris_t_virginica_training_last])

def iris_test():
    """Test set with 20 last  samples for both x and t.
       Test set with 20 first samples for both x and t."""
    #20 last samples for testing
    global iris_x_testing_last, iris_t_testing_last, iris_x_testing_first, iris_t_testing_first
    iris_x_setosa_testing_last, iris_t_setosa_testing_last = x[30:50], t[30:50]
    iris_x_versicolor_testing_last, iris_t_versicolor_testing_last = x[80:100], t[80:100]
    iris_x_virginica_testing_last, iris_t_virginica_testing_last = x[-20:], t[-20:]
    iris_x_testing_last = pd.concat([iris_x_setosa_testing_last, iris_x_versicolor_testing_last, iris_x_virginica_testing_last])
    iris_t_testing_last = pd.concat([iris_t_setosa_testing_last, iris_t_versicolor_testing_last, iris_t_virginica_testing_last])

    #20 first samples for testing
    iris_x_setosa_testing_first, iris_t_setosa_testing_first = x[30:50], t[30:50]
    iris_x_versicolor_testing_first, iris_t_versicolor_testing_first = x[80:100], t[80:100]
    iris_x_virginica_testing_first, iris_t_virginica_testing_first = x[-20:], t[-20:]
    iris_x_testing_first = pd.concat([iris_x_setosa_testing_first, iris_x_versicolor_testing_first, iris_x_virginica_testing_first])
    iris_t_testing_first = pd.concat([iris_t_setosa_testing_first, iris_t_versicolor_testing_first, iris_t_virginica_testing_first])

def sigmoid_this(y):
    """Eq:20"""
    return np.array(1 / (1 + np.exp(-y)))

#TODO: Train a linear classifier as described in subchapter 2.4 and 3.2. Tune the step factor alpha in equation 19 until the training converge.
def mse_1():
    print("Mean Squared Error for training- and test set one.")
    """Chapter 3.2 -  MSE based training of a linear classifier, three ways."""
    perceptron = Perceptron(max_iter=1000, random_state=42)         #Perceptron classifier with 1000 iterations and 42 as random seed
    perceptron.fit(iris_x_training_first, iris_t_training_first)    #Training the classifier
    t_pred = perceptron.predict(iris_x_testing_last)                #Predicting labels for the test set
    mse = mean_squared_error(iris_t_testing_last, t_pred)           #Calculating Mean Squared Error between test (true values) and predictet labels
    print("Mean Squared Error:", mse)

    ####RIDGE###
    param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}    #Define a grid of hyperparameters to search
    modelll = Ridge()                                               #Initialize Ridge regression model
    grid_search = GridSearchCV(modelll, param_grid, cv=5, scoring='neg_mean_squared_error') #Perform grid search
    grid_search.fit(iris_x_training_first, iris_t_training_first)
    best_alpha = grid_search.best_params_['alpha']                  #Get the best parameter
    print("Best alpha from Ridge:", best_alpha)

    ####LINEAR REGRESSION###
    model = LinearRegression()                                      #Training a linear regression model
    model.fit(iris_x_training_first, iris_t_training_first)
    predictions = model.predict(iris_x_testing_last)                #Make predictions on the test set
    mse = mean_squared_error(iris_t_testing_last, predictions)      #Calculate Mean Squared Error
    print("Mean Squared Error using Linear Regression:", mse)

    ####SVM####
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}        #Define a grid of hyperparameters to search
    modell = SVR(kernel='linear')                                   #Train an SVM model with SVR
    grid_search = GridSearchCV(modell, param_grid, cv=5, scoring='neg_mean_squared_error') #Perform grid search
    grid_search.fit(iris_x_training_first, iris_t_training_first)
    best_alpha = grid_search.best_params_['C']                      #Get the best parameter
    print("Best alpha from SVM:", best_alpha)

    modell.fit(iris_x_training_first, iris_t_training_first)
    predictions = modell.predict(iris_x_testing_last)               #Make predictions on the test set
    mse = mean_squared_error(iris_t_testing_last, predictions)      #Calculate Mean Squared Error
    print("Mean Squared Error using SVM:", mse)

    ####ACCURACY####
    model = LogisticRegression(max_iter=10000, random_state=42)     #Initialize and train the logistic regression model
    model.fit(iris_x_training_first, iris_t_training_first)         #Predict on the testing set
    predictions = model.predict(iris_x_testing_last)                #Evaluate accuracy
    accuracy = accuracy_score(iris_t_testing_last, predictions)
    print(f"Accuracy on testing set one is: {accuracy}")


def mse_2():
    print("Mean Squared Error for training- and test set two.")
    perceptron = Perceptron(max_iter=1000, random_state=42)         #Perceptron classifier with 1000 iterations and 42 as random seed
    perceptron.fit(iris_x_training_last, iris_t_training_last)      #Training the classifier
    t_pred = perceptron.predict(iris_x_testing_first)               #Predicting labels for the test set
    mse = mean_squared_error(iris_t_testing_first, t_pred)          #Calculating Mean Squared Error between test (true values) and predictet labels
    print("Mean Squared Error:", mse)

    ####RIDGE###
    param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}    #Define a grid of hyperparameters to search
    modelll = Ridge()                                               #Initialize Ridge regression model
    grid_search = GridSearchCV(modelll, param_grid, cv=5, scoring='neg_mean_squared_error') #Perform grid search
    grid_search.fit(iris_x_training_last, iris_t_training_last)
    best_alpha = grid_search.best_params_['alpha']                  #Get the best parameter
    print("Best alpha from Ridge:", best_alpha)

    ####LINEAR REGRESSION###
    model = LinearRegression()                                      #Training a linear regression model
    model.fit(iris_x_training_last, iris_t_training_last)
    predictions = model.predict(iris_x_testing_first)               #Make predictions on the test set
    mse = mean_squared_error(iris_t_testing_first, predictions)     #Calculate Mean Squared Error
    print("Mean Squared Error from Linear Regression is:", mse)

    ####SVM####
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}        #Define a grid of hyperparameters to search
    modell = SVR(kernel='linear')                                   #Train an SVM model with SVR
    grid_search = GridSearchCV(modell, param_grid, cv=5, scoring='neg_mean_squared_error') #Perform grid search
    grid_search.fit(iris_x_training_last, iris_t_training_last)
    best_alpha = grid_search.best_params_['C']                      #Get the best parameter
    print("Best alpha from SVM:", best_alpha)

    modell.fit(iris_x_training_last, iris_t_training_last)
    predictions = modell.predict(iris_x_testing_first)              #Make predictions on the test set
    mse = mean_squared_error(iris_t_testing_first, predictions)     #Calculate Mean Squared Error
    print("Mean Squared Error using SVM:", mse)

    ####ACCURACY####
    model = LogisticRegression(max_iter=10000, random_state=42)     #Initialize and train the logistic regression model
    model.fit(iris_x_training_last, iris_t_training_last)
    predictions = model.predict(iris_x_testing_first)               #Predict on the testing set
    accuracy = accuracy_score(iris_t_testing_first, predictions)    #Evaluate accuracy
    print(f"Accuracy on testing set two is: {accuracy}")




#TODO: Error Rate
def confusion_matrix():
    """Finding the confusion matrix and the error rate for the whole set."""
    corr = iris.corr()
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(corr, annot=True, ax=ax, cmap=sns.cubehelix_palette(as_cmap=True))
    plt.title("Confusion Matrix Iris Dataset")
    plt.show()
    
def confusion_matrix_training():
    """Find the confusion matrix and the error rate for the training set."""
    corr = iris_x_training_first.corr()
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(corr, annot=True, ax=ax, cmap=sns.cubehelix_palette(as_cmap=True))
    plt.title("Confusion Matrix Training Dataset")
    plt.show()

def confusion_matrix_test():
    """Find the confusion matrix and the error rate for the test set."""
    corr = iris_x_testing_last.corr()
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(corr, annot=True, ax=ax, cmap=sns.cubehelix_palette(as_cmap=True))
    plt.title("Confusion Matrix Testing Dataset")
    plt.show()

def plot_error_rates(error_rates_train, error_rates_test):
    iteration_numbers = range(len(error_rates_train))

    plt.plot(iteration_numbers, error_rates_train, label='Train')
    plt.plot(iteration_numbers, error_rates_test, label='Test')

    plt.xlabel("Iteration number")
    plt.ylabel("Error rate")
    plt.legend()

    plt.show()

#########################
##     SECOND PART     ##
#########################
# The second part has focus on features and linear separability. In this part the first 30 samples are used for training and the last 20 samples for test.
"""Produce histograms for each feature and class. Take away the feature which shows most
overlap between the classes. Train and test a classifier with the remaining three features.
(b) Repeat the experiment above with respectively two and one features.
(c) Compare the confusion matrixes and the error rates for the four experiments. Comment
on the property of the features with respect to linear separability both as a whole and
for the three separate classes."""


#########################
##       RUN CODE      ##
#########################

read_data()
iris_training()
iris_test()
#confusion_matrix()
#confusion_matrix_training()
#confusion_matrix_test()
mse_1()
mse_2()
