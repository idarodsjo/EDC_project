from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def read_data():
    global iris, x, t
    iris = pd.read_csv('iris.csv')
    iris = iris.drop(columns=['Id'])
    #iris_describe = iris.describe()
    #iris_info = iris.info()
    #iris_species = iris.iloc[:,-1:].value_counts()
    #iris_sum = iris.isnull().sum()
    #print(iris)
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
    #30 first samples for training
    global iris_x_training, iris_t_training
    iris_x_setosa_training, iris_t_setosa_training = x[:30], t[:30]
    iris_x_versicolor_training, iris_t_versicolor_training = x[50:80], t[50:80]
    iris_x_virginica_training, iris_t_virginica_training = x[100:130], t[100:130]
    iris_x_training = pd.concat([iris_x_setosa_training, iris_x_versicolor_training, iris_x_virginica_training])
    iris_t_training = pd.concat([iris_t_setosa_training, iris_t_versicolor_training, iris_t_virginica_training])

def iris_test():
    #20 last samples for testing
    global iris_x_testing, iris_t_testing        
    iris_x_setosa_testing, iris_t_setosa_testing = x[30:50], t[30:50]
    iris_x_versicolor_testing, iris_t_versicolor_testing = x[80:100], t[80:100]
    iris_x_virginica_testing, iris_t_virginica_testing = x[-20:], t[-20:]
    iris_x_testing = pd.concat([iris_x_setosa_testing, iris_x_versicolor_testing, iris_x_virginica_testing])
    iris_t_testing = pd.concat([iris_t_setosa_testing, iris_t_versicolor_testing, iris_t_virginica_testing])

def confusion_matrix():
    corr = iris.corr()
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(corr, annot=True, ax=ax, cmap=sns.cubehelix_palette(as_cmap=True))
    plt.title("Confusion Matrix Iris Dataset")
    plt.show()
    
def confusion_matrix_training():
    corr = iris_x_training.corr()
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(corr, annot=True, ax=ax, cmap=sns.cubehelix_palette(as_cmap=True))
    plt.title("Confusion Matrix Training Dataset")
    plt.show()

def confusion_matrix_test():
    corr = iris_x_testing.corr()
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(corr, annot=True, ax=ax, cmap=sns.cubehelix_palette(as_cmap=True))
    plt.title("Confusion Matrix Testing Dataset")
    plt.show()

read_data()
iris_training()
iris_test()
confusion_matrix()
confusion_matrix_training()
confusion_matrix_test()
