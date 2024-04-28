import matplotlib.pyplot as plt
import pandas as pd

###################################
##                               ##
##     PLOTTING IRIS DATASET     ##
##                               ##
###################################


def read_data():
    global iris
    iris = pd.read_csv('iris.csv')
    iris = iris.drop(columns=['Id'])

def iris_hist_plot():
    plt.figure(figsize=(10, 6))

    #SEPAL LENGTH
    setosa_sepal_width = iris[iris['Species'] == 'Iris-setosa']['SepalLengthCm']
    versicolor_sepal_width = iris[iris['Species'] == 'Iris-versicolor']['SepalLengthCm']
    virginica_sepal_width = iris[iris['Species'] == 'Iris-virginica']['SepalLengthCm']

    plt.subplot(2, 2, 1)
    plt.hist(setosa_sepal_width, color='plum', alpha=0.5, label='Setosa')
    plt.hist(versicolor_sepal_width, color='darkgreen', alpha=0.5, label='Versicolor')
    plt.hist(virginica_sepal_width, color='indigo', alpha=0.5, label='Virginica')
    plt.title('Sepal Width')
    plt.xlabel('Sepal Width [cm]')
    plt.ylabel('Number')
    plt.legend()

    #SEPAL WIDTH
    setosa_sepal_width = iris[iris['Species'] == 'Iris-setosa']['SepalWidthCm']
    versicolor_sepal_width = iris[iris['Species'] == 'Iris-versicolor']['SepalWidthCm']
    virginica_sepal_width = iris[iris['Species'] == 'Iris-virginica']['SepalWidthCm']

    plt.subplot(2, 2, 2)
    plt.hist(setosa_sepal_width, color='plum', alpha=0.5, label='Setosa')
    plt.hist(versicolor_sepal_width, color='darkgreen', alpha=0.5, label='Versicolor')
    plt.hist(virginica_sepal_width, color='indigo', alpha=0.5, label='Virginica')
    plt.title('Sepal Length')
    plt.xlabel('Sepal Length [cm]')
    plt.ylabel('Number')
    plt.legend()

    #PETAL LENGTH
    setosa_sepal_width = iris[iris['Species'] == 'Iris-setosa']['PetalLengthCm']
    versicolor_sepal_width = iris[iris['Species'] == 'Iris-versicolor']['PetalLengthCm']
    virginica_sepal_width = iris[iris['Species'] == 'Iris-virginica']['PetalLengthCm']

    plt.subplot(2, 2, 3)
    plt.hist(setosa_sepal_width, color='plum', alpha=0.5, label='Setosa')
    plt.hist(versicolor_sepal_width, color='darkgreen', alpha=0.5, label='Versicolor')
    plt.hist(virginica_sepal_width, color='indigo', alpha=0.5, label='Virginica')
    plt.title('Petal Length')
    plt.xlabel('Petal Length [cm]')
    plt.ylabel('Number')
    plt.legend()

    #PETAL WIDTH
    setosa_sepal_width = iris[iris['Species'] == 'Iris-setosa']['PetalWidthCm']
    versicolor_sepal_width = iris[iris['Species'] == 'Iris-versicolor']['PetalWidthCm']
    virginica_sepal_width = iris[iris['Species'] == 'Iris-virginica']['PetalWidthCm']

    plt.subplot(2, 2, 4)
    plt.hist(setosa_sepal_width, color='plum', alpha=0.5, label='Setosa')
    plt.hist(versicolor_sepal_width, color='darkgreen', alpha=0.5, label='Versicolor')
    plt.hist(virginica_sepal_width, color='indigo', alpha=0.5, label='Virginica')
    plt.title('Petal Width')
    plt.xlabel('Petal width [cm]')
    plt.ylabel('Number')
    plt.legend()

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

read_data()
iris_scatter()
iris_hist_plot()
