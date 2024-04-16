from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np

iris = load_iris()
svn = SVR()

"""The dataset is split into features (X) and the target variable (y). 
Features represent the measurements of the iris flowers, and the target 
variable is the species of each flower. The “Species” column is dropped 
to obtain the features."""
x, t = iris.data, iris.target

"""Splitting dataset into training set and testing set - used in task 1.1 and 2.1"""
#30 first samples for training
iris_x_setosa_training, iris_t_setosa_training = x[:30], t[:30]
iris_x_versicolor_training, iris_t_versicolor_training = x[50:80], t[50:80]
iris_x_virginica_training, iris_t_virginica_training = x[100:130], t[100:130]
iris_x_training = np.concatenate([iris_x_setosa_training, iris_x_versicolor_training, iris_x_virginica_training], axis=0)
iris_t_training = np.concatenate([iris_t_setosa_training, iris_t_versicolor_training, iris_t_virginica_training], axis=0)

#20 last samples for testing         
iris_x_setosa_testing, iris_t_setosa_testing = x[30:50], t[30:50]
iris_x_versicolor_testing, iris_t_versicolor_testing = x[80:100], t[80:100]
iris_x_virginica_testing, iris_t_virginica_testing = x[-20:], t[-20:]
iris_x_testing = np.concatenate([iris_x_setosa_testing, iris_x_versicolor_testing, iris_x_virginica_testing], axis=0)
iris_t_testing = np.concatenate([iris_t_setosa_testing, iris_t_versicolor_testing, iris_t_virginica_testing], axis=0)

"""Train a linear classifier as described in subchapter 2.4 and 3.2. Tune the step 
factor in equation 19 until the training converge."""

"""Chapter 3.2 -  MSE based training of a linear classifier"""
perceptron = Perceptron(max_iter=1000, random_state=42)     #Perceptron classifier with 1000 iterations and 42 as random seed
perceptron.fit(iris_x_training, iris_t_training)            #Training the classifier
t_pred = perceptron.predict(iris_x_testing)                 #Predicting labels for the test set
mse = mean_squared_error(iris_t_testing, t_pred)            #Calculating Mean Squared Error between test (true values) and predictet labels
print("Mean Squared Error:", mse)

svn = svn.fit(iris_x_training, iris_t_training)             #Feeding the training dataset into the algorithm by using the svn.fit()
predictions = svn.predict(iris_x_testing)                   #Predicting from the test dataset
#accuracy_score(iris_t_testing, predictions)                 #Calculate the accuracy

#Plott
_, ax = plt.subplots()
scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
_ = ax.legend(
    scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
)

#plt.show()
