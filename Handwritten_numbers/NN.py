# Link to read script: https://www.kaggle.com/code/hojjatk/read-mnist-dataset/notebook
# Link to data files: https://www.kaggle.com/datasets/hojjatk/mnist-dataset?select=t10k-images-idx3-ubyte
# KNN classifier tutorial: https://www.kaggle.com/code/prashant111/knn-classifier-tutorial

from keras.datasets import mnist
import numpy as np

(training_data, training_labels), (test_data, test_labels) = mnist.load_data()

print('Size: ' + str(np.size(test_data)))
print('Length: ' + str(len(test_data)))