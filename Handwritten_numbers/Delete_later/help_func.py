import numpy as np

def calculate_confusion_matrix(classified_labels, test_label, numClasses):
    confusion_matrix = np.zeros((numClasses, numClasses))
    for i in range(len(classified_labels)):
        confusion_matrix[test_label[i], classified_labels[i]] += 1
    return confusion_matrix

def calculate_error_rate(confusion_matrix):
    error = np.trace(confusion_matrix)
    return round(1 - (error / np.sum(confusion_matrix)), 5)

def split_data(training_data, training_labels):
    sorted_labels = np.argsort(training_labels)
    sorted_data = np.empty_like(training_data)

    for i in range(len(training_labels)):
        sorted_data[i] = training_data[sorted_labels[i]]

    return sorted_data