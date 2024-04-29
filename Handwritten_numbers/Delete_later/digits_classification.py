import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.cluster import KMeans
import seaborn as sns
import time
import multiprocessing

# Parameters
N_train = 60000
N_test = 10000
C = 10

class Plotter:
    @staticmethod
    def confusion_matrix(title, confusion_matrix, error_rate, visualize):
        if visualize:
            plt.figure(figsize=(10, 7))
            plt.title('Confusion matrix for ' + title + '\n' + 'Error rate: ' + str(error_rate * 100) + '%')
            sns.heatmap(confusion_matrix, annot=True, fmt='.0f')
            plt.xlabel('Predicted label')
            plt.ylabel('True label')
            plt.show()

    @staticmethod
    def comparison(test_data, mean_data, classified_labels, labels_indexes, N_plots, visualize):
        if visualize:
            plt.figure()
            for i in range(N_plots):
                lab_index = labels_indexes[i]

                test_image = test_data[lab_index]
                predicted_image = mean_data[classified_labels[lab_index]].reshape(28, 28)
                difference_image = test_image - predicted_image

                plt.subplot(N_plots, 3, 3 * i + 1)
                plt.imshow(test_image, cmap=plt.get_cmap('gray'))
                if i == 0:
                    plt.title('test image')

                plt.subplot(N_plots, 3, 3 * i + 2)
                plt.imshow(predicted_image, cmap=plt.get_cmap('gray'))
                if i == 0:
                    plt.title('Predicted image')

                plt.subplot(N_plots, 3, 3 * i + 3)
                plt.imshow(difference_image, cmap=plt.get_cmap('gray'))
                if i == 0:
                    plt.title('Difference image')

    @staticmethod
    def cluster_centers(centers):
        plt.figure(figsize=(10, 100))
        for i in range(100):
            plt.subplot(10, 10, i + 1)
            plt.imshow(centers[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
            plt.axis('off')
        plt.suptitle("Some cluster centers for each digit", fontsize=20)
        plt.show()

def split_data(training_data, training_labels):
    sorted_labels = np.argsort(training_labels)
    sorted_data = np.empty_like(training_data)

    for i in range(len(training_labels)):
        sorted_data[i] = training_data[sorted_labels[i]]

    return sorted_data

def mean_digit_value_image(train_data, train_label, C, N_pixels):
    mean_data = np.zeros((C, N_pixels))
    for i in range(C):
        mean_data[i] = np.mean(train_data[train_label == i], axis=0).reshape(N_pixels)
        print(f'Calculated mean data for class {i}: {mean_data[i]}')
    return mean_data

def cluster_data(num_clusters, training_data, training_labels, num_classes):
    start = time.time()

    class_list = split_data(training_data, training_labels)
    class_list_flattened = class_list.flatten().reshape(60000, 784)

    cluster_matrix = np.empty((num_classes, num_clusters, 784))

    classes = np.unique(training_labels)
    for class_i in enumerate(classes):
        cluster = KMeans(n_clusters=num_clusters, random_state=0).fit(class_list_flattened).cluster_centers_
        cluster_matrix[class_i] = cluster
        print(class_i)
    
    end = time.time()
    cluster_matrix_flattened = cluster_matrix.flatten().reshape(num_classes * num_clusters, 784)

    return cluster_matrix_flattened

def calculate_distances(test_data, train_data, distance_list):
    print('Calculating euclidian distances ...')
    distances = []
    for test_image in test_data:
        print('before distances per image')
        distances_per_image = [np.linalg.norm(test_image - train_image) for train_image in train_data]
        distances.append(distances_per_image)
        print('after distances per image, distances has size ' + str(len(distances)))
    print('Euclidian distances calculated!')
    distance_list.put(distances)
    #return distances

def classify_with_nearest_neighbor(distances, train_label, test_label):
    classified_labels = []
    correct_labels_indexes = []
    failed_labels_indexes = []

    print('Classifying with NN template ...')

    for i, test_distances in enumerate(distances):
        closest_train_index = np.argmin(test_distances)
        label = train_label[closest_train_index]
        classified_labels.append(label)
        if label == test_label[i]:
            correct_labels_indexes.append(i)
        else:
            failed_labels_indexes.append(i)
    print('Classification complete!')

    return classified_labels, correct_labels_indexes, failed_labels_indexes

def calculate_confusion_matrix(classified_labels, test_label, C):
    confusion_matrix = np.zeros((C, C))
    for i in range(len(classified_labels)):
        confusion_matrix[test_label[i], classified_labels[i]] += 1
    return confusion_matrix

def calculate_error_rate(confusion_matrix):
    error = np.trace(confusion_matrix)
    return round(1 - (error / np.sum(confusion_matrix)), 5)

def print_time(start_time, end_time):
    time = end_time - start_time
    hours = int(time // 3600)
    minutes = int((time % 3600) // 60)
    seconds = int(time % 60)
    print("Time: {:02d}:{:02d}:{:02d}".format(hours, minutes, seconds))

def save_to_file(file_name, confusion_matrix, error_rate, N_train, N_test, time):
    file_title = "Plots_and_results/" + file_name + "N_train_" + str(N_train) + "_N_test_" + str(N_test) + ".txt"
    with open(file_title, 'w') as f:
        f.write("Confusion matrix:\n")
        f.write(str(confusion_matrix))
        f.write("\nError rate: " + str(error_rate * 100) + "%\n")
        f.write("Time: " + str(time) + "\n")
    print("Saved to file: " + file_title)


def main():
    # Load MNIST data
    (train_data, train_label), (test_data, test_label) = mnist.load_data()
    train_data = train_data / 255
    test_data = test_data / 255

    # Parameters
    N_train = 60000
    N_test = 10000
    visualize_confusion_matrix = True
    data_chunk_train = 15000
    data_chunk_test = 2500
    num_classes = 10
    num_clusters = 64


    # Compute mean images for each class
    mean_data = mean_digit_value_image(train_data[:N_train], train_label[:N_train], num_classes, 28 * 28)

    # Classify test data with nearest neighbor
    time_start = time.time()
    # TODO: Create processes where the chunks are classified and trained with multiprocessing
    #distances = calculate_distances(test_data[:N_test], train_data[:N_train])
    total_distances = []
    p1_dist = multiprocessing.Queue()
    p1 = multiprocessing.Process(target=calculate_distances, args=(test_data[:data_chunk_test], train_data[:data_chunk_train], p1_dist))
    p2_dist = multiprocessing.Queue()
    p2 = multiprocessing.Process(target=calculate_distances, args=(test_data[data_chunk_test:2*data_chunk_test], train_data[data_chunk_train:2*data_chunk_train], p2_dist))
    p3_dist = multiprocessing.Queue()
    p3 = multiprocessing.Process(target=calculate_distances, args=(test_data[2*data_chunk_test:3*data_chunk_test], train_data[2*data_chunk_train:3*data_chunk_train], p3_dist))
    p4_dist = multiprocessing.Queue()
    p4 = multiprocessing.Process(target=calculate_distances, args=(test_data[3*data_chunk_test:4*data_chunk_test], train_data[3*data_chunk_train:4*data_chunk_train], p4_dist))

    p1.start()
    p2.start()
    p3.start()
    p4.start()

    total_distances.append(p1_dist)
    total_distances.append(p2_dist)
    total_distances.append(p3_dist)
    total_distances.append(p4_dist)

    p1.join()
    p2.join()
    p3.join()
    p4.join()

    classified_labels, _, _ = classify_with_nearest_neighbor(total_distances, train_label[:N_train], test_label[:N_test])
    time_end = time.time()
    training_time = time_end - time_start

    # Calculate confusion matrix and error rate
    confusion_matrix = calculate_confusion_matrix(classified_labels, test_label[:N_test], num_classes)
    error_rate = calculate_error_rate(confusion_matrix)

    Plotter.cluster_centers(cluster_data(num_clusters, train_data, train_label, num_classes))

    # Print time, save to file, and plot confusion matrix
    print_time(time_start, time_end)
    save_to_file("NN/CM_NN_", confusion_matrix, error_rate, N_train, N_test, training_time)
    Plotter.confusion_matrix("NN", confusion_matrix, error_rate, visualize_confusion_matrix)

    # Visualize comparison of test images with predicted images
    labels_indexes = [0, 1, 2]  # Example list of test image indices for comparison
    N_Comparisons = len(labels_indexes)
    Plotter.comparison(test_data[:N_test], mean_data, classified_labels, labels_indexes, N_Comparisons, visualize=True)

if __name__ == "__main__":
    main()