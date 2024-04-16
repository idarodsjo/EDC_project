import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

def loadData():
    sampleLabelData = [] # [ (sample, label) ]
    samples = []
    labels = []

    with open(f'Iris_task/iris_dataset.txt', 'rb') as csv_file:
        for row in csv_file:
            cells = [cell.strip() for cell in row.decode().split(',')]
            label = cells[len(cells) - 1]
            sample = np.array(cells[:len(cells) - 1], dtype=np.float32)
            
            sampleLabelData.append((sample, label))
            samples.append(sample)
            labels.append(label)
        
        #sampleLabelDataArr = np.array(sampleLabelData((150,2)))
        #samples, labels = np.tranpose(sampleLabelDataArr)

        return samples, labels

#####################################################
#                   CHANGE LATER                    #

def split_dataset(samples, labels, split_index):
    classes = np.unique(labels)

    indices_by_class = []
    for curr_class in classes:
        indices = np.where(labels == curr_class)[0]
        indices_by_class.append(indices)
    indices_by_class = np.array(indices_by_class)

    first_set_indices_by_class = indices_by_class[ :, :split_index ]
    last_set_indices_by_class = indices_by_class[ :, split_index:]

    first_set_indices = first_set_indices_by_class.flatten()
    last_set_indices = last_set_indices_by_class.flatten()

    samples = np.array(samples)
    labels = np.array(labels)

    #print(first_set_indices)
    first_set = samples.take(first_set_indices), labels.take(first_set_indices)
    last_set = samples.take(last_set_indices), labels.take(last_set_indices)

    return first_set, last_set

def get_all_features():
    return {0: 'Sepal length',
            1: 'Sepal width',
            2: 'Petal length',
            3: 'Petal width'}

def remove_feature(samples, feature_index):
    samples_without_feature = []
    for sample in samples:
        new_sample = []
        for i in range(len(sample)):
            if i != feature_index:
                new_sample.append(sample[i])
        new_sample = np.array(new_sample)
        samples_without_feature.append(new_sample)

    return np.array(samples_without_feature)

#####################################################






#####################################################
#                   CLASSIFIER                      #
#####################################################
# MSE based traning of linear classifier (eq. 3.20 in compendium)
def get_predicted_label_vectors(x, W):
    denominator = np.array([np.exp(-(np.matmul(W, x_i))) for x_i in x])
    predictor = 1 / denominator
    return predictor    # returns vectors g

def get_rounded_label_vector(label_vector):
    max_index = np.argmax(label_vector)
    rounded_label_vector = np.array([i == max_index for i in range(len(label_vector))], dtype=np.uint8)
    return rounded_label_vector

# For the GMM-based Plug-in-MAP calssifier there is no explicit solution to find the MSE
# We therefore use gradient based technique (= update the W matrix in the opposite direction of the gradient)
# eq. 3.22

def get_weighted_gradient_MSE(predicted_labels, labels, samples):

    #numFeatures = len(samples[0]) - 1
    # TODO: Find out why line 98 doesn't run
    numFeatures = 4
    numClasses = 3
    
    gGradientMSE = predicted_labels - labels
    zGradientG = predicted_labels * (1 - predicted_labels)

    WGradZ = np.array([np.reshape(sample, (1, numFeatures + 1)) for sample in samples])

    WGradMSE = np.sum(np.matmul(np.reshape(gGradientMSE[k] * zGradientG[k], (numClasses, 1)), WGradZ[k]) for k in range(len(gGradientMSE)))

    return WGradMSE

# 
def get_next_weighted_matrix(predicted_labels, labels, samples, prevW, alpha = 0.01):
    WGradMSE = get_weighted_gradient_MSE(predicted_labels, labels, samples)
    nextW = prevW - alpha * WGradMSE

    return nextW

# Eq. 3.19
def get_MSE(predicted_label_vectors, true_label_vectors):
    error = predicted_label_vectors - true_label_vectors
    errorT = np.transpose(error)
    MSE = 0.5 * (np.sum(np.matmul(errorT, error)))
    return MSE

def get_error_rate(predicted_label_vectors, true_label_vectors):
    numSamples = len(true_label_vectors)
    classes = np.unique(true_label_vectors)

    numErrors = 0
    for i in range(len(true_label_vectors)):
        if not np.array_equal(true_label_vectors[i], predicted_label_vectors[i]):
            numErrors += 1
    
    return numErrors / numSamples

def train_linear_classifier(train_samples, train_label_vectors, test_samples, test_label_vectors, features, numIterations = 1000, alpha = 0.01):
    classes = np.unique(train_label_vectors)
    numClasses = 3
    numFeatures = len(features)

    MSE_per_iteration = []
    error_rate_per_iteration = []

    # Initialise weight matrix
    W = np.zeros((numClasses, numFeatures + 1))

    for currentIteration in range(numIterations):
        # Training
        predicted_train_label_vectors = get_predicted_label_vectors(train_samples, W)
        W = get_next_weighted_matrix(predicted_train_label_vectors, train_label_vectors, train_samples, W, alpha)

        # Testing
        predicted_test_label_vectors = get_predicted_label_vectors(test_samples, W)

        # WHAT IS THIS SYNTAX ???
        predicted_test_label_vectors_rounded = np.array([( \
            get_rounded_label_vector(label_vector) \
            for label_vector in predicted_test_label_vectors \
        )])

        currentMSE = get_MSE(predicted_test_label_vectors, test_label_vectors)
        MSE_per_iteration.append(currentMSE)

        print('Test label in train lin classifier: ')
        print(test_label_vectors)
        currentErrorRate = get_error_rate(predicted_test_label_vectors_rounded, test_label_vectors)
        error_rate_per_iteration.append(currentErrorRate)

        return W, np.array(MSE_per_iteration), np.array(error_rate_per_iteration)


# Confusion Matrix
def get_confusion_matrix(predicted_label_strings, true_label_strings):
    classes = np.unique(true_label_strings)

    confusionMatrix = []

    for predictedClass in classes:
        row = []

        for trueClass in classes:
            # All occurences of current true_class in true_label_strings:
            true_indices = np.where(true_label_strings == trueClass)[0]

            # All occurences of current predicted_class in predicted_label_strings:
            predicted_indices = np.where(predicted_label_strings == predictedClass)[0]

            # We want to find the number of elements where these two matches:
            numOccurences = len(np.intersect1d(true_indices, predicted_indices))
            row.append(numOccurences)
        
        confusionMatrix.append(row)

    return np.array(confusionMatrix)


# Plots

def plot_confusion_matrix(confusion_matrix, classes, name="Confusion matrix"):
    df_cm = pd.DataFrame(confusion_matrix, index = classes, columns = classes)
    fig = plt.figure(num=name, figsize=(5,5))

    sn.heatmap(df_cm, annot = True)
    plt.show()

def plot_error_rates(error_rates_train, error_rates_test):
    iteration_numbers = range(len(error_rates_train))

    plt.plot(iteration_numbers, error_rates_train, label='Train')
    plt.plot(iteration_numbers, error_rates_test, label='Test')

    plt.xlabel('Iteration number')
    plt.ylabel('Error rate')
    plt.legend()

    plt.show()

# Conversion funcs - CHANGE!!
def label_string_to_vector(string_label, classes):
    index = np.where(classes == string_label)[0]
    vector_label = np.array([ i == index for i in range(len(classes)) ], dtype=np.uint8)
    vector_label = np.reshape(vector_label, len(vector_label))

    return vector_label

def label_vector_to_string(vector_label, classes):
    index = np.argmax(vector_label)
    vector_string = classes[index]

    return vector_string


# Show plots - CHANGE THESE!!!  
def show_error_rate_plots(train_dataset, test_dataset, features, alpha=0.005, numIterations=1000):
    train_samples, train_labels = train_dataset
    test_samples, test_labels = test_dataset

    classes = np.unique(train_labels)

     # We need to add an awkward 1 to x_k as described on page 15:
    train_samples = np.array([ np.append(sample, [1]) for sample in train_samples ])
    test_samples = np.array([ np.append(sample, [1]) for sample in test_samples ])

    # Get vector representation of label strings
    train_label_vectors = np.array([ label_string_to_vector(label, classes) for label in train_labels])
    test_label_vectors = np.array([ label_string_to_vector(label, classes) for label in test_labels])


    # Calculate error rate for TEST:
    _, _, error_rates_test = train_linear_classifier( \
        train_samples, train_label_vectors, \
        test_samples, test_label_vectors, \
        features, alpha=alpha, numIterations=numIterations\
    )

    # Calculate error rates for TRAIN:
    _, _, error_rates_train = train_linear_classifier( \
        train_samples, train_label_vectors, \
        train_samples, train_label_vectors, \
        features, alpha=alpha, numIterations=numIterations\
    )


    plot_error_rates(error_rates_train, error_rates_test)

def show_confusion_matrices(train_dataset, test_dataset, features, num_iterations=1000, alpha=0.005):
    train_samples, train_labels = train_dataset
    test_samples, test_labels = test_dataset

    classes = np.unique(train_labels)

     # We need to add an awkward 1 to x_k as described on page 15:
    train_samples = np.array([ np.append(sample, [1]) for sample in train_samples ])
    test_samples = np.array([ np.append(sample, [1]) for sample in test_samples ])

    # Get vector representation of label strings
    train_label_vectors = np.array([ label_string_to_vector(label, classes) for label in train_labels])
    test_label_vectors = np.array([ label_string_to_vector(label, classes) for label in test_labels])

    # Here we use num_iterations=30 annd alpha=0.005 because these values proved to
    # give the best results, as discussed in the report
    W, _, _ = train_linear_classifier( \
        train_samples, train_label_vectors, \
        train_samples, train_label_vectors, \
        features, num_iterations, alpha \
    )

    predicted_test_label_vectors = get_predicted_label_vectors(test_samples, W)
    predicted_test_label_vectors = np.array([ get_rounded_label_vector(label_vector) for label_vector in predicted_test_label_vectors ])
    predicted_test_label_strings = np.array([ label_vector_to_string(label, classes) for label in predicted_test_label_vectors ])

    predicted_train_label_vectors = get_predicted_label_vectors(train_samples, W)
    predicted_train_label_vectors = np.array([ get_rounded_label_vector(label_vector) for label_vector in predicted_train_label_vectors ])
    predicted_train_label_strings = np.array([ label_vector_to_string(label, classes) for label in predicted_train_label_vectors ])

    confusion_matrix_test = get_confusion_matrix(predicted_test_label_strings, test_labels)
    confusion_matrix_train = get_confusion_matrix(predicted_train_label_strings, train_labels)

    plot_confusion_matrix(confusion_matrix_test, classes, name="CM for test set")
    plot_confusion_matrix(confusion_matrix_train, classes, name="CM for train set")

    print('test label vectors, in show confusion matrix: ')
    print(test_label_vectors)
    error_rate_test = get_error_rate(predicted_test_label_vectors, test_label_vectors)
    error_rate_train = get_error_rate(predicted_train_label_vectors, train_label_vectors)

    print(f'Error rate for test set: {error_rate_test}')
    print(f'Error rate for train set: {error_rate_train}')

def plot_MSEs(MSEs_per_alpha, alphas):
    for i in range(len(alphas)):
        MSEs = MSEs_per_alpha[i]
        alpha = alphas[i]

        iteration_numbers = range(len(MSEs))
        plt.plot(iteration_numbers, MSEs, label='$\\alpha={' + str(alpha) + '}$')

    plt.xlabel("Iteration number")
    plt.ylabel("MSE")
    plt.legend()

    plt.show()

def show_MSE_plots(train_dataset, test_dataset, features, alphas, numIterations=1000):
    train_samples, train_labels = train_dataset
    test_samples, test_labels = test_dataset

    classes = np.unique(train_labels)

     # We need to add an awkward 1 to x_k as described on page 15:
    train_samples = np.array([ np.append(sample, [1]) for sample in train_samples ])
    test_samples = np.array([ np.append(sample, [1]) for sample in test_samples ])

    # Get vector representation of label strings
    train_label_vectors = np.array([ label_string_to_vector(label, classes) for label in train_labels])
    test_label_vectors = np.array([ label_string_to_vector(label, classes) for label in test_labels])

    MSEs_per_alpha = []
    for alpha in alphas:
        _, MSE_per_iteration, _ = train_linear_classifier( \
            train_samples, train_label_vectors, \
            test_samples, test_label_vectors, \
            features, alpha=alpha, numIterations=numIterations\
        )
        MSEs_per_alpha.append(MSE_per_iteration)

    plot_MSEs(MSEs_per_alpha, alphas)

all_samples, all_labels = loadData()
train_dataset, test_dataset = split_dataset(all_samples, all_labels, split_index=30)
features = get_all_features()

show_MSE_plots(train_dataset, test_dataset, features, alphas=[0.0025, 0.005, 0.0075, 0.01])
#show_error_rate_plots(train_dataset, test_dataset, features, alpha=0.005)
#show_confusion_matrices(train_dataset, test_dataset, features, num_iterations=300, alpha=0.005)

""" def gaussian_mixture_model(M, C, weight_matrix, feature_dimension, stdev_matrix, x, expval_matrix):
    for k in range(M):
        for i in range(C):
            (weight_matrix[i][k] / (np.sqrt((2 * np.pi)**feature_dimension * np.abs(stdev_matrix[i][k])))) * np.exp(- 0.5 * np.transpose(x - expval_matrix[i][k]) (stdev_matrix[i][k])**(-1)(x - expval_matrix[i][k]))

def linear_discriminant_classifier_binary(numClasses, x, classList, classOffset):
    g = []
    for i in numClasses:
        g_i = np.transpose(classList[i]) * x + classOffset
        g.append(g_i)

def linear_dicriminant_classifier_nonbinary(numClasses, numFeatures, W, x, w_0):
    g = np.empty(numClasses)
    for i in numClasses:
        for k in numFeatures:
            g.append(W[i][k] * x + w_0[numClasses])

print(numClasses) """