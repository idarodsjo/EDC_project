from keras.datasets import mnist
import matplotlib.pyplot as plt

# Retreiving data set directly from keras library
(trainX, trainY), (testX, testY) = mnist.load_data()

# Shapes of arrays:
    # trainX: (60000, 28, 28)
    # trainY = (60000, 1)

    # testX: (10000, 28, 28)
    # testY : (10000, 1) 

def test_plot():
    plt.gray()

    plt.figure(figsize = (10, 9))

    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(trainX[i])

    plt.show()


# Value range of each pixel: [0, 255]
minPixelValue = trainX.min() 
maxPixelValue = trainX.max()


# Normalising and reshaping the data
def normaliseX(trainX, testX):
    trainX = trainX.astype('float32')
    testX = testX.astype('float32')

    normTrainX = trainX / 255.0
    normTestX = testX / 255.0

    return normTrainX, normTestX    # Returns pixel values between [0, 1]

def reshapeX(trainX, testX):
    reshapedTrainX = trainX.reshape(len(trainX), -1)
    reshapedTestX = testX.reshape(len(testX), -1)
    return reshapedTrainX, reshapedTestX    # Reshaped to (60000, 784) and (10000, 7854), respctively


trainX, testX = reshapeX(trainX, testX)
