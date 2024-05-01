import numpy as np
import classifier
import processing as p

def get_all_predictions():
    allPredictions = np.array([])
    for n in range(10):
        prediction_n = np.loadtxt(f'Task1_data/Predictions/preds{n}.txt', dtype=int)
        allPredictions = np.append(allPredictions, prediction_n)
        print(f'File number {n}, Prediction list length = {allPredictions.shape}')
    return allPredictions

#   This was used to split up data into chunks, saves data to .txt file to avoid long computation time when plotting
def chunk_data(startChunk, numChunksTesting):
    totalErrorRate = np.array([])
    totalPreds = np.array([])
    testXChunks = np.array_split(p.testX, 10)
    testXChunks = np.array(testXChunks)

    for chunks in range(numChunksTesting):
        classifier.fit(p.trainX, p.trainY)
        print(f'Predicting chunk {chunks+startChunk}')

        preds = classifier.predict(testXChunks[chunks+startChunk])
        np.savetxt(f'preds{chunks+startChunk}.txt', preds)
        totalPreds = np.append(totalPreds, preds)

        print(f'Calculating errorRate for chunk {chunks + startChunk}')
        errorRate = classifier.errorRate(testXChunks[chunks + startChunk], p.testY[(chunks + startChunk)*1000:(chunks+1 + startChunk)*1000])
        totalErrorRate = np.append(totalErrorRate, errorRate)
        np.savetxt(f'errorRate{startChunk}{startChunk+numChunksTesting-1}.txt', totalErrorRate)