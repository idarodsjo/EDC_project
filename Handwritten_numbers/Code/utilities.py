import numpy as np

def get_total_predictions():
    total_predictions = np.array([])
    for n in range(10):
        prediction_n = np.loadtxt(f'Task1_data/Predictions/preds{n}.txt', dtype=int)
        total_predictions = np.append(total_predictions, prediction_n)
        print(f'File number {n}, Prediction list length = {total_predictions.shape}')
    return total_predictions
