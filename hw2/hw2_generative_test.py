# #!/bin/bash
# python hw2_generative_test.py ./train.csv ./test.csv ./X_train ./Y_train ./X_test ./prediction.csv
# python hw2_generative_train.py ./train.csv ./test.csv ./X_train ./Y_train ./X_test ./prediction.csv

import sys
import numpy as np
import pandas

def sigmoid(z):
    """
    :type z: float
    :return type: float
    """  
    # # Prevent overflow.
    # z = np.clip(z, -500, 500)
    # Calculate activation signal
    return 1 / (1 + np.exp(-z))

if __name__ == "__main__":
    # Load file path of train.csv from arguments.
    TRAIN_CSV_FILE_PATH = sys.argv[1]
    # Load file path of test.csv from arguments.
    TEST_CSV_FILE_PATH = sys.argv[2]
    # Load file path of X_train from arguments.
    X_TRAIN_FILE_PATH = sys.argv[3]
    # Load file path of Y_train from arguments.
    Y_TRAIN_FILE_PATH = sys.argv[4]
    # Load file path of X_test from arguments.
    X_TEST_FILE_PATH = sys.argv[5]
    # Load file path of prediction.csv from arguments.
    PREDICTION_CSV_FILE_PATH = sys.argv[6]

    # Read x data to test from testing set using pandas.
    x_data_test = pandas.read_csv(X_TEST_FILE_PATH).values

    # Read std and mean from npy files.
    x_data_train_std = np.load('generative_std.npy')
    x_data_train_mean = np.load('generative_mean.npy')

    # Normalization on testing set.
    for n in range(x_data_test.shape[1]):
        if (x_data_train_std[n] != 0):
            x_data_test[:, n] = (x_data_test[:, n] - x_data_train_mean[n]) / x_data_train_std[n]

    # Read weights and bias from npy files.
    w = np.load('generative_weights.npy')
    b = np.load('generative_bias.npy')

    # Z (Gaussian Distribution).
    z = np.dot(w, x_data_test.T) + b
    # Put z into simoid function.
    y = sigmoid(z)
    # Predict y.
    yPredicted = (y >= 0.5).astype(np.int)

    stringID = list()
    for n in range(yPredicted.shape[0]):
        stringID.append(str(n + 1))

    resultCSV_dict = {
        "id":       stringID,
        "label":    yPredicted.reshape(-1)
    }
    resultCSV = pandas.DataFrame(resultCSV_dict)
    resultCSV.to_csv(PREDICTION_CSV_FILE_PATH, index=False)