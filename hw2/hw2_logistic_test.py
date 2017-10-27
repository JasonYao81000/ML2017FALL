# #!/bin/bash
# python hw2_logistic_test.py ./train.csv ./test.csv ./X_train ./Y_train ./X_test ./prediction.csv
# python hw2_logistic_train.py ./train.csv ./test.csv ./X_train ./Y_train ./X_test ./prediction.csv

import sys
import numpy as np
import pandas

def sigmoid(z):
    """
    :type z: float
    :return type: float
    """  
    # Prevent overflow.
    z = np.clip(z, -500, 500)
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

    # Read education_num from testing set(in .csv) using pandas.
    x_data_education_num = pandas.read_csv(TEST_CSV_FILE_PATH).values[:, 4].astype(int)
    # Read x data to test from testing set using pandas.
    x_data_test = pandas.read_csv(X_TEST_FILE_PATH).values

    # Insert education_num to column 0 of x_data_test.
    x_data_test = np.concatenate((x_data_education_num.reshape(-1, 1), x_data_test), axis = 1)
    
    # Term list of square, cubic and quadratic.
    TERMS_SQUARE = [0, 1, 2, 4, 5, 6]
    TERMS_CUBIC = [0, 1, 2, 4, 5, 6]
    TERMS_QUADRATIC = [0, 1, 2, 4, 5, 6]
    TERMS_LOG = [0, 1, 2, 4, 5, 6]
    x_data_test = np.concatenate((
        x_data_test,  
        x_data_test[:, TERMS_SQUARE] ** 2,              # Add square terms of continuous column. 
        x_data_test[:, TERMS_CUBIC] ** 3,               # Add cubic terms of continuous column. 
        x_data_test[:, TERMS_QUADRATIC] ** 4,           # Add quadratic terms of continuous column.
        np.log(x_data_test[:, TERMS_LOG] + 1e-10)),     # Add log terms of continuous column.
        axis = 1)
    
    # Read std and mean from npy files.
    x_data_train_std = np.load('logistic_std.npy')
    x_data_train_mean = np.load('logistic_mean.npy')

    # List of normalization terms.
    NORM_LIST = [0, 1, 2, 4, 5, 6, 
        107, 108, 109, 110, 111, 112, 
        113, 114, 115, 116, 117, 118, 
        119, 120, 121, 122, 123, 124, 
        125, 126, 127, 128, 129, 130]

    # Normalization on testing set.
    for n in NORM_LIST:
    # for n in range(x_data_test.shape[1]):
        if (x_data_train_std[n] != 0):
            x_data_test[:, n] = (x_data_test[:, n] - x_data_train_mean[n]) / x_data_train_std[n]
    # Create a column of bias.
    bias = np.ones((x_data_test.shape[0], 1))
    # Insert bias column in front of column 0.
    x_data_test = np.hstack((bias, x_data_test))

    # Read model from npy files.
    w = np.load('logistic_model.npy')

    # Predict y.
    logistic = sigmoid(np.dot(x_data_test, w))
    yPredicted = (logistic > 0.5).astype(np.int)

    stringID = list()
    for n in range(yPredicted.shape[0]):
        stringID.append(str(n + 1))

    resultCSV_dict = {
        "id":       stringID,
        "label":    yPredicted.reshape(-1)
    }
    resultCSV = pandas.DataFrame(resultCSV_dict)
    resultCSV.to_csv(PREDICTION_CSV_FILE_PATH, index=False)