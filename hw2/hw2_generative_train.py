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

    # Number of attributes to train.
    NUMBER_ATTRIBUTES = 106
    print ("How many attributes to train: %d" %(NUMBER_ATTRIBUTES))

    # Attributes to train.
    ATTRIBUTES_TRAIN_LIST = np.array(range(NUMBER_ATTRIBUTES))
    # print ("ATTRIBUTES_TRAIN_LIST: ", ATTRIBUTES_TRAIN_LIST)
    
    # How many rows(people) to fetch from training set.
    TRAIN_NUMBER_ROWS = 32561 # - 8140
    # Row(people) offset to fetch from training set.
    TRAIN_OFFSET_ROW = 0
    print ("TRAINING SET: %d rows(people) and %d offset." %(TRAIN_NUMBER_ROWS, TRAIN_OFFSET_ROW))

    # How many rows(people) to fetch from validation set.
    VALID_NUMBER_ROWS = 32561
    # Row(people) offset to fetch from validation set.
    VALID_OFFSET_ROW = 0
    print ("VALIDATION SET: %d rows(people) and %d offset." %(VALID_NUMBER_ROWS, VALID_OFFSET_ROW))

    # Read x data to train from training set using pandas.
    x_data_train = pandas.read_csv(X_TRAIN_FILE_PATH).values[TRAIN_OFFSET_ROW:TRAIN_NUMBER_ROWS + TRAIN_OFFSET_ROW]
    # Read y data to train from training set using pandas.
    y_data_train = pandas.read_csv(Y_TRAIN_FILE_PATH).values[TRAIN_OFFSET_ROW:TRAIN_NUMBER_ROWS + TRAIN_OFFSET_ROW]

    # Read x data to valid from training set using pandas.
    x_data_valid = pandas.read_csv(X_TRAIN_FILE_PATH).values[VALID_OFFSET_ROW:VALID_NUMBER_ROWS + VALID_OFFSET_ROW]
    # Read y data to valid from training set using pandas.
    y_data_valid = pandas.read_csv(Y_TRAIN_FILE_PATH).values[VALID_OFFSET_ROW:VALID_NUMBER_ROWS + VALID_OFFSET_ROW]

    # Standard deviation of X_data_train.
    x_data_train_std = x_data_train.std(0)
    # Mean of X_data_train.
    x_data_train_mean = x_data_train.mean(0)
    # Save std and mean to npy files.
    np.save('generative_std.npy', x_data_train_std)
    np.save('generative_mean.npy', x_data_train_mean)

    # Normalization on training set.
    for n in range(x_data_train.shape[1]):
        if (x_data_train_std[n] != 0):
            x_data_train[:, n] = (x_data_train[:, n] - x_data_train_mean[n]) / x_data_train_std[n]
    # Normalization on validation set.
    for n in range(x_data_valid.shape[1]):
        if (x_data_train_std[n] != 0):
            x_data_valid[:, n] = (x_data_valid[:, n] - x_data_train_mean[n]) / x_data_train_std[n] 

    # mu of class 1.
    mu1 = np.zeros((NUMBER_ATTRIBUTES, ))
    # mu of class 2.
    mu2 = np.zeros((NUMBER_ATTRIBUTES, ))
    # count of class1.
    N1 = 0
    # count of class2.
    N2 = 0
    # Accumulate mu1 and mu2.
    for i in range(x_data_train.shape[0]):
        # It is class 1.
        if (y_data_train[i] == 1):
            mu1 += x_data_train[i]
            N1 += 1
        # It is class 2.
        else:
            mu2 += x_data_train[i]
            N2 += 1
    # Average of mu1 and mu2.
    mu1 /= N1
    mu2 /= N2

    # sigma of class 1.
    sigma1 = np.zeros((NUMBER_ATTRIBUTES, NUMBER_ATTRIBUTES))
    # sigma of class 2.
    sigma2 = np.zeros((NUMBER_ATTRIBUTES, NUMBER_ATTRIBUTES))
    # Accumulate sigma1 and sigma2.
    for i in range(x_data_train.shape[0]):
        # It is class 1.
        if (y_data_train[i] == 1):
            sigma1 += np.dot(np.transpose([x_data_train[i] - mu1]), [x_data_train[i] - mu1])
        # It is class 2.
        else:
            sigma2 += np.dot(np.transpose([x_data_train[i] - mu2]), [x_data_train[i] - mu2])
    # Average of sigma1 and sigma2.
    sigma1 /= N1
    sigma2 /= N2

    # Shared sigma.
    shared_sigma = (float(N1) / x_data_train.shape[0]) * sigma1 + (float(N2) / x_data_train.shape[0]) * sigma2
    # Inverse of shared sigma.
    shared_sigma_inv = np.linalg.inv(shared_sigma)
    # Weight.
    w = np.dot((mu1 - mu2), shared_sigma_inv)
    # Bias.
    b = -0.5 * np.dot(np.dot(mu1, shared_sigma_inv), mu1) + 0.5 * np.dot(np.dot(mu2, shared_sigma_inv), mu2) + np.log(float(N1) / N2)
    # Z (Gaussian Distribution).
    z = np.dot(w, x_data_valid.T) + b
    # Put z into simoid function.
    y = sigmoid(z).reshape(y_data_valid.shape)
    # Predict y.
    yPredicted = (y > 0.5).astype(np.int)
    
    # Calculate accuracy.
    accuracy = (y_data_valid == yPredicted).mean()
    print ("Accuracy: %02.6f%%" %(accuracy * 100))

    # Save weights and bias to npy files.
    np.save('generative_weights.npy', w)
    np.save('generative_bias.npy', b)
