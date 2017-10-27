# #!/bin/bash
# python hw2_best_test.py ./train.csv ./test.csv ./X_train ./Y_train ./X_test ./prediction.csv
# python hw2_best_train.py ./train.csv ./test.csv ./X_train ./Y_train ./X_test ./prediction.csv

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

    # Read education_num from training set(in .csv) using pandas.
    x_data_education_num = pandas.read_csv(TRAIN_CSV_FILE_PATH).values[:, 4].astype(int)
    # Read x data to train from training set using pandas.
    x_data_train = pandas.read_csv(X_TRAIN_FILE_PATH).values[TRAIN_OFFSET_ROW:TRAIN_NUMBER_ROWS + TRAIN_OFFSET_ROW]
    # Read y data to train from training set using pandas.
    y_data_train = pandas.read_csv(Y_TRAIN_FILE_PATH).values[TRAIN_OFFSET_ROW:TRAIN_NUMBER_ROWS + TRAIN_OFFSET_ROW]

    # Read x data to valid from training set using pandas.
    x_data_valid = pandas.read_csv(X_TRAIN_FILE_PATH).values[VALID_OFFSET_ROW:VALID_NUMBER_ROWS + VALID_OFFSET_ROW]
    # Read y data to valid from training set using pandas.
    y_data_valid = pandas.read_csv(Y_TRAIN_FILE_PATH).values[VALID_OFFSET_ROW:VALID_NUMBER_ROWS + VALID_OFFSET_ROW]

    # Insert education_num to column 0 of x_data_train.
    x_data_train = np.concatenate((x_data_education_num.reshape(-1, 1), x_data_train), axis = 1)
    # Insert education_num to column 0 of x_data_valid.
    x_data_valid = np.concatenate((x_data_education_num.reshape(-1, 1), x_data_valid), axis = 1)

    # Term list of square, cubic and quadratic.
    TERMS_SQUARE = [0, 1, 2, 4, 5, 6]
    TERMS_CUBIC = [0, 1, 2, 4, 5, 6]
    TERMS_QUADRATIC = [0, 1, 2, 4, 5, 6]
    TERMS_LOG = [0, 1, 2, 4, 5, 6]
    x_data_train = np.concatenate((
        x_data_train,  
        # x_data_train[:, TERMS_SQUARE] ** 2,             # Add square terms of continuous column. 
        # x_data_train[:, TERMS_CUBIC] ** 3,              # Add cubic terms of continuous column. 
        # x_data_train[:, TERMS_QUADRATIC] ** 4,          # Add quadratic terms of continuous column.
        np.log(x_data_train[:, TERMS_LOG] + 1e-10),    # Add log terms of continuous column.
        np.log(x_data_train[:, 0] * x_data_train[:, 1] + 1e-10).reshape((-1, 1)),
        np.log(x_data_train[:, 0] * x_data_train[:, 2] + 1e-10).reshape((-1, 1)),
        np.log(x_data_train[:, 0] * x_data_train[:, 4] + 1e-10).reshape((-1, 1)),
        np.log(x_data_train[:, 0] * x_data_train[:, 5] + 1e-10).reshape((-1, 1)),
        np.log(x_data_train[:, 0] * x_data_train[:, 6] + 1e-10).reshape((-1, 1)),
        np.log(x_data_train[:, 1] * x_data_train[:, 2] + 1e-10).reshape((-1, 1)),
        np.log(x_data_train[:, 1] * x_data_train[:, 4] + 1e-10).reshape((-1, 1)),
        np.log(x_data_train[:, 1] * x_data_train[:, 5] + 1e-10).reshape((-1, 1)),
        np.log(x_data_train[:, 1] * x_data_train[:, 6] + 1e-10).reshape((-1, 1)),
        np.log(x_data_train[:, 2] * x_data_train[:, 4] + 1e-10).reshape((-1, 1)),
        np.log(x_data_train[:, 2] * x_data_train[:, 5] + 1e-10).reshape((-1, 1)),
        np.log(x_data_train[:, 2] * x_data_train[:, 6] + 1e-10).reshape((-1, 1)),
        np.log(x_data_train[:, 4] * x_data_train[:, 5] + 1e-10).reshape((-1, 1)),
        np.log(x_data_train[:, 4] * x_data_train[:, 6] + 1e-10).reshape((-1, 1)),
        np.log(x_data_train[:, 5] * x_data_train[:, 6] + 1e-10).reshape((-1, 1))),
        axis = 1)
    x_data_valid = np.concatenate((
        x_data_valid,  
        # x_data_valid[:, TERMS_SQUARE] ** 2,             # Add square terms of continuous column. 
        # x_data_valid[:, TERMS_CUBIC] ** 3,              # Add cubic terms of continuous column. 
        # x_data_valid[:, TERMS_QUADRATIC] ** 4,          # Add quadratic terms of continuous column.
        np.log(x_data_valid[:, TERMS_LOG] + 1e-10),    # Add log terms of continuous column.
        np.log(x_data_valid[:, 0] * x_data_valid[:, 1] + 1e-10).reshape((-1, 1)),
        np.log(x_data_valid[:, 0] * x_data_valid[:, 2] + 1e-10).reshape((-1, 1)),
        np.log(x_data_valid[:, 0] * x_data_valid[:, 4] + 1e-10).reshape((-1, 1)),
        np.log(x_data_valid[:, 0] * x_data_valid[:, 5] + 1e-10).reshape((-1, 1)),
        np.log(x_data_valid[:, 0] * x_data_valid[:, 6] + 1e-10).reshape((-1, 1)),
        np.log(x_data_valid[:, 1] * x_data_valid[:, 2] + 1e-10).reshape((-1, 1)),
        np.log(x_data_valid[:, 1] * x_data_valid[:, 4] + 1e-10).reshape((-1, 1)),
        np.log(x_data_valid[:, 1] * x_data_valid[:, 5] + 1e-10).reshape((-1, 1)),
        np.log(x_data_valid[:, 1] * x_data_valid[:, 6] + 1e-10).reshape((-1, 1)),
        np.log(x_data_valid[:, 2] * x_data_valid[:, 4] + 1e-10).reshape((-1, 1)),
        np.log(x_data_valid[:, 2] * x_data_valid[:, 5] + 1e-10).reshape((-1, 1)),
        np.log(x_data_valid[:, 2] * x_data_valid[:, 6] + 1e-10).reshape((-1, 1)),
        np.log(x_data_valid[:, 4] * x_data_valid[:, 5] + 1e-10).reshape((-1, 1)),
        np.log(x_data_valid[:, 4] * x_data_valid[:, 6] + 1e-10).reshape((-1, 1)),
        np.log(x_data_valid[:, 5] * x_data_valid[:, 6] + 1e-10).reshape((-1, 1))),
        axis = 1)

    # Standard deviation of X_data_train.
    x_data_train_std = x_data_train.std(0)
    # Mean of X_data_train.
    x_data_train_mean = x_data_train.mean(0)
    # Save std and mean to npy files.
    np.save('best_std.npy', x_data_train_std)
    np.save('best_mean.npy', x_data_train_mean)

    # List of normalization terms.
    NORM_LIST = [0, 1, 2, 4, 5, 6, 
        107, 108, 109, 110, 111, 112,
        113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]

    # Normalization on training set.
    for n in NORM_LIST:
    # for n in range(x_data_train.shape[1]):
        if (x_data_train_std[n] != 0):
            x_data_train[:, n] = (x_data_train[:, n] - x_data_train_mean[n]) / x_data_train_std[n]
    # Normalization on validation set.
    for n in NORM_LIST:
    # for n in range(x_data_valid.shape[1]):
        if (x_data_train_std[n] != 0):
            x_data_valid[:, n] = (x_data_valid[:, n] - x_data_train_mean[n]) / x_data_train_std[n] 
    
    # Create a column of bias.
    bias = np.ones((TRAIN_NUMBER_ROWS, 1))
    # Insert bias column in front of column 0.
    x_data_train = np.hstack((bias, x_data_train))
    # Create a column of bias.
    bias = np.ones((VALID_NUMBER_ROWS, 1))
    # Insert bias column in front of column 0.
    x_data_valid = np.hstack((bias, x_data_valid))

    # initial value of weights.
    w = np.zeros((x_data_train.shape[1], 1))
    # learning rate
    lr = 10
    print ('Learning rate:' , lr)
    lr_w = np.zeros((x_data_train.shape[1], 1))
    # Regularization parameter.
    lambda2 = 0.1
    print ("Regularization-lambda2: %f" %(lambda2))
    # Limitation of iteration.
    iteration = 20000
    print ("Limitation of iteration: %d" %(iteration))

    # Iterations
    for i in range(iteration):
        # Difference between real y and predicted y.
        difference = y_data_train - sigmoid(np.dot(x_data_train, w))
        # Clear first column of weight.
        wTemp = w
        wTemp[0, :] = 0
        # Gradient of weights.
        w_grad = -(x_data_train.T).dot(difference) + lambda2 * wTemp
        
        # Updata learning rate using adagrad.
        lr_w = lr_w + w_grad ** 2
        # Update parameters using adagrad.
        w = w - lr / np.sqrt(lr_w) * w_grad 

        # Print iteration each %.
        if (i % (iteration / 1000) == 0):
            # Predict y.
            logistic = sigmoid(np.dot(x_data_valid, w))
            yPredicted = (logistic > 0.5).astype(np.int)
            # Calculate accuracy.
            accuracy = (y_data_valid == yPredicted).mean()
            print ("\rIteration: %02.01f %%, Accuracy: %02.6f%%" %(i / iteration * 100, accuracy * 100), end = '')
    print ("\nIteration has done.")

    # Save model to npy files.
    np.save('best_model.npy', w)