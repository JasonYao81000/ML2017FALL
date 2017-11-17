# #!/bin/bash
# python3 test.py ./test.csv ./prediction.csv
# python3 train.py ./train.csv ./prediction.csv
import sys
import numpy as np
import pandas
from keras.models import load_model
from keras import backend as K

if __name__ == "__main__":
    # Fix random seed for reproducibility.
    seed = 777
    np.random.seed(seed)
    print ('Fixed random seed for reproducibility.')

    # Load file path of test.csv from arguments.
    TEST_CSV_FILE_PATH = sys.argv[1]
    # Load file path of prediction.csv from arguments.
    PREDICTION_CSV_FILE_PATH = sys.argv[2]
    
    # Input image dimensions.
    rowsImage, colsImage = 48, 48

    print ('Reading labels and features from training set.')
    # Read features from testing set (column 1: feature) using pandas.
    x_features_test = pandas.read_csv(TEST_CSV_FILE_PATH).values[:, 1]
    x_data_test = np.zeros((x_features_test.shape[0], rowsImage * colsImage))
    # Split each feature to (rowsImage * colsImage) array.
    for n in range(x_features_test.shape[0]):
        x_data_test[n] = x_features_test[n].split(' ')
    # Reshape (rowsImage * colsImage) to (rowsImage, colsImage) array.
    if K.image_data_format() == 'channels_first':
        x_data_test = x_data_test.reshape((x_data_test.shape[0], 1, rowsImage, colsImage))
    else:
        x_data_test = x_data_test.reshape((x_data_test.shape[0], rowsImage, colsImage, 1))

    # Scale to 0 ~ 1 on testing set.
    x_data_test = x_data_test / 255.0
    print ('Scaled to 0 ~ 1 on training set.')

    # # Flatten x_data_train.
    # x_data_test = x_data_test.reshape(x_data_test.shape[0], -1)

    # # Read std and mean from npy files.
    # x_data_train_std = np.load('best_std.npy')
    # x_data_train_mean = np.load('best_mean.npy')
    # # Normalization on training set.
    # x_data_test = x_data_test - x_data_train_mean / (x_data_train_std + 1e-20)
    # print ('Normalized on testing set.')

    # Load model from h5 file.
    model = load_model('GEN_CNN_BEST.h5')
    # model = load_model('GEN_DNN.h5')
    # model = load_model('GEN_CNN.h5')

    # # Predict y.
    y_data_test = model.predict(x_data_test)
    # Convert from one hot to 1 column.
    y_data_test = y_data_test.argmax(1)

    stringID = list()
    for n in range(y_data_test.shape[0]):
        stringID.append(str(n))

    resultCSV_dict = {
        "id":       stringID,
        "label":    y_data_test
    }
    resultCSV = pandas.DataFrame(resultCSV_dict)
    resultCSV.to_csv(PREDICTION_CSV_FILE_PATH, index=False)