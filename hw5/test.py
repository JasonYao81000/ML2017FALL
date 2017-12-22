# python3 test.py ./test.csv ./prediction.csv ./movies.csv ./users.csv ./train.csv 1024 ./model.h5
# python3 test.py ./test.csv ./prediction.csv ./movies.csv ./users.csv ./model.h5
# bash hw5.sh ./test.csv ./prediction.csv ./movies.csv ./users.csv

import sys
import numpy as np
import pandas as pd
import pickle
import keras
import keras.backend as K
from keras.models import load_model

# Parameters of text classification.
# MAX_SEQUENCE_LENGTH = 39
# MAX_NB_WORDS = 8192
MAX_EPOCHS = 8192
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.1

def rmse(y_true, y_pred):
    # y_pred = K.clip(y_pred, 1.0, 5.0)
    return K.sqrt(K.mean(K.pow(y_true - y_pred, 2)))

if __name__ == "__main__":
    # Fix random seed for reproducibility.
    seed = 777
    np.random.seed(seed)
    print ('Fixed random seed for reproducibility.')

    # Load file path of test.csv from arguments.
    TEST_CSV_FILE_PATH = sys.argv[1]
    # Load file path of prediction.csv from arguments.
    PREDICTION_CSV_FILE_PATH = sys.argv[2]
    # Load file path of movies.csv from arguments.
    MOVIES_CSV_FILE_PATH = sys.argv[3]
    # Load file path of users.csv from arguments.
    USERS_CSV_FILE_PATH = sys.argv[4]
    # # Load file path of train.csv from arguments.
    # TRAIN_CSV_FILE_PATH = sys.argv[5]
    # # Load epochs to train from arguments.
    # EPOCHS = int(sys.argv[6])
    # Load file path of model from arguments.
    MODEL_FILE_PATH = sys.argv[5]

    # Load user2id, movie2id from pickle file.
    user2id = pickle.load(open('user2id.obj', 'rb'))
    movie2id = pickle.load(open('movie2id.obj', 'rb'))
    
    # Read test dataframe from csv file.
    test_df = pd.read_csv(TEST_CSV_FILE_PATH)

    # Apply unique index to ID column.
    test_df['UserID'] = test_df['UserID'].apply(lambda x: user2id[x])
    test_df['MovieID'] = test_df['MovieID'].apply(lambda x: movie2id[x])

    # Prepare X_test, id of users/movies for testing.
    X_test = test_df[['UserID', 'MovieID']].values
    id = test_df['TestDataID']

    # Load model from h5 file.
    model = load_model(MODEL_FILE_PATH, custom_objects={'rmse': rmse})
    print ('Loaded the model.')
    
    # Predict labels.
    print ('Predicting labels...')
    yTest = model.predict([X_test[:, 0], X_test[:, 1]], verbose=1).squeeze()

    # # Load std and mean to npy files.
    # Y_train_std = np.load('std.npy')
    # Y_train_mean = np.load('mean.npy')
    # # Denormalize on yTest.
    # yTest = yTest * Y_train_std + Y_train_mean

    # Limit ratings in range 1 ~ 5.
    yTest = yTest.clip(1.0, 5.0)

    prediction_df = pd.DataFrame({'TestDataID': id, 'Rating': yTest}, columns=('TestDataID', 'Rating'))
    prediction_df.to_csv(PREDICTION_CSV_FILE_PATH, index=False, columns=('TestDataID', 'Rating'))