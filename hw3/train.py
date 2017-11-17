# #!/bin/bash
# python3 test.py ./test.csv ./prediction.csv
# python3 train.py ./train.csv ./prediction.csv

import sys
import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

if __name__ == "__main__":
    # Fix random seed for reproducibility.
    seed = 777
    np.random.seed(seed)
    print ('Fixed random seed for reproducibility.')

    # Load file path of train.csv from arguments.
    TRAIN_CSV_FILE_PATH = sys.argv[1]
    # # Load file path of test.csv from arguments.
    # TEST_CSV_FILE_PATH = sys.argv[1]
    # Load file path of prediction.csv from arguments.
    PREDICTION_CSV_FILE_PATH = sys.argv[2]

    # Input image dimensions.
    rowsImage, colsImage = 48, 48

    print ('Reading labels and features from training set.')
    # Read features from training set (column 1: feature) using pandas.
    x_features_train = pandas.read_csv(TRAIN_CSV_FILE_PATH).values[:, 1]
    x_data_train = np.zeros((x_features_train.shape[0], rowsImage * colsImage))
    # Split each feature to (rowsImage * colsImage) array.
    for n in range(x_features_train.shape[0]):
        x_data_train[n] = x_features_train[n].split(' ')
    # Reshape (rowsImage * colsImage) to (rowsImage, colsImage) array.
    if K.image_data_format() == 'channels_first':
        x_data_train = x_data_train.reshape((x_data_train.shape[0], 1, rowsImage, colsImage))
    else:
        x_data_train = x_data_train.reshape((x_data_train.shape[0], rowsImage, colsImage, 1))
    # Read y_data_train from training set (column 0: label) using pandas.
    y_data_train = pandas.read_csv(TRAIN_CSV_FILE_PATH).values[:, 0]

    # Scale to 0 ~ 1 on training set.
    x_data_train = x_data_train / 255.0
    print ('Scaled to 0 ~ 1 on training set.')

    # # Standard deviation of X_data_train.
    # x_data_train_std = x_data_train.std(0)
    # # Mean of X_data_train.
    # x_data_train_mean = x_data_train.mean(0)
    # # Save std and mean to npy files.
    # np.save('std.npy', x_data_train_std)
    # np.save('mean.npy', x_data_train_mean)

    # # Normalization on training set.
    # x_data_train = (x_data_train - x_data_train_mean) / (x_data_train_std + 1e-20)
    # print ('Normalized on training set.')

    # Split validation set from training set.
    X_train = x_data_train
    Y_train = y_data_train
    # X_train = x_data_train[:-5000]
    # X_valid = x_data_train[-5000:]
    # Y_train = y_data_train[:-5000]
    # Y_valid = y_data_train[-5000:]
    
    # Convert labels to one hot encoding.
    # y_data_train = to_categorical(y_data_train)
    Y_train = to_categorical(Y_train)
    # Y_valid = to_categorical(Y_valid)
    print ('Converted labels to one hot encoding.')
    # Get number of classes.
    # numClasses = y_data_train.shape[1]
    numClasses = Y_train.shape[1]

    # Generate batches of tensor image data with real-time data augmentation. 
    datagen = ImageDataGenerator(
        rotation_range = 30,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        zoom_range = [0.8, 1.2],
        shear_range = 0.2,
        horizontal_flip = True)
    print ("Generated batches of tensor image data with real-time data augmentation.")

    # Create the model.
    model = Sequential()
    # Conv block 1: 64 output filters.
    model.add(Conv2D(64, kernel_size = (3, 3), padding = 'same', kernel_initializer = 'glorot_normal', input_shape = (rowsImage, colsImage, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 1 / 20))
    model.add(Conv2D(64, kernel_size = (3, 3), padding = 'same', kernel_initializer = 'glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 1 / 20))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))
    # Conv block 2: 128 output filters.
    model.add(Conv2D(128, kernel_size = (3, 3), padding = 'same', kernel_initializer = 'glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 1 / 20))
    model.add(Conv2D(128, kernel_size = (3, 3), padding = 'same', kernel_initializer = 'glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 1 / 20))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    # Conv block 3: 256 output filters.
    model.add(Conv2D(256, kernel_size = (3, 3), padding = 'same', kernel_initializer = 'glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 1 / 20))
    model.add(Conv2D(256, kernel_size = (3, 3), padding = 'same', kernel_initializer = 'glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 1 / 20))
    model.add(Conv2D(256, kernel_size = (3, 3), padding = 'same', kernel_initializer = 'glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 1 / 20))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    # Conv block 4: 512 output filters.
    model.add(Conv2D(512, kernel_size = (3, 3), padding = 'same', kernel_initializer = 'glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 1 / 20))
    model.add(Conv2D(512, kernel_size = (3, 3), padding = 'same', kernel_initializer = 'glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 1 / 20))
    model.add(Conv2D(512, kernel_size = (3, 3), padding = 'same', kernel_initializer = 'glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 1 / 20))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    # Conv block 5: 512 output filters.
    model.add(Conv2D(512, kernel_size = (3, 3), padding = 'same', kernel_initializer = 'glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 1 / 20))
    model.add(Conv2D(512, kernel_size = (3, 3), padding = 'same', kernel_initializer = 'glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 1 / 20))
    model.add(Conv2D(512, kernel_size = (3, 3), padding = 'same', kernel_initializer = 'glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 1 / 20))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    # Fully-connected classifier.
    model.add(Flatten())
    model.add(Dense(units = 4096, kernel_initializer = 'glorot_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units = 4096, kernel_initializer = 'glorot_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units = 1000, kernel_initializer = 'glorot_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(numClasses, activation = 'softmax'))
    print ('Created the model.')
    print (model.summary())
    
    # Compile the model.
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    print ('Compiled the model.')

    # # Fit the model.
    # fitHistory = model.fit(x_data_train, y_data_train,
    #     batch_size = 128, epochs = 100, verbose = 1,
    #     validation_split = 0.2, shuffle = True)

    # Fit the model with datagen.
    fitHistory = model.fit_generator(
        datagen.flow(X_train, Y_train, batch_size = 128),
        # validation_data=(X_valid, Y_valid), 
        steps_per_epoch = len(X_train) // 128,
        epochs = 256)

    # Save model to h5 file.
    model.save('GEN_CNN_BEST.h5')

    # Save history of acc to npy file.
    np.save('train_acc_history_GEN_CNN_BEST.npy', fitHistory.history['acc'])
    # np.save('valid_acc_history_GEN_CNN_BEST.npy', fitHistory.history['val_acc'])

    # Report index of highest accuracy in training set and validation set.
    print ('tra_acc: ', np.amax(fitHistory.history['acc']), 'at epochs = ', np.argmax(fitHistory.history['acc']))
    # print ('val_acc: ', np.amax(fitHistory.history['val_acc']), 'at epochs = ', np.argmax(fitHistory.history['val_acc']))

    # # Remove plt before summit to github.
    # import matplotlib.pyplot as plt
    # # Force matplotlib to not use any Xwindows backend.
    # plt.switch_backend('agg')
    # # Summarize history for accuracy
    # plt.plot(fitHistory.history['acc'])
    # plt.plot(fitHistory.history['val_acc'])
    # plt.title('Accuracy v.s. Epoch')
    # plt.ylabel('Accuracy')
    # plt.xlabel('# of epoch')
    # plt.legend(['train', 'valid'], loc='lower right')
    # plt.savefig('Accuracy v.s. Epoch.png')