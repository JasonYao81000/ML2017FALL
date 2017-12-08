# #!/bin/bash
# bash hw4_train.sh ./training_label.txt ./training_nolabel.txt
# python3 train_w2v.py ./training_label.txt ./training_nolabel.txt

import sys
import numpy as np
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, LeakyReLU
from keras.models import Model
from keras.models import Sequential
# from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.recurrent import GRU
from keras.layers.recurrent import SimpleRNN
from keras.layers.wrappers import Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# Parameters of text classification.
MAX_SEQUENCE_LENGTH = 39
MAX_NB_WORDS = 8192
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

if __name__ == "__main__":
    # Fix random seed for reproducibility.
    seed = 777
    np.random.seed(seed)
    print ('Fixed random seed for reproducibility.')

    # Load file path of training_label.txt from arguments.
    TRAIN_LABEL_FILE_PATH = sys.argv[1]
    # Load file path of training_semi_label.txt from arguments.
    TRAIN_SEMI_LABEL_FILE_PATH = sys.argv[2]

    print ('Reading texts and labels from training set...')
    # Assign empty list to texts, labels.
    texts = list()
    labels = list()
    # Open text file.
    with open(TRAIN_LABEL_FILE_PATH) as f:
        # Read line by line.
        for line in f:
            item = line.strip().split(' +++$+++ ')
            # Characters filters.
            filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
            for c in filters:
                item[1] = item[1].replace(c, ' ')
            # Split texts by ' '.
            texts.append(item[1].split(' '))
            labels.append(item[0])

    # print ('Reading texts and labels from training set (semi-labels)...')
    # # Counter of semi-labels data.
    # countSemiLabel = 0
    # countNoLabel = 0
    # # Open text file.
    # with open(TRAIN_SEMI_LABEL_FILE_PATH) as f:
    #     # Read line by line.
    #     for line in f:
    #         # Count how many no label data.
    #         countNoLabel += 1
    #         item = line.strip().split(' +++$+++ ')
    #         # Only read hard-labeled.
    #         if (item[0] != '-1'):
    #             # Avoid reading null string.
    #             if (len(item) == 2):
    #                 # Count how many semi-label data.
    #                 countSemiLabel += 1
    #                 texts.append(item[1])
    #                 labels.append(item[0])
    # print ("There are %d no label data." %(countNoLabel))
    # print ("There are %d semi-label data." %(countSemiLabel))

    # Load pre-trained Word2Vec model.
    print('Loading pre-trained Word2Vec model...')
    word2vec = Word2Vec.load('w2vModel_1.bin')
    # Convert texts to vectors.
    print('Converting texts to vectors...')
    dataVector = np.zeros((len(texts), MAX_SEQUENCE_LENGTH, EMBEDDING_DIM))
    for n in range(len(texts)):
        for i in range(min(len(texts[n]), MAX_SEQUENCE_LENGTH)):
            try:
                vector = word2vec[texts[n][i]]
                dataVector[n][i] = (vector - vector.mean(0)) / (vector.std(0) + 1e-20)
            except KeyError as e:
                print ('Word', texts[n][i], 'is not in dictionary.')

    # dataVector = []
    # for i in range(len(texts)):
    #     dataVector.append([])
    #     for j in range(len(texts[i])):
    #         try:
    #             vector = word2vec[texts[i][j]]
    #             dataVector[len(dataVector) - 1].append(vector)
    #             # Normalize [x=(x-mean(x))/std(x)]
    #             dataVector[i][j] = np.array((dataVector[i][j] - np.mean(dataVector[i][j])) / np.std(dataVector[i][j]))
    #         except KeyError as e:
    #             dataVector[len(dataVector) - 1].append(np.zeros(len(texts[i])))
    #     print ("Data %d" %i)

    # dataVector = np.array(dataVector)
    # dataVector = np.array((dataVector - np.mean(dataVector)) / np.std(dataVector))

    # # Vectorize the text samples into a 2D integer tensor.
    # print ('Vectorizing the text samples into a 2D integer tensor...')
    # tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    # tokenizer.fit_on_texts(texts)
    # sequences = tokenizer.texts_to_sequences(texts)

    # word_index = tokenizer.word_index
    # print('Found %s unique tokens.' % len(word_index))

    # # Truncate and pad input sequences.
    # data = sequence.pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    # Convert labels to one hot encoding.
    print ('Converting labels to one hot encoding...')
    labels = to_categorical(np.asarray(labels))
    # Get number of classes.
    numClasses = labels.shape[1]
    print ('There are', numClasses, 'classes.')

    print('Shape of dataVector tensor:', dataVector.shape)
    print('Shape of label tensor:', labels.shape)

    # Split the dataVector into a training set and a validation set.
    indices = np.arange(dataVector.shape[0])
    np.random.shuffle(indices)
    dataVector = dataVector[indices]
    labels = labels[indices]
    num_validation_samples = int(VALIDATION_SPLIT * dataVector.shape[0])
    
    # x_train = dataVector[:-num_validation_samples]
    # y_train = labels[:-num_validation_samples]
    # x_val = dataVector[-num_validation_samples:]
    # y_val = labels[-num_validation_samples:]
    x_train = dataVector
    y_train = labels

    # # Load pre-trained Word2Vec model.
    # print('Loading pre-trained Word2Vec model...')
    # word2vec = Word2Vec.load('my_corpus.model')

    # # Prepare embedding matrix.
    # print('Preparing embedding matrix...')
    # num_words = min(MAX_NB_WORDS, len(word_index))
    # embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    # for word, i in word_index.items():
    #     if i >= MAX_NB_WORDS:
    #         continue
    #     try:
    #         # embedding_vector = [x[1] for x in glove.most_similar(word, number = EMBEDDING_DIM + 1)]
    #         embedding_vector = word2vec[word]
    #         if embedding_vector is not None:
    #             # Words not found in embedding index will be all-zeros.
    #             embedding_matrix[i] = embedding_vector
    #     except:
    #         print ('Word', word, 'is not in dictionary.')

    # # Load pre-trained word embeddings into an Embedding layer.
    # # Note that we set 'trainable = False', so as to keep the embeddings fixed.
    # embedding_layer = Embedding(num_words,
    #                             EMBEDDING_DIM,
    #                             weights=[embedding_matrix],
    #                             input_length=MAX_SEQUENCE_LENGTH,
    #                             trainable=False)

    print('Building model...')
    model = Sequential()
    # model.add(embedding_layer)
    # model.add(LSTM(units=256, input_shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM), return_sequences=True))
    # model.add(LSTM(units=128, return_sequences=False))
    model.add(GRU(units=128, input_shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM), dropout=0.1, recurrent_dropout=0.1))
    model.add(Dense(units=256, kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    # model.add(Dense(units=128, kernel_initializer='glorot_normal'))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(units=64, kernel_initializer='glorot_normal'))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(units=32, kernel_initializer='glorot_normal'))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(numClasses, activation = 'softmax'))
    print ('Built the model.')
    print (model.summary())
    
    # Compile the model.
    # model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print ('Compiled the model.')

    # Config the early stopping.
    earlyStopping = EarlyStopping(monitor='val_acc', patience=3, verbose=1, mode='auto')

    print('Training model...')
    # Fit the model.
    fitHistory = model.fit(x_train, y_train,
        batch_size=128, 
        epochs=16, 
        # validation_data=(x_val, y_val), 
        # callbacks=[earlyStopping], 
        verbose=1)

    # Save model to h5 file.
    model.save('w2v.h5')

    # # Save history of acc to npy file.
    # np.save('train_acc_history_w2v.npy', fitHistory.history['acc'])
    # np.save('valid_acc_history_w2v.npy', fitHistory.history['val_acc'])

    # # Report index of highest accuracy in training set and validation set.
    # print ('tra_acc: ', np.amax(fitHistory.history['acc']), 'at epochs = ', np.argmax(fitHistory.history['acc']))
    # print ('val_acc: ', np.amax(fitHistory.history['val_acc']), 'at epochs = ', np.argmax(fitHistory.history['val_acc']))