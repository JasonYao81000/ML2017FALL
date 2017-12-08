# #!/bin/bash
# bash hw4_test.sh ./testing_data.txt ./prediction.csv
# python3 test_w2v.py ./testing_data.txt ./prediction.csv
# python3 train_w2v.py ./training_label.txt ./training_semi_label_0.93_18.txt

import sys
import numpy as np
import pandas
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import load_model
from keras import layers

# Parameters of text classification.
MAX_SEQUENCE_LENGTH = 39
MAX_NB_WORDS = 8192
EMBEDDING_DIM = 100

if __name__ == "__main__":
    # Fix random seed for reproducibility.
    seed = 777
    np.random.seed(seed)
    print ('Fixed random seed for reproducibility.')

    # Load file path of testing_data.txt from arguments.
    TEST_FILE_PATH = sys.argv[1]
    # Load file path of prediction.csv from arguments.
    PREDICTION_CSV_FILE_PATH = sys.argv[2]

    print ('Reading texts from testing set.')
    # Assign empty list to text.
    texts = list()
    # Open text file.
    with open(TEST_FILE_PATH) as f:
        # Read line by line.
        for line in f:
            deleteString = line.strip().split(',')[0] + ','
            item = line.replace(deleteString, '')
            # Characters filters.
            filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
            for c in filters:
                item = item.replace(c, ' ')
            # Split texts by ' '.
            texts.append(item.split(' '))
    # Remove first element in list.
    texts.pop(0)

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

    # # Vectorize the text samples into a 2D integer tensor.
    # print ('Vectorizing the text samples into a 2D integer tensor...')
    # tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    # tokenizer.fit_on_texts(texts)
    # sequences = tokenizer.texts_to_sequences(texts)

    # word_index = tokenizer.word_index
    # print('Found %s unique tokens.' % len(word_index))

    # # Truncate and pad input sequences.
    # data = sequence.pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    print('Shape of dataVector tensor:', dataVector.shape)

    xTest = dataVector

    # # Load pre-trained GloVe model.
    # print('Loading pre-trained GloVe model...')
    # glove = Glove.load('glove.model')

    # # Prepare embedding matrix.
    # print('Preparing embedding matrix...')
    # num_words = min(MAX_NB_WORDS, len(word_index))
    # embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    # for word, i in word_index.items():
    #     if i >= MAX_NB_WORDS:
    #         continue
    #     try:
    #         # embedding_vector = [x[1] for x in glove.most_similar(word, number = EMBEDDING_DIM + 1)]
    #         embedding_vector = glove.word_vectors[glove.dictionary[word]]
    #         if embedding_vector is not None:
    #             # Words not found in embedding index will be all-zeros.
    #             embedding_matrix[i] = embedding_vector
    #     except:
    #         print ('Word', word, 'is not in dictionary.')

    # Load model from h5 file.
    model = load_model('w2v.h5')
    print ('Loaded the model.')

    # # Change the weights of embbeding layer.
    # embedding_layer = model.get_layer('embedding_1')
    # embedding_layer.set_weights([embedding_matrix])
    # print ('Changed the weights of embbeding layer.')
    # print (model.summary())

    # Predict labels.
    print ('Predicting labels...')
    labels = model.predict(xTest, verbose=1)
    # Convert from one hot to 1 column.
    yTest = labels.argmax(1)
    # yTest = (labels > 0.5).astype(np.int)

    stringID = list()
    for n in range(yTest.shape[0]):
        stringID.append(str(n))

    resultCSV_dict = {
        "id":       stringID,
        "label":    yTest.reshape(-1)
    }
    resultCSV = pandas.DataFrame(resultCSV_dict)
    resultCSV.to_csv(PREDICTION_CSV_FILE_PATH, index=False)