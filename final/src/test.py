# python3 test.py ./testing_data.csv ./prediction.csv ./w2vModel_256_w16_mc3.bin

import sys
import numpy as np
import pandas as pd
import pickle
import jieba
from gensim.models import Word2Vec
from scipy import spatial

# Parameters of text classification.
EMBEDDING_DIM = 256

if __name__ == "__main__":
    # Fix random seed for reproducibility.
    seed = 777
    np.random.seed(seed)
    print ('Fixed random seed for reproducibility.')
    
    # Load file path of test.csv from arguments.
    TEST_CSV_FILE_PATH = sys.argv[1]
    # Load file path of prediction.csv from arguments.
    PREDICTION_CSV_FILE_PATH = sys.argv[2]
    # Load file path of w2v model from arguments.
    W2V_MODEL_FILE_PATH = sys.argv[3]

    # jieba custom setting.
    jieba.set_dictionary('jieba_dict/dict.txt+dict.txt.big.TW')
    # jieba.set_dictionary('jieba_dict/dict.txt')
    # jieba.set_dictionary('jieba_dict/dict.txt.big.TW')

    # # Add and delete some words in dictionary.
    # # Open del_words.txt file.
    # with open('./del_words.txt', 'r') as f:
    #     for line in f:
    #         jieba.del_word(line.strip())
    # # Open add_words.txt file.
    # with open('./add_words.txt', 'r') as f:
    #     for line in f:
    #         jieba.add_word(line.strip())

    # Load pre-trained Word2Vec model.
    print('Loading pre-trained Word2Vec model...')
    word2vec = Word2Vec.load(W2V_MODEL_FILE_PATH)

    # Read test dataframe from csv file.
    test_df = pd.read_csv(TEST_CSV_FILE_PATH)
    # Prepare id of testing data.
    id = test_df['id']
    dialogue = test_df['dialogue'].values
    options = test_df['options'].str.split('\t')

    # Remove 'A:', 'B:', etc in dialogue.
    for index in range(dialogue.shape[0]):
        dialogue[index] = dialogue[index].replace('\t', '').replace('\n', '').replace(' ', '').replace('Ａ:', '').replace('A:', '').replace('B:', '').replace('C:', '').replace('D:', '').replace('E:', '')

    # Remove 'A:', 'B:', etc in options.
    for index in range(options.shape[0]):
        for op in range(len(options[index])):
            options[index][op] = options[index][op].replace('\n', '').replace(' ', '').replace('Ａ:', '').replace('A:', '').replace('B:', '').replace('C:', '').replace('D:', '').replace('E:', '')

    # Create an empty list for OOV.
    oov_word = []
    oov_vec = []
    # Create an empty list for answer.
    ans = []
    for index in range(dialogue.shape[0]):
        avg_dlg_emb = []
        # jieba.cut 會把dialogue作分詞
        # 對於有在word2vec裡面的詞我們才把它取出
        # 最後詞向量加總取平均 作為句子的向量表示
        for word in jieba.cut(dialogue[index]):
            if word in word2vec:
                avg_dlg_emb.append(word2vec[word])
            else:
                # If this word didn't recorded in oov.
                if word not in oov_word:
                    # Record this word in list.
                    oov_word.append(word)
                    # Set a random vector to this oov word.
                    oov_vec.append(np.random.uniform(-9, 9, size=EMBEDDING_DIM))
                # avg_dlg_emb.append(np.zeros(EMBEDDING_DIM))
                avg_dlg_emb.append(oov_vec[oov_word.index(word)])
        avg_dlg_emb = np.array(avg_dlg_emb).mean(axis=0)

        max_idx = -1
        max_sim = -1000
        # 在六個回答中 每個答句都取詞向量平均作為向量表示
        # 我們選出與dialogue句子向量表示cosine similarity最高的短句
        for idx, opt in enumerate(options[index]):
            avg_opt_emb = []
            for word in jieba.cut(opt):
                if word in word2vec:
                    avg_opt_emb.append(word2vec[word])
                else:
                    # If this word didn't recorded in oov.
                    if word not in oov_word:
                        # Record this word in list.
                        oov_word.append(word)
                        # Set a random vector to this oov word.
                        oov_vec.append(np.random.uniform(-9, 9, size=EMBEDDING_DIM))
                    # avg_opt_emb.append(np.zeros(EMBEDDING_DIM))
                    avg_opt_emb.append(oov_vec[oov_word.index(word)])
            avg_opt_emb = np.array(avg_opt_emb).mean(axis=0)
            # Cosine similarity.
            # sim = 1 - spatial.distance.cosine(avg_dlg_emb, avg_opt_emb)
            sim = np.dot(avg_dlg_emb, avg_opt_emb) / np.linalg.norm(avg_dlg_emb) / np.linalg.norm(avg_opt_emb)
            # print("Ans#%d: %f" % (idx, sim))
            if sim > max_sim:
                max_idx = idx
                max_sim = sim
        # print("Answer:%d" % max_idx)
        ans.append(max_idx)

    prediction_df = pd.DataFrame(
        {'id': id, 'ans': ans}, 
        columns=('id', 'ans'))
    prediction_df.to_csv(PREDICTION_CSV_FILE_PATH, 
        index=False, columns=('id', 'ans'))