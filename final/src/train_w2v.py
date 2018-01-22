# python3 train_w2v.py ./trainSeg.txt ./w2vModel_64_w24_mc3_iter300.bin

import sys
import logging
from gensim.models import word2vec
import multiprocessing

# Load file path of trainSeg.txt from arguments.
TRAIN_SEG_FILE_PATH = sys.argv[1]
# Load file path of w2v model from arguments.
W2V_MODEL_FILE_PATH = sys.argv[2]

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.Text8Corpus(TRAIN_SEG_FILE_PATH)
print ('Training model...')
model = word2vec.Word2Vec(sentences, 
                size=64, window=24, sample=1e-4, 
                negative=10, hs=0, sg=0, iter=300, 
                workers=8, min_count=3)
# model = word2vec.Word2Vec(sentences, size=180,iter =10,window=8, workers=8, min_count=5)
print ('Saving model...')
model.save(W2V_MODEL_FILE_PATH)

# Summarize the loaded model.
print(model)
# Summarize vocabulary.
print(list(model.wv.vocab))
# Access vector for one word.
print(model['嗨'])