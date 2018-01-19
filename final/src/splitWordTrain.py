# -*- coding: utf-8 -*-
# python3 splitWordTrain.py

import io
import jieba

jieba.set_dictionary('jieba_dict/dict.txt+dict.txt.big.TW')
# jieba.set_dictionary('jieba_dict/dict.txt')

main_datafile = ['./1_train.txt', './2_train.txt', './3_train.txt', './4_train.txt', './5_train.txt']
output_file = './trainSeg.txt'
output = io.open(output_file, 'w', encoding='utf-8')

for datafile in main_datafile:
    with io.open(datafile, 'r', encoding='utf-8') as content:
        for line in content:
            words = jieba.cut(line, cut_all=False)
            wordCount = 0
            for id, word in enumerate(words):
                if word != '\n' and word != ' ':
                    output.write(word + ' ')
                    wordCount = wordCount + 1
            if wordCount != 0:
                output.write(u'\n')

output.close()