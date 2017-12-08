#!/bin/bash
wget -O w2vModel_1.bin https://www.dropbox.com/s/0xvfk0pcr1zisr5/w2vModel_1.bin?dl=1
wget -O w2vModel_1.bin.syn1neg.npy https://www.dropbox.com/s/sd04jjfnsbov1hq/w2vModel_1.bin.syn1neg.npy?dl=1
wget -O w2vModel_1.bin.wv.syn0.npy https://www.dropbox.com/s/996snqnr3mqjz9i/w2vModel_1.bin.wv.syn0.npy?dl=1
python3 test_w2v.py $1 $2