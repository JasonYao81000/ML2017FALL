#!/bin/bash
wget -O GEN_CNN_BEST.h5 https://www.dropbox.com/s/apj5e5xxo43h0us/GEN_CNN_BEST.h5?dl=1
python3 test.py $1 $2 $3