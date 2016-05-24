#!/bin/bash

# some experiments

#generating models to perform  simulation to guided learning
# python3 experiments.py -gm logit_models crude_tfidf.lsvm 9000
# python3 experiments.py -gt logit_models crude.lsvm 9000 389 9214

NEWS_DIR=/Users/evelin.amorim/Documents/Cornell/20_newsgroups/
# train 20 news dataset in a crossvalidation setup, # 
python3 experiments.py -gl $NEWS_DIR  17000

# test each fold for logit model and 20 news dataset
# python3 experiments.py -gt logit_models sci.lsvm_fold0 18000 3599 14440 0
# python3 experiments.py -gt logit_models sci.lsvm_fold1 18000 3618 14381 1
# python3 experiments.py -gt logit_models sci.lsvm_fold2 18000 3607 14392 2
# python3 experiments.py -gt logit_models sci.lsvm_fold3 18000 3590 14409 3
# python3 experiments.py -gt logit_models sci.lsvm_fold4 18000 3625 14374 4
# python3 experiments.py -gt logit_models sci.lsvm_fold5 18000 3607 14392 5
# python3 experiments.py -gt logit_models sci.lsvm_fold6 18000 3595 14404 6
# python3 experiments.py -gt logit_models sci.lsvm_fold7 18000 3584 14415 7
# python3 experiments.py -gt logit_models sci.lsvm_fold8 18000 3584 14415 8
# python3 experiments.py -gt logit_models sci.lsvm_fold9 18000 3592 14407 9

# generate unigram words vectors with booleans weights in crude dataset
# parameters in experiments: vec_type = 'bin', split = 'LEWISSPLIT', class = ['crude'] 
# python3 experiments.py -d /Users/evelin.amorim/Documents/Cornell/reuters-data/
