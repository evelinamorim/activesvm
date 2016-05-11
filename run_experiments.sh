#!/bin/bash

# some experiments

#generating models to perform  simulation to guided learning
# python3 experiments.py -gm logit_models crude.lsvm 9000
# python3 experiments.py -gt logit_models crude.lsvm 9000 389 9214

# generate data to 20 news dataset in a crossvalidation set up
# python3 experiments.py -d /Users/evelin.amorim/Documents/Cornell/20_newsgroups/

# train 20 news dataset in a crossvalidation setup, i.e., build a model for each 
# fold 
# python3 experiments.py -gm logit_models sci.lsvm 18000

# test each fold for logit model and 20 news dataset
python3 experiments.py -gt logit_models sci.lsvm_fold0 18000 3599 14440
