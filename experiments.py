import sys
import os
import random
import numpy as np
from scipy import sparse
from sklearn import svm

from data import DataList
from utils import *

NFEATURES = 10000

random.seed(179426321)

class Experiments:

    def __init__(self, split_by_class = False, data_format = 'sgm'):

        self.train_data = []
        self.heldout_data = [] # it can be a pool as well
        self.test_data = []
        self.labels = []
        self.split = split_by_class
        self.data_format = data_format
        self.data_obj = DataList(NFEATURES)

        self.L = 59


    def split_pos_neg(self, X, y):

        ninstances = X.shape[0]

        positive = None
        negative = None

        for i in range(ninstances):
            if int(y[i]) == 1:
                if positive == None:
                    positive = X.getrow(i)
                else:
                    positive = sparse.vstack([positive, X.getrow(i)])
            else:
                if negative == None:
                    negative = X.getrow(i)
                else:
                    negative = sparse.vstack([negative, X.getrow(i)])

        return positive, negative

    def read_data(self, dir_name, out_file = 'model_svm.lsv'):



        # read the files with text data
        i = 0
        for subdir, dirs, files in os.walk(dir_name):
            for f in files:
                file_path = subdir + os.path.sep + f
                print(file_path)
                self.data_obj.read(file_path)
                if file_path.endswith('sgm'):
                    i = i + 1

        self.data_obj.vectorize()

    def write_data(self, out_file, X, y):
        """
        Write features in a libsvm format
        """

        self.data_obj.write_data(out_file, X, y)

    def load_data(self, out_file):
        """
        Load features from a libsvm format
        """

        return self.data_obj.load_data(out_file)

    def active_learning(self, positive, negative, per_queries = 0.01):
        """
        per_queries: percentage of data that should be selected in a pool of 
        1000 examples
        """

        npositive = positive.shape[0]
        nnegative = negative.shape[0]
        X_activeset = sparse.vstack([positive.getrow(npositive - 1), negative.getrow(nnegative - 1)])
        y_activeset = [1, 0]

        delete_row_csr(positive, npositive - 1)
        delete_row_csr(negative, nnegative - 1)


        X_train = [(p, 1)  for p in positive] +  [(n, 0)  for n in negative]
        random.shuffle(X_train)


        clf = svm.SVC(kernel = 'linear')
        while X_train != []:
            ntrain = len(X_train)
            print(ntrain)
            if self.L < ntrain:
                small_pool_idx = random.sample(range(ntrain), self.L)
            else:
                small_pool_idx = list(range(ntrain))

            # nao existe modo mais rapido de retreinar o svm do scikit?
            clf.fit(X_activeset, y_activeset)
            
            small_pool = [(clf.decision_function(X_train[i][0]), X_train[i][1], i) for i in small_pool_idx]

            small_pool = sorted(small_pool, key = lambda t: abs(t[0]))
            # pegar os mais proximos em small pool do hiperplano de svm_model
            # nao tem como tirar mais de um? Nao, no artigo:
            # i) learn an SVM on the existing training data
            # ii) select the closest instance to the hyperplane
            # iii) add the new selected instance to the training set and train again

            dist, s_y, idx = small_pool[0]
            s_x, s_y = X_train.pop(idx)
            X_activeset = sparse.vstack([X_activeset, s_x])
            y_activeset = y_activeset + [s_y]
        
    def perform_raw_data(self, dir_name, class_names, features_file):

        self.read_data(dir_name)
        for c in class_names:
            train_p, train_n, test_p, test_n = self.data_obj.split_by_class(c)
            
            numntrain = train_p.shape[0]
            numptrain = train_n.shape[0]
            X = sparse.vstack([train_p, train_n])

            y = [1]*numptrain + [0]*numntrain
            ftr_file_name = 'features/train_%s_%s' % (c, features_file)
            print('Writing %s...' % ftr_file_name)
            self.write_data(ftr_file_name, X, y)

            numntest = test_p.shape[0]
            numptest = test_n.shape[0]
            X =  sparse.vstack([test_p, test_n])
            y = [1]*numptest + [0]*numntest
            ftr_file_name = 'features/test_%s_%s' % (c, features_file)
            print('Writing %s...' % ftr_file_name)
            self.write_data(ftr_file_name, X, y)

            # it returns a model and a new train data
            # self.active_learning(train_p, train_n)

    def perform_vec_data(self, data_file):

        X_train, y_train, X_test, y_test = self.load_data(data_file)

        # split positive and negative
        train_p, train_n = self.split_pos_neg(X_train, y_train)
        test_p, test_n = self.split_pos_neg(X_test, y_test)

        # active learning

        self.active_learning(train_p, train_n)

    def perform(self, dir_name = '', data_file = '', class_names  = []):
 
        
        if data_file == '' and dir_name != '':
            self.perform_raw_data(dir_name, class_names, 'ftr_file.lsvm')
        elif dir_name == '' and data_file != '':
            self.perform_vec_data(data_file)
        else:
            print('Something is wrong with your data entry...')
            return -1

            
    
if __name__ == '__main__':
    option = sys.argv[1]

    e = Experiments()

    class_names = ['crude', 'grain', 'interest', 'money-fx', 'ship', 'wheat']
    if option == '-d':
        dir_name = sys.argv[2]
        e.perform(dir_name = dir_name, class_names = class_names)
    elif option == '-f':
        data_file = sys.argv[2]
        e.perform(data_file = data_file)
    # else:
    #    print('Something is wrong with your option...')
