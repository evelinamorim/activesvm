import sys
import os
import random
import time

import numpy as np
from scipy import sparse
from sklearn import svm, linear_model, metrics
from sklearn.externals import joblib

from data import DataList
from utils import *
import matplotlib.pyplot as plt
from scipy.interpolate import spline

NFEATURES = 10000

random.seed(179426321)

# Create a canvas to place the subgraphs
canvas = plt.figure()

class Experiments:

    def __init__(self, split_by_class = False, data_format = 'sgm', 
                 split_type = 'LEWISSPLIT'):

        self.train_data = []
        self.heldout_data = [] # it can be a pool as well
        self.test_data = []
        self.labels = []
        self.split = split_by_class
        self.split_type = split_type
        self.data_format = data_format
        self.data_obj = DataList(NFEATURES, split_type)

        self._nfold = 10
        self._idx_fold = -1

        self.L = 59

    def set_idx_fold(self, i):
        self._idx_fold = i

    def set_nfold(self, nfold):
        self._nfold = nfold

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


    def write_data(self, out_file, X, y):
        """
        Write features in a libsvm format
        """

        self.data_obj.write_data(out_file, X, y)

    def load_test_data(self, out_file):
        """
        Load features from a libsvm format
        """

        return self.data_obj.load_test_data(out_file)

    def load_data(self, out_file):
        """
        Load features from a libsvm format
        """

        print("Loading data %s..." % out_file)

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

       
        X_train = list(zip(positive, [1]*npositive)) +   list(zip(negative, [0]*nnegative))
        random.shuffle(X_train)


        clf = svm.SVC(kernel = 'linear')
        while X_train != []:
            ntrain = len(X_train)
            # print(ntrain)
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

    def guided_learning(self, model_dir, test_file, max_batch_size, npos, nneg):

        b = 250
        step = 250
        rho = 0.5

        count_pos = 0
        count_neg = 0

        current_npos = npos
        current_nneg = nneg

        X_test, y_test = self.load_test_data(test_file)
 
        num_instances = []
        auc_values = []
        while b < max_batch_size:

            if current_npos > (rho*step):
                current_npos = current_npos - (rho*step)
                count_pos = count_pos + (rho*step)
            else:
                count_pos = count_pos + current_npos
                current_npos = 0

            if current_nneg > ((1-rho)*step):
                current_nneg = current_nneg - ((1-rho)*step)
                count_neg = count_neg +  ((1-rho)*step)
            else:
                count_neg = count_neg + current_neg
                current_nneg = 0

            num_instances.append(count_neg + count_pos)
            clf = joblib.load('%s/logit_b%d.pkl' % (model_dir, b))
            scores = clf.predict_proba(X_test)
            fpr, tpr, thresholds = metrics.roc_curve(y_test, scores[:, 1])
            roc_auc = metrics.auc(fpr, tpr)
            auc_values.append(roc_auc)
            # print("AUC: %.3f" % roc_auc)
            b = b + step         

        x_smooth = np.linspace(min(num_instances), max(num_instances), 200)
        y_smooth = spline(num_instances, auc_values, x_smooth)

        sp1 = canvas.add_subplot(1,1,1, axisbg='w')
        sp1.plot(x_smooth, y_smooth, 'blue', linewidth=1)

        sp1.set_xlabel('#instances')
        sp1.set_ylabel('AUC')
        sp1.set_title('Crude dataset')



        # plt.savefig("crude_.png")
        plt.show()

    def get_ninstances(self, lst_ninstances, X, n):

        elem = list(range(X.shape[0]))
        elem_idx = random.sample(elem, n)

        for idx in elem_idx:
            if lst_ninstances is None:
                lst_ninstances = X.getrow(idx)
            else:
                # print(type(lst_ninstances), type( X.getrow(idx)))
                lst_ninstances = sparse.vstack([lst_ninstances, X.getrow(idx)])
        elem_idx.sort()
        while elem_idx != []:
            idx = elem_idx.pop()
            delete_row_csr(X, idx)

        return lst_ninstances

    def guided_learning_train(self, output_dir, input_data_file, max_batch_size, fold = None):

        if fold is not None:
            X_train, y_train, X_test, y_test = self.load_data("%s_fold%d" % (input_data_file, fold))
        else:
            X_train, y_train, X_test, y_test = self.load_data(input_data_file)

        rho = 0.5
        epochs = 10

        # split positive and negative
        start = time.time()
        train_pos, train_neg = self.split_pos_neg(X_train, y_train)
        print("Split dataset: ", time.time()-start, "s")
        npos = train_pos.shape[0]        
        nneg = train_neg.shape[0]        

        # train b instances 
        b = 250 
        step = 250
        batch_pos_len = 0
        batch_neg_len = 0
        batch_pos = None
        batch_neg = None

        lr_clf = linear_model.LogisticRegression()

        while b < max_batch_size:
            npos = int(rho*step)
            nneg = int((1-rho)*step)

            current_npos = train_pos.shape[0]
            current_nneg = train_neg.shape[0]


            if current_npos > step:
                batch_pos = self.get_ninstances(batch_pos, train_pos, npos)
            else:
                batch_pos = self.get_ninstances(batch_pos, train_pos, current_npos)

            if current_nneg > step:
                batch_neg = self.get_ninstances(batch_neg, train_neg, nneg)
            else:
                batch_neg = self.get_ninstances(batch_neg, train_pos, current_nneg)

            batch_train = sparse.vstack([batch_pos, batch_neg])
            # ninstances esta zerando train_pos e train_neg...rever

            batch_neg_size = batch_neg.shape[0]
            batch_pos_size = batch_pos.shape[0]
            y = batch_pos_size*[1] + batch_neg_size*[-1]

            print("Training batch %d[%d] ..." % (b,len(y)))
            lr_clf.fit(batch_train, y)
            if fold is not None:
                joblib.dump(lr_clf, '%s/logit_b%d.pkl_fold%d' % (output_dir, b, fold))
            else:
                joblib.dump(lr_clf, '%s/logit_b%d.pkl' % (output_dir, b))
            b = b + step

    def guided_learning_models(self, output_dir, input_data_file, max_batch_size):       
        """
         It generates models to perform a guided learning simulation

        """

        if self.split_type == "CROSSVALIDATION":
            for i in range(self._nfold):
                self.guided_learning_train(output_dir, input_data_file, max_batch_size, i)
        else:
            self.guided_learning_train(output_dir, input_data_file, max_batch_size)


    def build_file_data(self, class_names, features_file, fold = None):

        for c in class_names:
            start_split_class = time.time()
            train_p, train_n, test_p, test_n = self.data_obj.split_by_class(c)
            print("Split by class time: ", time.time() - start_split_class,"s")
            
            numntrain = train_n.shape[0]
            numptrain = train_p.shape[0]
            X = sparse.vstack([train_p, train_n])

            y = [1]*numptrain + [0]*numntrain
            if fold is not None:
                ftr_file_name = 'features/train_%s_%s_fold%d' % (c, features_file, fold)
            else:
                ftr_file_name = 'features/train_%s_%s' % (c, features_file)
            print('Writing %s...' % ftr_file_name)
            self.write_data(ftr_file_name, X, y)

            numntest = test_n.shape[0]
            numptest = test_p.shape[0]
            X =  sparse.vstack([test_p, test_n])
            y = [1]*numptest + [0]*numntest
            if fold is not None:
                ftr_file_name = 'features/test_%s_%s_fold%d' % (c, features_file, fold)
            else:
                ftr_file_name = 'features/test_%s_%s' % (c, features_file)
            print('Writing %s...' % ftr_file_name)
            self.write_data(ftr_file_name, X, y)

    def process_raw_data(self, dir_name, class_names, features_file):

        self.read_data(dir_name)
        if self.split_type == 'CROSSVALIDATION':
            self.data_obj.shuffle_data()
            for i in range(self._nfold):
                self.data_obj.build_fold(i, self._nfold)
                start_vectorize = time.time()
                self.data_obj.vectorize()
                print("Vectorize time: ", time.time() - start_vectorize,"s")
                self.build_file_data( class_names, features_file, i)
        else:
            self.data_obj.vectorize()
            self.build_file_data(class_names, features_file)


    def perform_vec_data(self, data_file):

        X_train, y_train, X_test, y_test = self.load_data(data_file)

        # split positive and negative
        train_p, train_n = self.split_pos_neg(X_train, y_train)
        test_p, test_n = self.split_pos_neg(X_test, y_test)

        # active learning

        self.active_learning(train_p, train_n)

    def perform(self, dir_name = '', data_file = '', class_names  = []):
 
        
        if data_file == '' and dir_name != '':
            self.process_raw_data(dir_name, class_names, 'ftr_file.lsvm')
        elif dir_name == '' and data_file != '':
            self.perform_vec_data(data_file)
        else:
            print('Something is wrong with your data entry...')
            return -1

            
    
if __name__ == '__main__':
    option = sys.argv[1]

    split_data = 'CROSSVALIDATION' # or LEWISSPLIT (default)
    e = Experiments(split_type = split_data)

    # class_names = ['corn', 'crude', 'grain', 'interest', 'money-fx', 'ship', 'wheat', 'trade']
    # class_names = ['crude']
    class_names = ['sci']
    if option == '-d':
        dir_name = sys.argv[2]
        e.perform(dir_name = dir_name, class_names = class_names)
    elif option == '-f':
        data_file = sys.argv[2]
        e.perform(data_file = data_file)
    elif option ==  '-gm':
        output_dir = sys.argv[2]
        data_file = sys.argv[3]
        max_iter_batch = int(sys.argv[4])
        e.guided_learning_models(output_dir, data_file, max_iter_batch)
    elif option == '-gt':
        model_dir = sys.argv[2]
        test_file = sys.argv[3]
        max_iter_batch = int(sys.argv[4])
        npos = int(sys.argv[5])
        nneg = int(sys.argv[6])
        
        e.set_idx_fold(int(sys.argv[7]))
        e.guided_learning(model_dir, test_file, max_iter_batch, npos, nneg)


    # else:
    #    print('Something is wrong with your option...')
