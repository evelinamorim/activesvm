import sys
import os
import random
import time
import math
import numpy as np

from scipy import sparse
from scipy.interpolate import spline
from sklearn import svm, linear_model, metrics, cross_validation
from sklearn.externals import joblib
from sklearn.utils import shuffle
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

from data import DataList
from utils import *
import matplotlib.pyplot as plt

NFEATURES = 10000

random.seed(179426321)

# Create a canvas to place the subgraphs
canvas = plt.figure()

# verbose levels
VERBOSE_LOW = 0
VERBOSE_MEDIUM = 1
VERBOSE_HIGH = 2

class Experiments:

    def __init__(self, split_by_class = False, data_format = 'sgm', 
                 split_type = 'LEWISSPLIT', vec_type = 'tfidf'):

        self.train_data = []
        self.heldout_data = [] # it can be a pool as well
        self.test_data = []
        self.labels = []
        self.split = split_by_class
        self.split_type = split_type
        self.data_format = data_format
        self.data_obj = DataList(NFEATURES, split_type, vec_type)

        self._nfold = 10
        self._idx_fold = -1

        self.L = 59
        self.vec_type = vec_type

        self._verbose = VERBOSE_LOW

    def set_verbose_level(self, num):
        self._verbose = num
        self.data_obj.set_verbose_level(num)

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
                if self._verbose >= VERBOSE_HIGH:
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

            if self._idx_fold != -1:
                model_file_name = '%s/logit_b%d.pkl_fold%d' % (model_dir, b, self._idx_fold)
            else:
                model_file_name = '%s/logit_b%d.pkl' % (model_dir, b)

            clf = joblib.load(model_file_name)

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
        sp1.set_title('%s' % model_file_name)



        plt.savefig("%s.png" % model_file_name)
        plt.show()

    def join_instances(self, x, y):

        if type(x) == list and type(y) == list:
            return x + y
        else:
            return sparse.vstack([x,y])

    def get_ninstances_list(self, lst_ninstances, X, n):


        elem = list(range(len(X)))
        if n > len(elem):
            elem_idx = list(range(len(elem)))
        else:
            elem_idx = random.sample(elem, n)

        for idx in elem_idx:
            lst_ninstances.append(X[idx])
        elem_idx.sort()
        while elem_idx != []:
            idx = elem_idx.pop()
            X.pop(idx)

        return lst_ninstances

    def get_ninstances(self, lst_ninstances, X, n):

        if type(X) == list:
            return self.get_ninstances_list(lst_ninstances, X, n)

        elem = list(range(X.shape[0]))
        if n > len(elem):
            elem_idx = list(range(len(elem)))
        else:
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

    def guided_learning_train(self, input_data, max_batch_size, class_name):
        """
        @input_data: if it is LEWISSPLIT split_type, then it is a string 
                      with train/test libsvm files separated by ","
                     if it is CROSSVALIDATION, then it is a directory with raw text
        """

        self.read_data(input_data)

        # delete random instances of class_name class, remove 95% of 
        # class_name class instances
        self.data_obj.remove_ninstances(class_name, r=0.95)
        self.data_obj.shuffle_data()

        cv = 10

        rho = 0.5
        epochs = 10
        batch_size = 250 # batch size

        count_cv = 0
        lr_clf = linear_model.LogisticRegression()

        print("Fold, batch, roc_auc, f1, precision, recall, accuracy, auc, g_means")
        while count_cv < cv:

            # print("Fold %d" % count_cv)

            # split train and test for the i-th fold
            self.data_obj.build_kfold(cv, count_cv)

            train_p, train_n, test_p, test_n = \
            self.data_obj.split_by_class(class_name)

            X_test = self.join_instances(test_p, test_n)
            y_test = len(test_p)*[1] + len(test_n)*[-1]
            # print("2 >>> ", len(test_p), len(test_n))
            # print("0 >>> ", len(train_p), len(train_n))

            len_batch_p = int(batch_size * rho)
            len_batch_n = int(batch_size * (1 - rho))

            b = batch_size
            batch_p = []
            batch_n = []
            while b <= max_batch_size:

                npos_instances = self.data_obj.get_len_instance(train_p)
                nneg_instances = self.data_obj.get_len_instance(train_n)

                # the number of instances that I have to get is great than 
                # the number of instances in positive set, then the rest of 
                # instances will come from the negative set
                if npos_instances < (batch_size*rho):
                    len_batch_n = int((batch_size*rho) - npos_instances + batch_size*(1-rho))
                else:
                    len_batch_n = int(batch_size*(1-rho))

                if nneg_instances < (batch_size*(1-rho)):
                    len_batch_p = int( (batch_size*(1-rho)) - nneg_instances)
                else:
                    len_batch_p = int(batch_size*rho)
                
                # print("1 >> ", len_batch_p, len_batch_n)

                batch_p = self.get_ninstances(batch_p, train_p, len_batch_p)
                batch_n = self.get_ninstances(batch_n, train_n, len_batch_n)

                batch_train = self.join_instances(batch_p, batch_n)

                len_batch_p = self.data_obj.get_len_instance(batch_p)
                len_batch_n = self.data_obj.get_len_instance(batch_n)

                y_train = len_batch_p*[1] + len_batch_n*[-1]
                # print("2 >>> ", len_batch_p, len_batch_n)

                vec_train, vec_test = self.data_obj.vectorize_adhoc(batch_train, X_test)

                lr_clf.fit(vec_train, y_train)
                y_pred = lr_clf.predict(vec_test)
                #print(vec_train.shape, vec_test.shape)
                # print(y_pred)
                roc_auc = metrics.roc_auc_score(y_test, y_pred)
                f1_score = metrics.f1_score(y_test, y_pred)
                precision_score = metrics.precision_score(y_test, y_pred)
                recall_score = metrics.recall_score(y_test, y_pred)
                accuracy_score = metrics.accuracy_score(y_test, y_pred)
                auc_score = metrics.auc(y_test, y_pred)
                g_means = math.sqrt(precision_score*recall_score)
                # print(metrics.confusion_matrix(y_test, y_pred))
                print("%d, %d, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f" %  (count_cv, b, roc_auc, f1_score, precision_score, recall_score, accuracy_score, auc_score, g_means))
                b = b + batch_size
            self.data_obj.train_data = []
            self.data_obj.test_data = []
            count_cv = count_cv + 1

    def guided_learning_train_batch(self, input_data, max_batch_size, class_name = ''):
        """
        @input_data: if it is LEWISSPLIT split_type, then it is a string 
                      with train/test libsvm files separated by ","
                     if it is CROSSVALIDATION, then it is a directory with raw text
        """

        if self.split_type == 'CROSSVALIDATION':
            self.read_data(input_data)
            self.data_obj.shuffle_data()
            cv = 10
        elif self.split_type == 'LEWISSPLIT':
            input_train_file, input_test_file = input_data.split(",")
            X_train, y_train = self.load_test_data(input_train_file)
            X_test, y_test = self.load_test_data(input_test_file)
            cv = 1


        rho = 0.5
        epochs = 10
        b = 250 # batch size

        count_cv = 0
        lr_clf = linear_model.LogisticRegression()

        while count_cv < cv:

            print("Fold %d" % count_cv)
            print()

            if self.split_type == 'CROSSVALIDATION':
                batch_p = []
                batch_n = []

                # split train and test for the i-th fold
                t0 = time.time()
                self.data_obj.build_kfold(cv, count_cv)
                t1 = time.time()
                if self._verbose >= VERBOSE_MEDIUM:
                   print("Build kfold: ", t1-t0,"s")

                t0 = time.time()
                train_p, train_n, test_p, test_n = \
                self.data_obj.split_by_class(class_name)
                t1 = time.time()
                if self._verbose >= VERBOSE_MEDIUM:
                   print("Split by class: ", t1-t0,"s")
            else:                
                batch_p = None
                batch_n = None

                # split positive and negative in train fold data
                start = time.time()
                train_p, train_n = self.split_pos_neg(X_train, y_train)
                if self._verbose >= VERBOSE_MEDIUM:
                    print("Split dataset: ", time.time()-start, "s")

            if self.split_type == 'CROSSVALIDATION':
                self.data_obj.train_data = []
                self.data_obj.test_data = []

            # loop to run each bach in guided learning
            batch_size = b

            len_batch_p = int(b * rho)
            len_batch_n = int(b * (1 - rho))

            X_test = self.join_instances(test_p, test_n)

            ntest_n = self.data_obj.get_len_instance(test_n)
            ntest_p = self.data_obj.get_len_instance(test_p)
            y_test = ntest_p*[1] + ntest_n*[-1]

            while batch_size <= max_batch_size:
                npos_instances = self.data_obj.get_len_instance(train_p)
                nneg_instances = self.data_obj.get_len_instance(train_n)

                # the number of instances that I have to get is great than 
                # the number of instances in positive set, then the rest of 
                # instances will come from the negative set
                if npos_instances < (b*rho):
                    len_batch_n = int(len_batch_n + (b*rho) - npos_instances)
                if nneg_instances < (b*(1-rho)):
                    len_batch_p = int(len_batch_p + (b*(1-rho)) - nneg_instances)
                
                batch_p = self.get_ninstances(batch_p, train_p, len_batch_p)
                batch_n = self.get_ninstances(batch_n, train_n, len_batch_n)

                batch_train = self.join_instances(batch_p, batch_n)

                len_batch_p = self.data_obj.get_len_instance(batch_p)
                len_batch_n = self.data_obj.get_len_instance(batch_n)

                y = len_batch_p*[1] + len_batch_n*[-1]

                t0 = time.time()
                vec_batch_train, vec_test = self.data_obj.vectorize_adhoc(batch_train, X_test)
                t1 = time.time()
                print("Batch %d vectorize time: " % batch_size, t1-t0, "s")


                t0 = time.time()
                lr_clf.fit(vec_batch_train, y)
                print("N features: ", vec_batch_train.shape, vec_test.shape)
                t1 = time.time()
                if self._verbose >= VERBOSE_MEDIUM:
                    print("Batch %d Training time: " % batch_size, t1-t0, "s")

                t0 = time.time()
                y_pred = lr_clf.predict(vec_test)
                t1 = time.time()
                if self._verbose >= VERBOSE_MEDIUM:
                    print("Batch %d Testing time: " % batch_size, t1-t0, "s")
                print("Batch %s AUC: %.2f" % (batch_size, metrics.roc_auc_score(y_test, y_pred)))
                batch_size = batch_size + b
            count_cv = count_cv + 1
            print()

    def build_file_data(self, class_names, features_file):

        for c in class_names:
            t0 = time.time()
            train_p, train_n, test_p, test_n = self.data_obj.split_by_class(c)
            t1 = time.time()
            print("Split by class time: ", t1 - t0,"s")
            
            numntrain = train_n.shape[0]
            numptrain = train_p.shape[0]
            X = sparse.vstack([train_p, train_n])

            y = [1]*numptrain + [0]*numntrain
            if self.split_type == "CROSSVALIDATION":
                ftr_file_name = 'features/data_%s_%s' % (c, features_file)
            else:
                ftr_file_name = 'features/train_%s_%s' % (c, features_file)

            print('Writing %s...' % ftr_file_name)
            self.write_data(ftr_file_name, X, y)

            numntest = test_n.shape[0]
            numptest = test_p.shape[0]
            X =  sparse.vstack([test_p, test_n])
            y = [1]*numptest + [0]*numntest
            if self.split_type == "CROSSVALIDATION":
                ftr_file_name = 'features/data_%s_%s' % (c, features_file)
            else:
                ftr_file_name = 'features/test_%s_%s' % (c, features_file)
            print('Writing %s...' % ftr_file_name)
            self.write_data(ftr_file_name, X, y)

    def process_raw_data(self, dir_name, class_names, features_file):

        self.read_data(dir_name)

        t0 = time.time()
        self.data_obj.vectorize()
        t1 = time.time()
        print("Vectorize time: ", t1 - t0,"s")
        self.build_file_data( class_names, features_file)


    def perform_vec_data(self, data_file):

        X_train, y_train, X_test, y_test = self.load_data(data_file)

        # split positive and negative
        train_p, train_n = self.split_pos_neg(X_train, y_train)
        test_p, test_n = self.split_pos_neg(X_test, y_test)

        # active learning

        self.active_learning(train_p, train_n)

    def perform(self, dir_name = '', data_file = '', class_names  = []):
 
        
        if data_file == '' and dir_name != '':
            self.process_raw_data(dir_name, class_names, 'ftr_%s.lsvm' % self.vec_type)
        elif dir_name == '' and data_file != '':
            self.perform_vec_data(data_file)
        else:
            print('Something is wrong with your data entry...')
            return -1

            
    
if __name__ == '__main__':
    option = sys.argv[1]

    ## begin configuration parameters ##

    split_data = 'CROSSVALIDATION' # or LEWISSPLIT (default)
    # split_data = 'LEWISSPLIT' # (default)
    vec_type = 'tfidf' # bin, or tfidf, or count



    ## end configuration parameters ##

    e = Experiments(split_type = split_data, vec_type = vec_type)

    # class_names = ['corn', 'crude', 'grain', 'interest', 'money-fx', 'ship', 'wheat', 'trade']
    class_names = ['crude']
    # class_names = ['sci']
    if option == '-d':
        dir_name = sys.argv[2]
        e.perform(dir_name = dir_name, class_names = class_names)
    elif option == '-f':
        data_file = sys.argv[2]
        e.perform(data_file = data_file)
    elif option ==  '-gl':
        input_data = sys.argv[2]
        max_iter_batch = int(sys.argv[3])
        class_name = 'sci'
        e.guided_learning_train(input_data, max_iter_batch, class_name)
    elif option == '-gt':
        model_dir = sys.argv[2]
        test_file = sys.argv[3]
        max_iter_batch = int(sys.argv[4])
        npos = int(sys.argv[5])
        nneg = int(sys.argv[6])
        
        e.set_idx_fold(int(sys.argv[7]))
        e.guided_learning(model_dir, test_file, max_iter_batch, npos, nneg)

