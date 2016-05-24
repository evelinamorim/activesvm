import sys
import os
from bs4 import BeautifulSoup
import nltk
import sklearn
import numpy as np
import scipy
from scipy import sparse
import re
import random
import math

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import cross_validation
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

stemmer = PorterStemmer()
stop_list = stopwords.words('english')

num = re.compile('\d+')
random.seed(179426321)

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

class Data:

    def __init__(self):

        self.cat = {}
        self.text = ""
        self.lewis_split = None
        self.topics = None
        self.file_name = ""


class DataList:

    def __init__(self, nfeatures, split = 'LEWISSPLIT', vec_type = 'tfidf'):
        self.data_list = []
        self.labels = []
        self.nfeatures = nfeatures

        self.train_data = []
        self.test_data = []

        self.split = split
        self.vec_type = vec_type   

        self._verbose = 0

    def set_verbose_level(self, num):
        self._verbose = num

    def set_split(self, split):
        self.split = split

    def get_len_data(self):
        return len(self.data_list)

    def get_len_instance(self, i):

        if type(i) == list:
            return len(i)
        else:
            return i.shape[0]

    def remove_ninstances(self, class_name, r):
        """
         Randomly remove  instance of a class (class_name) according to the 
         ratio r (0 < r < 1)
        """

        class_data = []
        not_class_data = []

        for d in self.data_list:
            if class_name in d.cat['topics']:
                class_data.append(d)
            else:
                not_class_data.append(d)

        k = math.ceil((1-r)*len(class_data))
        self.data_list = not_class_data + random.sample(class_data, k)

    def build_kfold(self, cv, idxfold):

        ninstances = len(self.data_list)
        len_fold = math.ceil(ninstances / cv)
        begin_test = idxfold*len_fold
        end_test = min(begin_test + len_fold, ninstances)
        idx = 0
        while idx < ninstances:
            if idx >= begin_test and idx < end_test:
                self.test_data.append(self.data_list[idx])
            else:
                self.train_data.append(self.data_list[idx])

            idx = idx + 1 
     
        self.labels = [f.cat['topics'] for f in self.train_data] +\
                      [f.cat['topics'] for f in self.test_data]

        ####
        # y = []
        # for l in self.labels:
        #  if l == 'sci':
        #      y.append(1)
        #  else:
        #      y.append(-1)

        ####
        # self.train_data_tmp = self.train_data
        # self.test_data_tmp = self.test_data

        self.train_data = [f.text for f in self.train_data]
        self.test_data = [f.text for f in self.test_data]

        ######begin########
        # data = self.train_data + self.test_data
        # self.train_data, self.test_data, self.y_train, self.y_test = cross_validation.train_test_split(data, y, test_size = 0.1)
        ######end########

    def read(self, file_name):

        path, f = os.path.split(file_name)

        if file_name.endswith("sgm"):
            self.readsgm(file_name)
        elif num.match(f):
            self.readtext(file_name)
        # else:
        #    print("File format not valid.")

    def readtext(self, file_name):

        d = Data()
        d.text = open(file_name, "r").read()

        lst_path = file_name.split(os.sep)

        d.cat['topics'] = lst_path[len(lst_path)-2].split('.')[0]
        self.data_list.append(d)
        d.file_name = file_name

    def readsgm(self, file_name):

        soup = BeautifulSoup(open(file_name, 'rb').read(), "lxml")

        elem_lst = soup.find_all("reuters")

        for e in elem_lst:
            d = Data()
            d.text = e.text

            d.lewis_split = e['lewissplit']
            d.topics = e['topics']

            d.cat['exchanges'] = e.exchanges.find_all('d') 
            d.cat['exchanges'] = [k.get_text() for k in d.cat['exchanges']]

            d.cat['orgs'] = e.orgs.find_all('d')
            d.cat['orgs'] = [k.get_text() for k in d.cat['orgs']]

            d.cat['people'] = e.people.find_all('d') 
            d.cat['people'] = [k.get_text() for k in d.cat['people']]

            d.cat['places'] = e.places.find_all('d')
            d.cat['places'] = [k.get_text() for k in d.cat['places']]

            d.cat['topics'] = e.topics.find_all('d') 
            d.cat['topics'] = [k.get_text() for k in d.cat['topics']]

            self.data_list.append(d)

    def split_by_class_list(self, class_name = ''):

        
        ntrain = len(self.train_data)
        nlabels = len(self.labels)


        positive_train = []
        negative_train = []

        positive_test = []
        negative_test = []

        self.train_pos = []
        self.train_neg = []

        self.test_pos = []
        self.test_neg = []

        k = 0
        l = 0
        for i in range(nlabels):

            if class_name in self.labels[i]:
                if i >= ntrain:
                    # print("1")
                    # print("FILE: ", self.test_data_tmp[k].file_name)
                    # print("TEXT: ", self.test_data_tmp[k].text)
                    # print("CAT: ", self.test_data_tmp[k].cat['topics'])
                    # print("labels[i]: ", self.labels[i])
                    # print()
                    positive_test.append(self.test_data[k])
                    k = k + 1
                else:
                    # print("2")
                    # print("FILE: ", self.train_data_tmp[l].file_name)
                    # print("TEXT: ", self.train_data_tmp[l].text)
                    # print("CAT: ", self.train_data_tmp[l].cat['topics'])
                    # print("labels[i]: ", self.labels[i])
                    # print()
                    positive_train.append(self.train_data[l])
                    l = l + 1
            else:
                if i>= ntrain:
                    #print("3")
                    #print("FILE: ", self.test_data_tmp[k].file_name)
                    #print("TEXT: ", self.test_data_tmp[k].text)
                    #print("CAT: ", self.test_data_tmp[k].cat['topics'])
                    #print("labels[i]: ", self.labels[i])
                    #print()
                    negative_test.append(self.test_data[k])
                    k = k + 1
                else:
                    # print("4")
                    # print("FILE: ", self.train_data_tmp[l].file_name, len(negative_train))
                    # print("TEXT: ", self.train_data_tmp[l].text)
                    # print("CAT: ", self.train_data_tmp[l].cat['topics'])
                    # print("labels[i]: ", self.labels[i])
                    # print()
                    negative_train.append(self.train_data[l])
                    l = l + 1

        return positive_train, negative_train, positive_test, negative_test 

    def split_by_class(self, class_name = ''):
        """
        Given a class name, split data according to that class in 
        two classes: positive and negative
        """

        try:
            ntrain = self.train_data.shape[0]
        except AttributeError:
            return self.split_by_class_list(class_name)
        nlabels = len(self.labels)

        positive_train = None
        negative_train = None

        positive_test = None
        negative_test = None

        k = 0
        for i in range(nlabels):

            # print(">>>", class_name, self.labels[i], class_name == self.labels[i])
            if class_name in self.labels[i]:
                if i < ntrain:
                    if positive_train == None:
                        positive_train = self.train_data[k]
                    else:
                        positive_train = sparse.vstack([positive_train, self.train_data[k]])
                else:
                    if positive_test == None:
                        k = 0
                        positive_test = self.test_data[k]
                    else: 
                        positive_test =  sparse.vstack([positive_test, self.test_data[k]])
            else:
                if i < ntrain:
                    if negative_train == None:
                        negative_train = self.train_data[k]
                    else:
                        negative_train = sparse.vstack([negative_train, self.train_data[k]])
                    # negative_train.append(self.train_data[k])
                else:
                    if negative_test == None:
                        k = 0
                        negative_test = self.test_data[k]
                    else:
                        negative_test =  sparse.vstack([negative_test, self.test_data[k]])

            k = k + 1
        # (389, 8958) (9214, 8958) (189, 8958) (3110, 8958)
        # print('>>>> ', positive_train.shape, negative_train.shape, positive_test.shape, negative_test.shape)
        return positive_train, negative_train, positive_test, negative_test 

    def shuffle_data(self):
        random.shuffle(self.data_list)

    def build_fold(self, idx_fold, nfold):

        ninstances = len(self.data_list)
        tam_fold = int(ninstances/nfold)

        begin = idx_fold*tam_fold
        end = idx_fold*tam_fold + tam_fold

        self.train_data = self.data_list[0:begin] + self.data_list[end:ninstances]
        self.test_data = self.data_list[begin:end]

    def get_train_test(self):

        """
        get train and test in ModApte
        """

        train_data = []
        test_data = []


        if self.split == 'CROSSVALIDATION':
            train_data = self.train_data
            test_data = self.test_data

        elif self.split == 'LEWISSPLIT':
            for d in self.data_list:

                if d.topics == 'YES':
                    if d.lewis_split == 'TRAIN':
                        train_data.append(d)
                    elif d.lewis_split == 'TEST':
                        test_data.append(d)

        return train_data, test_data

    def vectorize_adhoc(self, X_train, X_test):


        # vectorizer by tf idf model    
        vec_model = None
        if self.vec_type == 'tfidf':
            vec_model = TfidfVectorizer(sublinear_tf=True, stop_words='english', max_df=0.5)
        elif self.vec_type == 'count':
            vec_model = CountVectorizer(tokenizer=tokenize, stop_words='english', min_df=5)
        elif self.vec_type == 'bin':
            vec_model = CountVectorizer(tokenizer=tokenize, stop_words='english', min_df=5, binary=True)

        vec_train = vec_model.fit_transform(X_train)
        vec_test = vec_model.transform(X_test)
 
        return vec_train, vec_test

    def vectorize(self):

        train_data, test_data = self.get_train_test()
        self.labels = [f.cat['topics'] for f in train_data] +\
                      [f.cat['topics'] for f in test_data]

        train_data = [f.text for f in train_data]
        test_data = [f.text for f in test_data]

        # vectorizer by tf idf model    
        # acho que eu tenho que colocar todas as palavras no mesmo modelo tfidf, certo?
        vec_model = None
        if self.vec_type == 'tfidf':
            # vec_model = tfidfvectorizer(tokenizer=tokenize, stop_words='english', min_df=5, max_features=1000)
            # vec_model = TfidfVectorizer(sublinear_tf=True, stop_words='english', max_df=0.5)
            vec_model = TfidfVectorizer(sublinear_tf=True, stop_words='english', max_df=0.5)
        elif self.vec_type == 'count':
            vec_model = CountVectorizer(tokenizer=tokenize, stop_words='english', min_df=5)
        elif self.vec_type == 'bin':
            vec_model = CountVectorizer(tokenizer=tokenize, stop_words='english', min_df=5, binary=True)

        self.train_data = vec_model.fit_transform(train_data)
        self.test_data = vec_model.transform(test_data)
        # print(">>>", self.train_data.shape, self.test_data.shape)
  
    def write_data(self, out_file, X, y):
    
        sklearn.datasets.dump_svmlight_file(X, y, out_file)

    def load_test_data(self, input_file):

        # test_file  = "features/test_%s" % input_file

        data_test = sklearn.datasets.load_svmlight_file(input_file)
        return data_test[0], data_test[1]

    def load_data(self, input_file):

        test_file  = "features/test_%s" % input_file
        train_file = "features/train_%s" % input_file

        data_test = sklearn.datasets.load_svmlight_file(test_file)
        data_train = sklearn.datasets.load_svmlight_file(train_file)
        return data_train[0], data_train[1], data_test[0], data_test[1]
