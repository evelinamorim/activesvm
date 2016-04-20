import sys
import os
from bs4 import BeautifulSoup
import nltk
import sklearn
import numpy as np
import scipy
from scipy import sparse

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

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


class DataList:

    def __init__(self, nfeatures):
        self.data_list = []
        self.labels = []
        self.nfeatures = nfeatures

        self.train_data = []
        self.test_data = []

    def read(self, file_name):

        if file_name.endswith("sgm"):
            self.readsgm(file_name)
        # else:
        #    print("File format not valid.")

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

    def split_by_class(self, class_name = ''):
        """
        Given a class name, split data according to that class in 
        two classes: positive and negative
        """
        
        ntrain = self.train_data.shape[0]
        nlabels = len(self.labels)

        positive_train = None
        negative_train = None

        positive_test = None
        negative_test = None


        k = 0
        for i in range(nlabels):

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

        return positive_train, negative_train, positive_test, negative_test 

    def get_train_test(self):

        """
        get train and test in ModApte
        """

        train_data = []
        test_data = []

        for d in self.data_list:

            if d.topics == 'YES':
                if d.lewis_split == 'TRAIN':
                    train_data.append(d)
                elif d.lewis_split == 'TEST':
                    test_data.append(d)

        return train_data, test_data

    def vectorize(self):
        # TODO : still specific to topic model classification 

        train_data, test_data = self.get_train_test()

        self.labels = [f.cat['topics'] for f in train_data] +\
                      [f.cat['topics'] for f in test_data]

        train_data = [f.text for f in train_data]
        test_data = [f.text for f in test_data]

        # vectorizer by tf idf model    
        # acho que eu tenho que colocar todas as palavras no mesmo modelo tfidf, certo?
        tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english', min_df=3)

        self.train_data = tfidf.fit_transform(train_data)
        # self.test_data = tfidf.fit_transform(test_data)
        self.test_data = self.train_data

        # print(self.nfeatures, self.train_data.shape, self.test_data.shape)
        # 10000 (9603, 14056) (3299, 6733)
        # self.train_data = sparse.lil_matrix(sparse.csr_matrix(self.train_data)[:,range(self.nfeatures)])
        # self.test_data = sparse.lil_matrix(sparse.csr_matrix(self.test_data)[:,range(self.nfeatures)])
  
    def write_data(self, out_file, X, y):
    
        sklearn.datasets.dump_svmlight_file(X, y, out_file)

    def load_data(self, input_file):

        test_file  = "features/test_%s" % input_file
        train_file = "features/train_%s" % input_file

        data_test = sklearn.datasets.load_svmlight_file(test_file)
        data_train = sklearn.datasets.load_svmlight_file(train_file)
        return data_train[0], data_train[1], data_test[0], data_test[1]
