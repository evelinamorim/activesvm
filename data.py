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
import json
import string
import time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import cross_validation, feature_extraction
from nltk.corpus import stopwords

import utils
from argument import ArgumentRelation, Argument


NUM_PATTERN = re.compile('\d+')
REASON_ID_PATTERN = re.compile('\d+\_\d+')
random.seed(179426321)


def tokenize(text):
   
    tokens = nltk.word_tokenize(text.lower())
    # tokens = [t for t in tokens if t not in stop_list]
    # stems = stem_tokens(tokens, stemmer)
    return tokens


class Data:

    def __init__(self):

        self.cat = {}
        self.text = ""
        self.text2 = "" # a secondary text in document to store
        self.lewis_split = None
        self.topics = None
        self.file_name = ""
        self.id = {}
        self.features = {}

    def add_features(self, name, value):
        self.features[name] = value

class DataList:

    def __init__(self, split = 'LEWISSPLIT', vec_type = 'tfidf'):
        self.data_list = []
        self.labels = []
        # self.nfeatures = nfeatures

        self.train_data = []
        self.test_data = []

        self.split = split
        self.vec_type = vec_type   

        self._verbose = 0

        self._vocabulary = {}
        self._inv_vocabulary = {}

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

    def make_cross_product(self, tok1, tok2):
        bigram_list = []
        # nvoc = len(self._vocabulary.keys())

        for w1 in tok1:
            for w2 in tok2:
                bigram_list.append((w1, w2))

        return bigram_list

    def make_word_pairs(self, relations, data_rel):

        k = 0 
        for r in relations:
            tok_prop1, tok_prop2 = r.get_arguments_tokens()
            d = Data()
            word_pairs = self.make_cross_product(tok_prop1, tok_prop2)
            d.add_features('word_pairs', word_pairs)
            data_rel.append(d)
            k = k + 1

    def first_last_first3(argrel, relations, data_rel):

        k = 0
        for r in relations:
            tok_prop1, tok_prop2 = r.get_arguments_tokens()
            first_prop1 = tok_prop1[0]
            first_3_prop1 = tok_prop1[:3]
            last_prop1 = tok_prop1[-1]

            first_prop2 = tok_prop2[0]
            first_3_prop2 = tok_prop2[:3]
            last_prop2 = tok_prop2[-1]

            data_rel[k].add_features('first_prop1', first_prop1)
            data_rel[k].add_features('first_3_prop1', first_3_prop1)
            data_rel[k].add_features('last_prop1', last_prop1)

            data_rel[k].add_features('first_prop2', first_prop2)
            data_rel[k].add_features('first_3_prop2', first_3_prop2)
            data_rel[k].add_features('last_prop2', last_prop2)
            k = k + 1
       

    def verb_features(argrel, relations, data_rel):

        k = 0
        # import pdb
        # pdb.set_trace()
        for r in relations:

 
            # first verb feature: counting of verbs that are in the same Levin 
            # English class
            levin_pairs = r.get_levin_pairs(verbs_prop1, verbs_prop2)
            data_rel[k].add_features('count_levin', len(levin_pairs))

            r.arguments[1].build_chunk()
            r.arguments[2].build_chunk()

            vp_prop1 = r.arguments[1].get_verb_phrases()
            vp_prop2 = r.arguments[2].get_verb_phrases()

            # second verb feature: the average length of verb phrases
            avg_vp_len = 0
            nvp = len(vp_prop1)
            tok_vp1 = []
            for vp1 in vp_prop1:
                avg_vp_len = avg_vp_len + len(vp1)
                tok_vp1.extend(vp1.leaves())

            nvp = nvp + len(vp_prop2)
            tok_vp2 = []
            for vp2 in vp_prop2:
                avg_vp_len = avg_vp_len + len(vp2)
                tok_vp2.extend(vp2.leaves())

            data_rel[k].add_features('avg_vp_len', avg_vp_len / nvp)

            # third verb feature: cross product of verb phrases
            vp_cross_product = self.make_cross_product(tok_vp1, tok_vp2)

            data_rel[k].add_features('vp_cross_product', vp_cros_product)

            # fourth verb feature: pos tag of main verb for each argument
            r.arguments[1].build_deptree()
            main_verb_arg1 = r.arguments[1].get_main_verb()

            r.arguments[2].build_deptree() 
            main_verb_arg2 = r.arguments[2].get_main_verb()
            k = k + 1


    def make_relations(self, i, arguments):

        reason_i = arguments[i].reason
        tok_i = arguments[i].reason
        relations = []

        if reason_i is not None:
            for r in reason_i:

                rel = ArgumentRelation()
                rel.add_argument(arguments[i])
                rel.set_connection()

                if REASON_ID_PATTERN.match(r):
                    reasons = r.split("_")
                    new_argument = Argument()
                    new_argument.tokens = []
                    new_argument.text = ""
                    new_argument.posTagging = []

                    for c in reasons:
                        new_argument.text = " " + propositions[int(c)].tokens
                        new_argument.tokens.extend(propositions[int(c)].tokens)
                        new_argument.posTagging.extend(propositions[int(c)].posTagging)

                    rel.add_argument(new_argument)

                elif NUM_PATTERN.match(r):
                    rel.add_argument(propositions[int(r)])

                relations.append(rel)
        else:
            for p in propositions:
                if p != i:
                   rel = ArgumentRelation()
                   rel.add_argument(propositions[i])
                   rel.add_argument(propostions[int(p)])
                   rel.unset_connection()

                   relations.append(rel)

        return relations

    def extract_features(self, propositions):

        output_data = []
        # for each proposition p in a propositions set, build relations for p 
        # considering propositions of the given propositions set
        for p in propositions:
            relations = self.make_relations(p, propositions)
            data_rel = []
            self.make_word_pairs(relations, data_rel)
            self.first_last_first3(relations, data_rel)
            self.verb_features(relations, data_rel)
            output_data.extend(data_rel)

        return output_data

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

    def build_kfold_dict(self, cv, idxfold):

        ninstances = len(self.data_list.keys())
        len_fold = math.ceil(ninstances / cv)
        begin_test = idxfold*len_fold
        end_test = min(begin_test + len_fold, ninstances)
        idx = 0

        test_data = {}
        train_data = {}

        # separando o treino o teste e enquanto isso tokeniza cada proposicao
        # em cada argumento
        for k in self.data_list:
            if idx >= begin_test and idx < end_test:
                test_data[k] = self.process_argument(self.data_list[k])
            else:
                train_data[k] = self.process_argument(self.data_list[k])

            idx = idx + 1 
     
        print(len(test_data), len(train_data))

        # agora aqui eh feita uma lista de bigramas para cada argumento...
        #...primeiro no conjunto de treino...
        self.train_data = []
        for k in train_data:
             self.train_data.extend(self.extract_features(train_data[k]))


        #...depois no conjunto de teste...
        self.test_data = []
        for k in test_data:
             self.test_data.extend(self.extract_features(test_data[k]))


    def build_kfold(self, cv, idxfold):

        if type(self.data_list) == dict:
            self.build_kfold_dict(cv, idxfold)
        else:
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

    def process_argument(self, arg):
        """
        Given a argument as a dictionary type, it appends the pairs of reasons  
        to the list of data_list. Each pair is a Data object.
        """
        
        # iterating through propositions
        arg_dict = {}
        for p in arg:
            id_p = p["id"]
            text_p = p["text"]
            reason_p = p["reasons"]

            arg_dict[id_p] = Argument(text_p, reason_p)
            arg_dict.process()

        return arg_dict    

    def read(self, file_name):

        path, f = os.path.split(file_name)

        if file_name.endswith("sgm"):
            self.readsgm(file_name)
        elif file_name.endswith("jsonlist"):
            self.readjson(file_name)
        elif NUM_PATTERN.match(f):
            self.readtext(file_name)
        # else:
        #    print("File format not valid.")

    def readjson(self, file_name):

        # TODO: separar treino e teste -> supor validacao cruzada
        json_fd = open(file_name, "rb")
        self.data_list = {}
        for line in json_fd:
            d = json.loads(line.decode('UTF-8'))
            self.data_list[d["commentID"]] = d["propositions"]
        json_fd.close()

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

    def vectorize_dict(self):

        nvoc = len(self._vocabulary.keys())
        train_vec = []

        for d in self.train_data:
            bigram_list = []
            for (w1,w2) in d:
                if  (w1,w2) not in self._vocabulary:
                    if (w2, w1) not in self._vocabulary:
                        self._vocabulary[(w1,w2)] = nvoc    
                        nvoc = nvoc + 1
                    else:
                        bigram_list.append(self._vocabulary[(w2, w1)])
                else:
                    bigram_list.append(self._vocabulary[(w1, w2)])

            train_vec.append(bigram_list)

        dim_data = len(self._vocabulary)
        self.train_data = utils.create_matrix_csr(train_vec, dim_data)

        # sera que posso customizar o CountVectorizer para fazer o produto das frases?
        test_vec = []
        for d in self.test_data:
            bigram_list = []
            for (w1,w2) in d:
                if  (w1,w2) in self._vocabulary:
                    bigram_list.append(self._vocabulary[(w1, w2)])
                else:
                    if (w2, w1) in self._vocabulary:
                        bigram_list.append(self._vocabulary[(w2, w1)])

            test_vec.append(bigram_list)

        self.test_data = utils.create_matrix_csr(test_vec, dim_data)
        #print(len(self.test_data), len(self._vocabulary))

    def vectorize(self):

        if type(self.data_list) == dict:
            self.vectorize_dict()
        else:
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

if __name__ == "__main__":
    d = DataList()
    d.read('/Users/evelin.amorim/Documents/Cornell/ArgRel/apr1.jsonlist')
    d.build_kfold(10, 0)
    # d.vectorize() # TODO: fix according to the new featuredata structure

    # ninstances = d.train_data.shape[0]
    # y = d.labels[:ninstances]   
    # d.write_data('argrel_sample.libsvm')
