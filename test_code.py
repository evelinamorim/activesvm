# python script to test code
import numpy as np
import scipy
from scipy import sparse
import arff

from sklearn.datasets import load_svmlight_file
from sklearn import  metrics
import matplotlib.pyplot as plt
from scipy.interpolate import spline

import sys
import os
import re

num = re.compile('\d+')
# Create a canvas to place the subgraphs
canvas = plt.figure()

def count_pos_neg(file_name):

    fd = open(file_name, 'r')
    p = 0
    n = 0
    for l in fd:
        ftrs = l.split()
    
        c = ftrs[0]
        if c == '-1':
            n = n+1
        else:
            p = p+1
    print(n, p)
    fd.close()

def get_batch(name):

    b = ''
    nnames = len(name) - 1    

    for i in range(nnames, -1, -1):
       if name[i] == 't':
           break

       b = name[i] + b

    return b

def get_output_y(fname):
    y = []
    fd = open(fname, "r")
    for line in fd:
       y.append(float(line))

    fd.close()
    return y

def libsvm2arff(input_file, out_file):

    X, y = load_svmlight_file(input_file)
    l,c = X.shape
    data = np.zeros((l,c+1))
    data[:,:-1] = X.toarray()
    data[:,c] = y

    arff.dump(out_file, data)

def compute_auc(ftr_file, output_dir):

    data = load_svmlight_file(ftr_file)

    y = data[1]

    output_results = []

    for f in os.listdir(output_dir):
        if f.endswith('.txt'):
            fname_path = os.path.join(output_dir, f)
            filename, file_extension = os.path.splitext(f)
            batch = get_batch(filename)
            if num.match(batch):
                batch_num = int(batch)
                output_results.append((fname_path, batch_num))

    output_results = sorted(output_results, key=lambda x:x[1])

    auc_values = []
    num_instances = []
    for (f, b) in output_results:
        scores = get_output_y(f)
        fpr, tpr, thresholds = metrics.roc_curve(y, scores)
        roc_auc = metrics.auc(fpr, tpr)
        auc_values.append(roc_auc)
        num_instances.append(b)

    x_smooth = np.linspace(min(num_instances), max(num_instances), 200)
    y_smooth = spline(num_instances, auc_values, x_smooth)

    sp1 = canvas.add_subplot(1,1,1, axisbg='w')
    sp1.plot(x_smooth, y_smooth, 'blue', linewidth=1)

    sp1.set_xlabel('#instances')
    sp1.set_ylabel('AUC')
    sp1.set_title('Crude dataset')
    plt.savefig("crude_lasvm.png")
    plt.show()

# compute_auc("features/test_crude.lsvm", "output_svm/")
# count_pos_neg(sys.argv[1])
input_file = sys.argv[1]
out_file = sys.argv[2]
libsvm2arff(input_file, out_file)    
